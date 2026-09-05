from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from backend import config
from backend.models import MerchantOrder, PaymentAttempt, RecoveryAction

logger = logging.getLogger(__name__)


def _record_recovery_action(db: Session, order: MerchantOrder, action_type: str, reason: str, attempt_number: int, status: str = "active", case_status: str | None = None) -> RecoveryAction:
    action = RecoveryAction(
        order_id=order.internal_id,
        action_type=action_type,
        reason=reason,
        attempt_number=attempt_number,
        status=status,
        case_status=case_status or status,
        created_at=datetime.utcnow(),
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def _historical_recovery_order_ids(db: Session) -> set[int]:
    rows = (
        db.query(RecoveryAction.order_id)
        .filter(RecoveryAction.action_type.in_({"recovery_candidate", "abandonment_candidate", "retry_payment", "recovery_retry_pending", "recovery_waiting", "recovery_stop", "recovery_success", "recovery_exhausted"}))
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def evaluate_recovery_policy(db: Session, order: MerchantOrder) -> tuple[bool, str, int | None]:
    """Return (eligible, reason, attempt_number).

    The deterministic baseline counts failed/recovery attempt states for a single
    Razorpay order. The first failed payment is recovery attempt 1, and the order
    stops being eligible once the attempt count reaches the configured maximum.
    """
    if order.status == "paid":
        return False, "order already paid", None

    if order.status == "abandoned":
        existing = db.query(RecoveryAction).filter(
            RecoveryAction.order_id == order.internal_id,
            RecoveryAction.action_type == "abandonment_candidate",
        ).first()
        if existing:
            return True, "checkout was abandoned; recovery remains available", 1
        return True, "checkout was abandoned before any payment attempt", 1

    payment_attempts = db.query(PaymentAttempt).filter(PaymentAttempt.razorpay_order_id == order.razorpay_order_id).all()
    failed_payment_count = sum(
        1
        for attempt in payment_attempts
        if attempt.status in {"failed", "attempt_failed", "recovery_retry", "recovered"}
    )

    if not failed_payment_count:
        return False, "no failed payment attempts to recover", None
    recovery_interventions_so_far = max(0, failed_payment_count - 1)
    if recovery_interventions_so_far >= config.MAX_RECOVERY_ATTEMPTS:
        return False, "max recovery interventions reached", recovery_interventions_so_far
    return True, "eligible for retry against existing Razorpay order", recovery_interventions_so_far + 1


def mark_order_recoverable(db: Session, order: MerchantOrder) -> RecoveryAction:
    from ml.decision_engine import AIDecisionEngine
    
    eligible, reason, attempt_number = evaluate_recovery_policy(db, order)
    if not eligible:
        action = _record_recovery_action(db, order, "recovery_exhausted", reason, attempt_number or 0, status="exhausted", case_status="exhausted")
        order.status = "exhausted"
        db.commit()
        return action

    # Run AI evaluation
    latest_attempt = (
        db.query(PaymentAttempt)
        .filter(PaymentAttempt.razorpay_order_id == order.razorpay_order_id)
        .order_by(PaymentAttempt.attempt_number.desc())
        .first()
    )

    case_dict = {
        "order_id": order.razorpay_order_id,
        "amount": order.amount,
        "customer_id": order.customer_id,
        "payment_method": (latest_attempt.method if latest_attempt and latest_attempt.method else "card"),
        "failure_code": (latest_attempt.failure_code if latest_attempt and latest_attempt.failure_code else "PAYMENT_ERROR"),
        "attempt_number": attempt_number or 1,
        "time_since_failure": 0,
        "previous_success_rate": 0.5,
        "previous_failed_attempts": attempt_number - 1 if attempt_number else 0,
        "customer_history": "stable",
        "already_paid": order.status == "paid",
        "max_attempts": config.MAX_RECOVERY_ATTEMPTS,
    }

    engine = AIDecisionEngine()
    outcome = engine.decide(case_dict)
    
    # Store the decision in the action
    if outcome.decision == "RETRY":
        action_type = "recovery_retry_pending"
        status = "active"
    elif outcome.decision == "WAIT":
        action_type = "recovery_waiting"
        status = "waiting"
    else:
        action_type = "recovery_stop"
        status = "stopped"

    # Save to db
    action = _record_recovery_action(
        db, 
        order, 
        action_type=action_type, 
        reason=outcome.reason, 
        attempt_number=attempt_number or 1, 
        status=status, 
        case_status=status
    )
    
    # Add AI explanation context to the action
    action.recovery_probability = str(outcome.recovery_probability)
    action.expected_recovered_value = str(outcome.expected_recovered_value)
    
    # We will serialize net_ev and other metrics into `error` (a Text field)
    import json
    action.error = json.dumps({
        "net_expected_value": outcome.net_expected_value,
        "intervention_cost": outcome.intervention_cost,
        "ai_decision": outcome.decision
    })
    
    if outcome.decision in {"RETRY", "WAIT"}:
        order.status = "recoverable"
    else:
        order.status = "exhausted"

    db.commit()
    return action


def mark_order_abandoned(db: Session, order: MerchantOrder) -> RecoveryAction:
    """Create an abandonment candidate without fabricating a payment attempt."""
    existing = db.query(RecoveryAction).filter(
        RecoveryAction.order_id == order.internal_id,
        RecoveryAction.action_type == "abandonment_candidate",
    ).first()
    if existing:
        return existing
    order.status = "abandoned"
    action = _record_recovery_action(
        db,
        order,
        "abandonment_candidate",
        "checkout was abandoned before a payment attempt",
        1,
        status="active",
        case_status="active",
    )
    db.commit()
    return action


def register_retry_attempt(db: Session, order: MerchantOrder) -> RecoveryAction:
    eligible, reason, attempt_number = evaluate_recovery_policy(db, order)
    if not eligible:
        action = _record_recovery_action(db, order, "recovery_exhausted", reason, attempt_number or 0, status="exhausted", case_status="exhausted")
        order.status = "exhausted"
        db.commit()
        logger.info("Retry rejected for order=%s because=%s", order.razorpay_order_id, reason)
        return action

    action = _record_recovery_action(db, order, "retry_payment", reason, attempt_number or 1, status="active", case_status="active")
    logger.info("Retry recorded for order=%s as recovery action_id=%s", order.razorpay_order_id, action.id)
    db.commit()
    return action


def mark_recovery_success(db: Session, order: MerchantOrder, attempt_number: int | None = None) -> RecoveryAction:
    existing_success = (
        db.query(RecoveryAction)
        .filter(RecoveryAction.order_id == order.internal_id, RecoveryAction.action_type == "recovery_success")
        .first()
    )
    if existing_success:
        order.status = "paid"
        db.commit()
        return existing_success

    db.query(RecoveryAction).filter(
        RecoveryAction.order_id == order.internal_id,
        RecoveryAction.status.in_({"pending", "waiting", "active"}),
    ).update({"status": "success", "case_status": "recovered", "completed_at": datetime.utcnow()})

    action = _record_recovery_action(db, order, "recovery_success", "payment recovered successfully", attempt_number or 1, status="success", case_status="recovered")
    order.status = "paid"
    db.commit()
    return action


def compute_dashboard_metrics(db: Session) -> dict[str, Any]:
    orders = db.query(MerchantOrder).all()
    total_orders = len(orders)
    total_expected_revenue = sum(order.amount for order in orders)
    successful_orders = sum(1 for order in orders if order.status == "paid")

    historical_case_ids = _historical_recovery_order_ids(db)
    historical_case_orders = db.query(MerchantOrder).filter(MerchantOrder.internal_id.in_(historical_case_ids)).all()
    recovery_cases_history = len(historical_case_orders)
    active_recovery_candidates = db.query(MerchantOrder).filter(MerchantOrder.status == "recoverable").count()

    failed_payment_attempts = db.query(PaymentAttempt).filter(PaymentAttempt.status == "failed").count()
    recovery_attempts = db.query(RecoveryAction).filter(
        RecoveryAction.action_type.in_({"retry_payment", "recovery_retry_pending"})
    ).count()

    recovered_order_ids = (
        db.query(RecoveryAction.order_id)
        .filter(RecoveryAction.action_type == "recovery_success")
        .distinct()
        .all()
    )
    recovered_order_ids = {row[0] for row in recovered_order_ids}
    recovered_orders = len(recovered_order_ids)
    recovered_revenue = sum(
        order.amount for order in orders if order.internal_id in recovered_order_ids
    )

    exhausted_cases = db.query(RecoveryAction).filter(RecoveryAction.action_type == "recovery_exhausted").count()
    revenue_identified_as_recoverable = sum(order.amount for order in historical_case_orders)
    
    # Revenue at risk is the revenue of candidates minus what we've already recovered
    revenue_at_risk = max(0, revenue_identified_as_recoverable - recovered_revenue)
    
    recovery_rate = (recovered_orders / recovery_cases_history) if recovery_cases_history else 0.0
    revenue_recovery_rate = (recovered_revenue / revenue_identified_as_recoverable) if revenue_identified_as_recoverable else 0.0

    return {
        "total_orders": total_orders,
        "total_expected_revenue": total_expected_revenue,
        "revenue_at_risk": revenue_at_risk,
        "successful_orders": successful_orders,
        "failed_payment_attempts": failed_payment_attempts,
        "recovery_candidates": recovery_cases_history,
        "recovery_cases_history": recovery_cases_history,
        "active_recovery_candidates": active_recovery_candidates,
        "recovery_attempts": recovery_attempts,
        "recovered_orders": recovered_orders,
        "recovered_revenue": recovered_revenue,
        "recovery_rate": recovery_rate,
        "revenue_recovery_rate": revenue_recovery_rate,
        "exhausted_escalated_cases": exhausted_cases,
    }
