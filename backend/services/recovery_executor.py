from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from backend import config
from backend.models import MerchantOrder, RecoveryAction
from backend.services.recovery import evaluate_recovery_policy


class InvalidRecoveryTransition(ValueError):
    """Raised when an execution request cannot advance the recovery case."""


@dataclass(frozen=True)
class RecoveryExecution:
    action: str
    status: str
    outcome: str
    reason: str
    recovery_action_id: int | None
    checkout_required: bool = False


def _existing_execution(
    db: Session,
    order: MerchantOrder,
    action: str,
    attempt_number: int,
    planned_method: str | None = None,
) -> RecoveryAction | None:
    return (
        db.query(RecoveryAction)
        .filter(
            RecoveryAction.order_id == order.internal_id,
            RecoveryAction.action_type == action,
            RecoveryAction.attempt_number == attempt_number,
            RecoveryAction.status.in_({"pending", "waiting", "stopped", "success"}),
        )
        .order_by(RecoveryAction.id.desc())
        .first()
    )


def _record_execution(
    db: Session,
    order: MerchantOrder,
    *,
    action: str,
    status: str,
    outcome: str,
    reason: str,
    recovery_probability: float,
    expected_recovered_value: float,
    attempt_number: int,
    planned_method: str | None = None,
    error: str | None = None,
) -> RecoveryAction:
    record = RecoveryAction(
        order_id=order.internal_id,
        action_type=action,
        reason=reason,
        attempt_number=attempt_number,
        status=status,
        case_status=status,
        recovery_probability=str(recovery_probability),
        expected_recovered_value=str(expected_recovered_value),
        outcome_status=outcome,
        error=error,
        planned_method=planned_method,
        created_at=datetime.utcnow(),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def execute_recovery_decision(
    db: Session,
    order: MerchantOrder,
    *,
    decision: str,
    recovery_probability: float = 0.0,
    expected_recovered_value: float = 0.0,
    attempt_number: int,
    max_attempts: int | None = None,
    failure_code: str | None = None,
    reason: str = "AI recovery decision",
    planned_method: str | None = None,
) -> RecoveryExecution:
    """Execute one bounded, idempotent customer-facing recovery decision.

    RETRY records that checkout should be reopened. It never charges a customer.
    WAIT records deferred state, and STOP records a terminal decision.
    """
    decision = decision.upper()
    if decision not in {"RETRY", "WAIT", "STOP"}:
        raise ValueError(f"unsupported recovery decision: {decision}")

    max_attempts = min(
        int(max_attempts or config.MAX_RECOVERY_ATTEMPTS),
        int(config.MAX_RECOVERY_ATTEMPTS),
    )
    if order.status == "paid":
        record = _record_execution(
            db, order, action="recovery_stop", status="stopped", outcome="blocked",
            reason="order already paid", recovery_probability=recovery_probability,
            expected_recovered_value=expected_recovered_value, attempt_number=attempt_number,
            planned_method=planned_method,
        )
        return RecoveryExecution(decision, "stopped", "blocked", "order already paid", record.id)

    if order.status in {"stopped", "exhausted"} and decision != "STOP":
        raise InvalidRecoveryTransition(
            f"cannot execute {decision} from terminal order state {order.status}"
        )

    if decision == "WAIT" and attempt_number >= max_attempts:
        record = _record_execution(
            db, order, action="recovery_stop", status="stopped", outcome="blocked",
            reason="maximum recovery attempts reached", recovery_probability=recovery_probability,
            expected_recovered_value=expected_recovered_value, attempt_number=attempt_number,
        )
        order.status = "exhausted"
        db.commit()
        return RecoveryExecution("WAIT", "stopped", "blocked", record.reason, record.id)

    if decision == "RETRY":
        existing = _existing_execution(db, order, "recovery_retry_pending", attempt_number)
        if existing:
            return RecoveryExecution("RETRY", existing.status, "duplicate", existing.reason, existing.id, True)
        eligible, policy_reason, policy_attempt = evaluate_recovery_policy(db, order)
        if not eligible or attempt_number >= max_attempts:
            stop_reason = policy_reason if not eligible else "maximum recovery attempts reached"
            record = _record_execution(
                db, order, action="recovery_stop", status="stopped", outcome="blocked",
                reason=stop_reason, recovery_probability=recovery_probability,
                expected_recovered_value=expected_recovered_value, attempt_number=policy_attempt or attempt_number,
                planned_method=planned_method,
            )
            order.status = "exhausted" if "max" in stop_reason.lower() else order.status
            db.commit()
            return RecoveryExecution("RETRY", "stopped", "blocked", stop_reason, record.id)
        if failure_code in {"AUTHENTICATION_ERROR", "INSUFFICIENT_FUNDS", "CARD_DECLINED", "BANK_DECLINED", "INVALID_ACCOUNT", "CURRENCY_NOT_ALLOWED"}:
            record = _record_execution(
                db, order, action="recovery_stop", status="stopped", outcome="blocked",
                reason="failure is classified as non-recoverable", recovery_probability=recovery_probability,
                expected_recovered_value=expected_recovered_value, attempt_number=attempt_number,
                planned_method=planned_method,
            )
            return RecoveryExecution("RETRY", "stopped", "blocked", record.reason, record.id)
        record = _record_execution(
            db, order, action="recovery_retry_pending", status="pending", outcome="checkout_required",
            reason=reason, recovery_probability=recovery_probability,
            expected_recovered_value=expected_recovered_value, attempt_number=attempt_number,
            planned_method=planned_method,
        )
        order.status = "recoverable"
        db.commit()
        return RecoveryExecution("RETRY", "pending", "checkout_required", reason, record.id, True)

    action_type = "recovery_waiting" if decision == "WAIT" else "recovery_stop"
    status = "waiting" if decision == "WAIT" else "stopped"
    existing = _existing_execution(db, order, action_type, attempt_number)
    if existing:
        return RecoveryExecution(decision, existing.status, "duplicate", existing.reason, existing.id, decision == "WAIT")
    record = _record_execution(
        db, order, action=action_type, status=status, outcome="deferred" if decision == "WAIT" else "stopped",
        reason=reason, recovery_probability=recovery_probability,
        expected_recovered_value=expected_recovered_value, attempt_number=attempt_number,
    )
    if decision == "WAIT":
        order.status = "recoverable"
    else:
        order.status = "stopped"
    db.commit()
    return RecoveryExecution(decision, status, record.outcome_status or status, reason, record.id, decision == "WAIT")