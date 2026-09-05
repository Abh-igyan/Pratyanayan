import json
import logging
from datetime import datetime

from sqlalchemy.orm import Session

import backend.config as config
from backend.models import AgentTrace, MerchantOrder, PaymentAttempt, RecoveryAction
from backend.services.recovery import evaluate_recovery_policy
from backend.services.recovery_executor import execute_recovery_decision
from backend.services.customer_history import get_customer_payment_history
from ml.decision_engine import AIDecisionEngine
from ml.synthetic_world import PAYMENT_METHODS

logger = logging.getLogger(__name__)

class RecoveryAgent:
    """
    Stateful Orchestration Layer for AI Revenue Recovery.
    Explicitly models the Observe, Diagnose, Evaluate, Plan, Execute, Verify, Replan, Audit loop.
    Supports dynamic budgeting and alternative method evaluation.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.engine = AIDecisionEngine()

    def observe_failure(self, order: MerchantOrder, failed_attempt: PaymentAttempt) -> AgentTrace:
        """Observe payment.failed event and trigger the agent loop."""
        trace = AgentTrace(
            order_id=order.internal_id,
            razorpay_order_id=order.razorpay_order_id,
            event="payment.failed",
            created_at=datetime.utcnow()
        )
        self.db.add(trace)
        
        # 1. Observe context
        eligible, reason, attempt_number = evaluate_recovery_policy(self.db, order)
        attempt_number = attempt_number or 1
        
        # 2. Diagnose
        diagnosis, method_degrading = self._diagnose(order, failed_attempt, eligible, reason)
        trace.diagnosis = diagnosis
        
        if not eligible:
            # 4. Plan (STOP)
            trace.selected_action = "STOP"
            trace.execution_result = reason
            
            # Execute STOP
            execute_recovery_decision(
                self.db,
                order,
                decision="STOP",
                attempt_number=attempt_number,
                max_attempts=config.MAX_RECOVERY_ATTEMPTS,
                failure_code=failed_attempt.failure_code,
                reason=reason,
            )
            self.db.commit()
            return trace
            
        # 3. Evaluate multiple methods
        best_outcome = None
        best_method = None
        evaluation_results = {}
        customer_history = get_customer_payment_history(
            self.db,
            order.customer_id,
            exclude_order_id=order.internal_id,
            exclude_attempt_id=failed_attempt.internal_id,
        )
        
        failed_by_method = {}
        for attempt in self.db.query(PaymentAttempt).filter(
            PaymentAttempt.razorpay_order_id == order.razorpay_order_id,
            PaymentAttempt.status == "failed",
        ).all():
            failed_by_method[attempt.method or "card"] = failed_by_method.get(attempt.method or "card", 0) + 1
        available_methods = [
            method for method in PAYMENT_METHODS
            if failed_by_method.get(method, 0) < 2
        ]
        if not available_methods:
            trace.selected_action = "STOP"
            trace.execution_result = "all payment methods exhausted"
            execute_recovery_decision(
                self.db,
                order,
                decision="STOP",
                attempt_number=attempt_number,
                max_attempts=config.MAX_RECOVERY_ATTEMPTS,
                failure_code=failed_attempt.failure_code,
                reason="all payment methods exhausted",
            )
            return trace
        for method in available_methods:
            case_dict = self._build_case_dict(
                order, failed_attempt, attempt_number, method, customer_history
            )
            outcome = self.engine.decide(case_dict)
            evaluation_results[method] = outcome.to_dict()
            
            # Choose best method based on net EV, subject to not switching to a degrading method unnecessarily
            if best_outcome is None or outcome.net_expected_value > best_outcome.net_expected_value:
                best_outcome = outcome
                best_method = method
                
        # Persist inputs and evaluation results
        trace.decision_inputs = json.dumps({
            "base_case": self._build_case_dict(order, failed_attempt, attempt_number, failed_attempt.method or "card"),
            "evaluations": evaluation_results,
            "customer_history": customer_history.to_dict(),
        })
        
        # 4. Plan
        # If the best method differs from the failed one, we recommend a switch (if it's viable)
        current_method = failed_attempt.method or "card"
        
        trace.probability = best_outcome.recovery_probability
        trace.expected_value = best_outcome.expected_recovered_value
        trace.net_ev = best_outcome.net_expected_value
        
        if best_outcome.decision == "STOP":
            trace.selected_action = "STOP"
            trace.planned_method = None
        elif best_outcome.decision == "WAIT":
            trace.selected_action = "WAIT"
            trace.planned_method = best_method
        else: # RETRY
            if best_method != current_method:
                trace.selected_action = "SWITCH_PAYMENT_METHOD"
                trace.planned_method = best_method
            else:
                trace.selected_action = "RETRY"
                trace.planned_method = current_method
        
        # 5. Execute
        execution = self._execute_plan(
            order, trace.selected_action, trace.planned_method,
            best_outcome, attempt_number, failed_attempt.failure_code,
        )
        trace.execution_result = execution.outcome
        
        if trace.selected_action in {"RETRY", "WAIT", "SWITCH_PAYMENT_METHOD"}:
            order.status = "recoverable"
        else:
            order.status = "exhausted"
            
        self.db.commit()
        return trace

    def observe_abandonment(self, order: MerchantOrder) -> AgentTrace:
        """Evaluate an abandoned checkout without inventing a payment attempt."""
        trace = AgentTrace(
            order_id=order.internal_id,
            razorpay_order_id=order.razorpay_order_id,
            event="checkout.abandoned",
            diagnosis="Checkout abandoned before a payment attempt.",
            created_at=datetime.utcnow(),
        )
        self.db.add(trace)
        eligible, reason, attempt_number = evaluate_recovery_policy(self.db, order)
        attempt_number = attempt_number or 1
        customer_history = get_customer_payment_history(
            self.db, order.customer_id, exclude_order_id=order.internal_id
        )
        best_outcome = None
        best_method = None
        evaluations = {}
        for method in PAYMENT_METHODS:
            case = self._build_case_dict(order, None, attempt_number, method, customer_history)
            outcome = self.engine.decide(case)
            evaluations[method] = outcome.to_dict()
            if best_outcome is None or outcome.net_expected_value > best_outcome.net_expected_value:
                best_outcome, best_method = outcome, method
        if not eligible or best_outcome is None:
            trace.selected_action = "STOP"
            trace.execution_result = reason
            execute_recovery_decision(
                self.db, order, decision="STOP", attempt_number=attempt_number,
                reason=reason, max_attempts=config.MAX_RECOVERY_ATTEMPTS,
            )
            return trace
        trace.decision_inputs = json.dumps({
            "evaluations": evaluations,
            "customer_history": customer_history.to_dict(),
            "abandonment": True,
        })
        trace.probability = best_outcome.recovery_probability
        trace.expected_value = best_outcome.expected_recovered_value
        trace.net_ev = best_outcome.net_expected_value
        trace.planned_method = best_method if best_outcome.decision != "STOP" else None
        trace.selected_action = best_outcome.decision
        execution = self._execute_plan(
            order, trace.selected_action, trace.planned_method,
            best_outcome, attempt_number, None,
        )
        trace.execution_result = execution.outcome
        order.status = "recoverable" if trace.selected_action in {"RETRY", "WAIT"} else "stopped"
        self.db.commit()
        return trace

    def verify_captured(self, order: MerchantOrder, attempt: PaymentAttempt) -> AgentTrace:
        """Verify payment.captured event."""
        trace = AgentTrace(
            order_id=order.internal_id,
            razorpay_order_id=order.razorpay_order_id,
            event="payment.captured",
            created_at=datetime.utcnow()
        )
        self.db.add(trace)
        
        # Check if there was a previous failure and a retry action
        has_failed_attempt = self.db.query(PaymentAttempt).filter(
            PaymentAttempt.razorpay_order_id == order.razorpay_order_id,
            PaymentAttempt.status == "failed",
            PaymentAttempt.attempt_number < attempt.attempt_number,
        ).first() is not None
        
        pending_action = self.db.query(RecoveryAction).filter(
            RecoveryAction.order_id == order.internal_id,
            RecoveryAction.action_type == "recovery_retry_pending",
            RecoveryAction.status.in_({"pending", "active"}),
            RecoveryAction.attempt_number == max(1, attempt.attempt_number - 1),
        ).order_by(RecoveryAction.id.desc()).first()
        
        has_abandonment_candidate = self.db.query(RecoveryAction).filter(
            RecoveryAction.order_id == order.internal_id,
            RecoveryAction.action_type == "abandonment_candidate",
        ).first() is not None

        if (has_failed_attempt or has_abandonment_candidate) and pending_action:
            # RECOVERY_SUCCESS
            trace.diagnosis = f"Payment captured via {attempt.method} after a recovery intervention."
            trace.webhook_outcome = "RECOVERY_SUCCESS"
            trace.recovered_revenue = float(order.amount)
            
            # Close out pending actions
            self._mark_success_actions(order, attempt.attempt_number)
            pending_action.payment_attempt_id = attempt.internal_id
            pending_action.actual_payment_method = attempt.method
            pending_action.outcome_status = "captured"
            trace.actual_payment_method = attempt.method
            existing_success = self.db.query(RecoveryAction).filter(
                RecoveryAction.order_id == order.internal_id,
                RecoveryAction.action_type == "recovery_success",
            ).first()
            if not existing_success:
                from backend.services.recovery import mark_recovery_success
                mark_recovery_success(self.db, order, attempt_number=attempt.attempt_number)
            
        else:
            # NATURAL_SUCCESS
            trace.diagnosis = f"Payment captured via {attempt.method} without an active recovery intervention or prior failure."
            trace.webhook_outcome = "NATURAL_SUCCESS"
            trace.actual_payment_method = attempt.method
            
            # Just close pending actions as "paid_before_execution"
            self.db.query(RecoveryAction).filter(
                RecoveryAction.order_id == order.internal_id,
                RecoveryAction.status.in_({"pending", "waiting", "active"}),
            ).update({
                "status": "success",
                "case_status": "recovered",
                "outcome_status": "paid_before_execution",
                "completed_at": datetime.utcnow(),
            })

        order.status = "paid"
        order.updated_at = datetime.utcnow()
        self.db.commit()
        return trace

    def verify_order_paid(self, order: MerchantOrder) -> AgentTrace:
        """Verify order.paid event (late capture)."""
        trace = AgentTrace(
            order_id=order.internal_id,
            razorpay_order_id=order.razorpay_order_id,
            event="order.paid",
            diagnosis="Order marked paid (late capture). Natural success.",
            webhook_outcome="NATURAL_SUCCESS",
            created_at=datetime.utcnow()
        )
        trace.actual_payment_method = None
        self.db.add(trace)
        
        self.db.query(RecoveryAction).filter(
            RecoveryAction.order_id == order.internal_id,
            RecoveryAction.status.in_({"pending", "waiting", "active"}),
        ).update({
            "status": "success",
            "case_status": "recovered",
            "outcome_status": "paid_before_execution",
            "completed_at": datetime.utcnow(),
        })
        order.status = "paid"
        order.updated_at = datetime.utcnow()
        self.db.commit()
        return trace

    def _diagnose(self, order: MerchantOrder, failed_attempt: PaymentAttempt, eligible: bool, reason: str) -> tuple[str, bool]:
        if not eligible:
            return f"Non-recoverable: {reason}", False
            
        # Check historical failures in this journey
        attempts = self.db.query(PaymentAttempt).filter(
            PaymentAttempt.razorpay_order_id == order.razorpay_order_id,
            PaymentAttempt.status == "failed"
        ).order_by(PaymentAttempt.attempt_number.asc()).all()
        
        method = failed_attempt.method or "card"
        method_failures = [a for a in attempts if a.method == method]
        
        if len(method_failures) >= 2:
            return f"METHOD_SPECIFIC_DEGRADATION: {method} failed {len(method_failures)} times. Suggesting alternate method if viable.", True
        elif len(attempts) > 1:
            return f"REPEATED_FAILURE: Journey has {len(attempts)} failed attempts.", False
        else:
            return f"TRANSIENT_FAILURE: Initial failure via {method} ({failed_attempt.failure_code or 'PAYMENT_ERROR'}).", False

    def _build_case_dict(self, order: MerchantOrder, latest_attempt: PaymentAttempt | None, attempt_number: int, method: str, history=None) -> dict:
        history = history or get_customer_payment_history(
            self.db,
            order.customer_id,
            exclude_order_id=order.internal_id,
            exclude_attempt_id=latest_attempt.internal_id if latest_attempt else None,
        )
        history_segment = (
            "strong" if history.success_rate >= 0.75 else
            "stable" if history.success_rate >= 0.55 else
            "mixed" if history.success_rate >= 0.35 else
            "risky" if history.total_attempts else "stable"
        )
        case = {
            "order_id": order.razorpay_order_id,
            "amount": order.amount,
            "customer_id": order.customer_id,
            "payment_method": method,
            "failure_code": latest_attempt.failure_code if latest_attempt else "PAYMENT_ERROR",
            "attempt_number": attempt_number,
            "time_since_failure": 0,
            "previous_success_rate": history.success_rate,
            "previous_failed_attempts": history.failed_attempts,
            "customer_history": history_segment,
            "customer_history_segment": history_segment,
            "customer_total_orders": history.total_attempts,
            "customer_successful_orders": history.successful_attempts,
            "customer_failed_orders": history.failed_attempts,
            "customer_success_rate": history.success_rate,
            "customer_failure_rate": 1.0 - history.success_rate if history.total_attempts else 0.0,
            "customer_average_order_value": history.customer_average_order_value,
            "customer_total_spend": history.customer_total_spend,
            "customer_last_success_age": history.customer_last_success_age,
            "customer_last_failure_age": history.customer_last_failure_age,
            "already_paid": order.status == "paid",
            "max_attempts": config.MAX_RECOVERY_ATTEMPTS,
        }
        case.update(history.for_method(method))
        return case

    def _execute_plan(self, order: MerchantOrder, selected_action: str, planned_method: str | None, outcome, attempt_number: int, failure_code: str | None):
        decision = "RETRY" if selected_action == "SWITCH_PAYMENT_METHOD" else selected_action
        reason = outcome.reason
        if selected_action == "SWITCH_PAYMENT_METHOD":
            reason = f"Recommend {planned_method}; customer chooses payment method. " + reason
        return execute_recovery_decision(
            self.db,
            order,
            decision=decision,
            recovery_probability=outcome.recovery_probability,
            expected_recovered_value=outcome.expected_recovered_value,
            attempt_number=attempt_number,
            max_attempts=config.MAX_RECOVERY_ATTEMPTS,
            failure_code=failure_code,
            planned_method=planned_method,
            reason=reason,
        )

    def _mark_success_actions(self, order: MerchantOrder, attempt_number: int):
        existing = self.db.query(RecoveryAction).filter(
            RecoveryAction.order_id == order.internal_id, 
            RecoveryAction.action_type == "recovery_success"
        ).first()
        if existing:
            return
            
        self.db.query(RecoveryAction).filter(
            RecoveryAction.order_id == order.internal_id,
            RecoveryAction.status.in_({"pending", "waiting", "active"}),
        ).update({"status": "success", "case_status": "recovered", "completed_at": datetime.utcnow()})
