
    
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import AgentTrace, MerchantOrder, PaymentAttempt
from backend.services.payment_verification import resolve_final_payment_state, verify_order_and_payment_state
from backend.services.recovery import evaluate_recovery_policy, mark_order_recoverable, register_retry_attempt
from backend.services.recovery_executor import execute_recovery_decision
from backend.services.recovery import mark_order_abandoned

router = APIRouter(prefix="/payments", tags=["payments"])


@router.get("/order/{razorpay_order_id}/status")
def get_order_payment_status(razorpay_order_id: str, db: Session = Depends(get_db)):
    order = db.query(MerchantOrder).filter(MerchantOrder.razorpay_order_id == razorpay_order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return verify_order_and_payment_state(db, order)


@router.post("/order/{razorpay_order_id}/retry")
def retry_payment(razorpay_order_id: str, db: Session = Depends(get_db)):
    order = db.query(MerchantOrder).filter(MerchantOrder.razorpay_order_id == razorpay_order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    if order.status == "paid":
        raise HTTPException(status_code=400, detail="This order is already paid and cannot be retried.")

    if order.status in {"exhausted", "stopped"}:
        raise HTTPException(status_code=400, detail="This order is not eligible for recovery (exhausted or explicitly stopped).")

    eligible, reason, attempt_number = evaluate_recovery_policy(db, order)
    if not eligible:
        raise HTTPException(status_code=400, detail=reason)

    latest_attempt = (
        db.query(PaymentAttempt)
        .filter(PaymentAttempt.razorpay_order_id == order.razorpay_order_id)
        .order_by(PaymentAttempt.attempt_number.desc())
        .first()
    )
    execution = execute_recovery_decision(
        db,
        order,
        decision="RETRY",
        attempt_number=attempt_number or 1,
        failure_code=latest_attempt.failure_code if latest_attempt else None,
        reason=reason,
    )
    if execution.status != "pending":
        raise HTTPException(status_code=400, detail=execution.reason)
    return {
        "eligible": True,
        "order_id": order.razorpay_order_id,
        "attempt_number": attempt_number,
        "reason": reason,
        "recovery_action_id": execution.recovery_action_id,
        "message": "Retry is allowed for the existing Razorpay order. Re-open checkout for the same order_id.",
    }


@router.post("/order/{razorpay_order_id}/abandon")
def abandon_checkout(razorpay_order_id: str, db: Session = Depends(get_db)):
    order = db.query(MerchantOrder).filter(
        MerchantOrder.razorpay_order_id == razorpay_order_id
    ).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    if order.status == "paid":
        return {"status": "already_paid", "order_id": razorpay_order_id}
    if db.query(PaymentAttempt).filter(
        PaymentAttempt.order_id == order.internal_id
    ).count():
        raise HTTPException(
            status_code=409,
            detail="This order has a payment attempt; use payment recovery instead.",
        )
    mark_order_abandoned(db, order)
    from backend.agent import RecoveryAgent
    trace = RecoveryAgent(db).observe_abandonment(order)
    return {
        "status": order.status,
        "order_id": razorpay_order_id,
        "decision": trace.selected_action,
        "planned_method": trace.planned_method,
    }


@router.get("/history/{razorpay_order_id}")
def get_payment_history(razorpay_order_id: str, db: Session = Depends(get_db)):
    import json
    from backend.models import RecoveryAction
    
    order = db.query(MerchantOrder).filter(MerchantOrder.razorpay_order_id == razorpay_order_id).first()
    if not order:
        return []
        
    attempts = db.query(PaymentAttempt).filter(PaymentAttempt.razorpay_order_id == razorpay_order_id).all()
    actions = db.query(RecoveryAction).filter(
        RecoveryAction.order_id == order.internal_id,
        RecoveryAction.action_type.in_(["abandonment_candidate", "recovery_retry_pending", "recovery_waiting", "recovery_stop", "recovery_exhausted"])
    ).all()
    
    action_map = {action.attempt_number: action for action in actions}
    
    result = []
    for attempt in sorted(attempts, key=lambda item: item.created_at):
        data = {
            "internal_id": attempt.internal_id,
            "razorpay_payment_id": attempt.razorpay_payment_id,
            "status": attempt.status,
            "method": attempt.method,
            "failure_code": attempt.failure_code,
            "failure_description": attempt.failure_description,
            "attempt_number": attempt.attempt_number,
        }
        
        # Merge AI decision if available for this attempt
        action = action_map.get(attempt.attempt_number)
        if action and action.error:
            try:
                ai_data = json.loads(action.error)
                data["ai_decision"] = ai_data.get("ai_decision")
                data["net_expected_value"] = ai_data.get("net_expected_value")
                data["ai_reason"] = action.reason
                data["expected_recovered_value"] = action.expected_recovered_value
                data["recovery_probability"] = action.recovery_probability
            except Exception:
                pass
                
        result.append(data)
        
    return result


@router.post("/final-state/{internal_id}")
def final_state(internal_id: int, db: Session = Depends(get_db)):
    order = db.query(MerchantOrder).filter(MerchantOrder.internal_id == internal_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return resolve_final_payment_state(db, order)


@router.get("/candidates")
def get_recovery_candidates(db: Session = Depends(get_db)):
    import json
    from backend.models import RecoveryAction
    
    actions = db.query(RecoveryAction).filter(
        RecoveryAction.action_type.in_(["abandonment_candidate", "recovery_retry_pending", "recovery_waiting", "recovery_stop", "recovery_exhausted"])
    ).order_by(RecoveryAction.created_at.desc()).all()
    
    # We want to group by order_id, but pick the action that actually has the AI data
    order_action_map = {}
    for action in actions:
        if action.order_id not in order_action_map:
            order_action_map[action.order_id] = action
        elif action.error and not order_action_map[action.order_id].error:
            # If the current best action has no AI data, but this older one does, prefer this one
            order_action_map[action.order_id] = action
            
    result = []
    for order_id, action in order_action_map.items():
        order = action.order
        if not order: continue
        
        attempt = db.query(PaymentAttempt).filter(
            PaymentAttempt.order_id == order.internal_id,
            PaymentAttempt.attempt_number == action.attempt_number
        ).first()
        
        is_recovered = db.query(RecoveryAction).filter(
            RecoveryAction.order_id == order.internal_id,
            RecoveryAction.action_type == "recovery_success"
        ).first() is not None
        
        ai_decision = None
        net_expected_value = None
        if action.error:
            try:
                ai_data = json.loads(action.error)
                ai_decision = ai_data.get("ai_decision")
                net_expected_value = ai_data.get("net_expected_value")
            except Exception:
                pass
                
        status = "Recovered" if is_recovered else order.status.capitalize()
        if order.status == "paid" and not is_recovered:
            status = "Paid"
            
        result.append({
            "razorpay_order_id": order.razorpay_order_id,
            "amount": order.amount,
            "failure_code": attempt.failure_code if attempt else "Unknown",
            "recovery_probability": action.recovery_probability,
            "expected_recovered_value": action.expected_recovered_value,
            "net_expected_value": net_expected_value,
            "ai_decision": ai_decision,
            "ai_reason": action.reason,
            "status": status,
            "attempt_number": action.attempt_number
        })
    
    return result

@router.get("/order/{razorpay_order_id}/trace")
def get_agent_trace(razorpay_order_id: str, db: Session = Depends(get_db)):
    import json
    traces = db.query(AgentTrace).filter(
        AgentTrace.razorpay_order_id == razorpay_order_id
    ).order_by(AgentTrace.created_at.asc()).all()
    
    result = []
    for t in traces:
        decision_context = {}
        if t.decision_inputs:
            try:
                decision_context = json.loads(t.decision_inputs)
            except json.JSONDecodeError:
                decision_context = {}
        result.append({
            "id": t.id,
            "event": t.event,
            "diagnosis": t.diagnosis,
            "probability": t.probability,
            "expected_value": t.expected_value,
            "net_ev": t.net_ev,
            "selected_action": t.selected_action,
            "planned_method": t.planned_method,
            "actual_payment_method": t.actual_payment_method,
            "customer_history": decision_context.get("customer_history"),
            "execution_result": t.execution_result,
            "webhook_outcome": t.webhook_outcome,
            "recovered_revenue": t.recovered_revenue,
            "created_at": t.created_at.isoformat()
        })
    return result


@router.get("/customer/{customer_id}/history")
def get_customer_history(customer_id: str, db: Session = Depends(get_db)):
    from backend.services.customer_history import get_customer_payment_history

    return get_customer_payment_history(db, customer_id).to_dict()


@router.get("/order/{razorpay_order_id}/voice-recovery")
def voice_recovery_message(razorpay_order_id: str, db: Session = Depends(get_db)):
    from backend.services.voice_recovery import generate_recovery_message
    import json

    order = db.query(MerchantOrder).filter(
        MerchantOrder.razorpay_order_id == razorpay_order_id
    ).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    trace = db.query(AgentTrace).filter(
        AgentTrace.razorpay_order_id == razorpay_order_id
    ).order_by(AgentTrace.created_at.desc()).first()
    if not trace:
        raise HTTPException(status_code=404, detail="No recovery recommendation found")
    history = None
    if trace.decision_inputs:
        try:
            history = json.loads(trace.decision_inputs).get("customer_history")
        except json.JSONDecodeError:
            history = None
    return generate_recovery_message(
        scenario=trace.event,
        decision=trace.selected_action or "STOP",
        recommended_method=trace.planned_method,
        reason=trace.diagnosis or "recovery recommendation",
        customer_history=history,
    )


@router.get("/simulation/recovery-journeys")
def simulate_recovery_journeys():
    from backend.services.synthetic_journey import run_batch_simulation
    seeds = [42, 100, 777, 999, 2024]
    results = []
    
    total_baseline_revenue = 0.0
    total_ai_revenue = 0.0
    total_risk = 0.0
    total_baseline_interventions = 0
    total_ai_interventions = 0
    total_baseline_attributed_revenue = 0.0
    total_ai_attributed_revenue = 0.0
    total_baseline_natural_revenue = 0.0
    total_ai_natural_revenue = 0.0
    total_baseline_switches = 0
    total_ai_switches = 0
    total_baseline_exhausted = 0
    total_ai_exhausted = 0
    total_baseline_natural_count = 0
    total_ai_natural_count = 0
    
    for seed in seeds:
        res = run_batch_simulation(seed, num_journeys=1000)
        results.append(res)
        
        total_baseline_revenue += res["baseline"]["recovered_revenue"]
        total_ai_revenue += res["ai"]["recovered_revenue"]
        total_risk += res["ai"]["total_revenue_at_risk"]
        
        total_baseline_interventions += res["baseline"]["interventions"]
        total_ai_interventions += res["ai"]["interventions"]
        total_baseline_attributed_revenue += res["baseline"]["intervention_attributed_recovered_revenue"]
        total_ai_attributed_revenue += res["ai"]["intervention_attributed_recovered_revenue"]
        total_baseline_natural_revenue += res["baseline"]["natural_success_revenue"]
        total_ai_natural_revenue += res["ai"]["natural_success_revenue"]
        total_baseline_switches += res["baseline"]["switches"]
        total_ai_switches += res["ai"]["switches"]
        total_baseline_exhausted += res["baseline"]["exhausted"]
        total_ai_exhausted += res["ai"]["exhausted"]
        total_baseline_natural_count += res["baseline"]["natural_success_count"]
        total_ai_natural_count += res["ai"]["natural_success_count"]

    return {
        "seeds_evaluated": seeds,
        "journeys_per_seed": 1000,
        "aggregate": {
            "total_revenue_at_risk": total_risk,
            "baseline_recovered_revenue": total_baseline_revenue,
            "ai_recovered_revenue": total_ai_revenue,
            "baseline_intervention_attributed_revenue": total_baseline_attributed_revenue,
            "ai_intervention_attributed_revenue": total_ai_attributed_revenue,
            "baseline_natural_success_revenue": total_baseline_natural_revenue,
            "ai_natural_success_revenue": total_ai_natural_revenue,
            "ai_revenue_lift": total_ai_attributed_revenue - total_baseline_attributed_revenue,
            "baseline_recovery_rate": total_baseline_attributed_revenue / max(1, total_risk),
            "ai_recovery_rate": total_ai_attributed_revenue / max(1, total_risk),
            "baseline_interventions": total_baseline_interventions,
            "ai_interventions": total_ai_interventions,
            "baseline_switches": total_baseline_switches,
            "ai_switches": total_ai_switches,
            "baseline_exhausted": total_baseline_exhausted,
            "ai_exhausted": total_ai_exhausted,
            "baseline_natural_success_count": total_baseline_natural_count,
            "ai_natural_success_count": total_ai_natural_count,
        },
        "details": results
    }
