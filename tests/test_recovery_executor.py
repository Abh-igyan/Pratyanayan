from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.database import Base, engine, get_db
from backend.main import app
from backend.models import AgentTrace, MerchantOrder, PaymentAttempt, RecoveryAction
from backend.services.recovery import mark_order_recoverable
from backend.services.recovery_executor import InvalidRecoveryTransition, execute_recovery_decision


@pytest.fixture
def db_session():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = Session(bind=engine)
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)


@pytest.fixture
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_failed_order(db_session, order_id="executor_order", failure_code="PAYMENT_ERROR"):
    order = MerchantOrder(
        razorpay_order_id=order_id,
        amount=50000,
        currency="INR",
        customer_id="cust_executor",
        status="created",
    )
    db_session.add(order)
    db_session.commit()
    db_session.refresh(order)
    db_session.add(PaymentAttempt(
        razorpay_payment_id=f"pay_{order_id}",
        razorpay_order_id=order_id,
        amount=50000,
        status="failed",
        failure_code=failure_code,
        attempt_number=1,
        order_id=order.internal_id,
    ))
    db_session.commit()
    mark_order_recoverable(db_session, order)
    return order


def test_retry_execution_is_bounded_and_audited(db_session):
    order = make_failed_order(db_session)

    result = execute_recovery_decision(
        db_session,
        order,
        decision="RETRY",
        recovery_probability=0.8,
        expected_recovered_value=40000,
        attempt_number=1,
    )

    assert result.status == "pending"
    assert result.checkout_required is True
    action = db_session.query(RecoveryAction).filter_by(id=result.recovery_action_id).one()
    assert action.action_type == "recovery_retry_pending"
    assert action.outcome_status == "checkout_required"
    assert action.recovery_probability == "0.8"
    assert action.expected_recovered_value == "40000"


def test_wait_execution_is_explicit(db_session):
    order = make_failed_order(db_session, "wait_order")

    result = execute_recovery_decision(
        db_session, order, decision="WAIT", recovery_probability=0.5,
        expected_recovered_value=25000, attempt_number=1, reason="cooling window",
    )

    assert result.status == "waiting"
    assert result.checkout_required is True
    assert order.status == "recoverable"


def test_stop_execution_records_terminal_state(db_session):
    order = make_failed_order(db_session, "stop_order")

    result = execute_recovery_decision(
        db_session, order, decision="STOP", attempt_number=1, reason="low value",
    )

    assert result.status == "stopped"
    assert result.checkout_required is False
    assert order.status == "stopped"


def test_retry_guards_attempt_limit_and_paid_order(db_session):
    order = make_failed_order(db_session, "exhausted_order")
    result = execute_recovery_decision(
        db_session, order, decision="RETRY", attempt_number=2, max_attempts=2,
    )
    assert result.status == "stopped"
    assert "maximum" in result.reason

    paid = make_failed_order(db_session, "paid_order")
    paid.status = "paid"
    db_session.commit()
    paid_result = execute_recovery_decision(
        db_session, paid, decision="RETRY", attempt_number=1,
    )
    assert paid_result.status == "stopped"
    assert paid_result.outcome == "blocked"


def test_retry_rejects_nonrecoverable_and_duplicate_execution(db_session):
    order = make_failed_order(db_session, "nonrecoverable_order", "CARD_DECLINED")
    order.status = "recoverable" # Reset to test executor layer rejection
    blocked = execute_recovery_decision(
        db_session, order, decision="RETRY", attempt_number=1,
        failure_code="CARD_DECLINED",
    )
    assert blocked.status == "stopped"

    retry_order = make_failed_order(db_session, "duplicate_order")
    first = execute_recovery_decision(
        db_session, retry_order, decision="RETRY", attempt_number=1,
    )
    second = execute_recovery_decision(
        db_session, retry_order, decision="RETRY", attempt_number=1,
    )
    assert second.outcome == "duplicate"
    assert second.recovery_action_id == first.recovery_action_id
    assert db_session.query(RecoveryAction).filter_by(
        order_id=retry_order.internal_id, action_type="recovery_retry_pending"
    ).count() == 1


def test_invalid_decision_is_rejected(db_session):
    order = make_failed_order(db_session, "invalid_order")
    with pytest.raises(ValueError):
        execute_recovery_decision(db_session, order, decision="CHARGE", attempt_number=1)

    order.status = "stopped"
    db_session.commit()
    with pytest.raises(InvalidRecoveryTransition):
        execute_recovery_decision(db_session, order, decision="WAIT", attempt_number=1)


def test_late_payment_success_marks_order_paid_after_executor_retry(client, db_session):
    order = make_failed_order(db_session, "late_success_order")
    execute_recovery_decision(
        db_session, order, decision="RETRY", attempt_number=1,
    )

    payload = {
        "event": "payment.captured",
        "event_id": "late_success_event",
        "payload": {"payment": {"entity": {
            "id": "late_pay_2", "order_id": order.razorpay_order_id,
            "amount": 50000, "status": "captured", "method": "card",
        }}},
    }
    body = json.dumps(payload).encode("utf-8")
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"
    signature = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()
    try:
        response = client.post(
            "/webhooks/razorpay",
            content=body,
            headers={"X-Razorpay-Signature": signature},
        )
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original

    assert response.status_code == 200
    db_session.refresh(order)
    assert order.status == "paid"
    assert db_session.query(RecoveryAction).filter_by(
        order_id=order.internal_id, action_type="recovery_success"
    ).count() == 1


def test_payment_failed_closes_pending_retry_as_failed(client, db_session):
    order = make_failed_order(db_session, "retry_failed_order")
    result = execute_recovery_decision(
        db_session, order, decision="RETRY", attempt_number=1,
    )

    payload = {
        "event": "payment.failed",
        "event_id": "retry_failed_event",
        "payload": {"payment": {"entity": {
            "id": "retry_failed_pay_2", "order_id": order.razorpay_order_id,
            "amount": 50000, "method": "card", "error_code": "TIMEOUT",
        }}},
    }
    body = json.dumps(payload).encode("utf-8")
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"
    signature = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()
    try:
        response = client.post(
            "/webhooks/razorpay",
            content=body,
            headers={"X-Razorpay-Signature": signature},
        )
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original

    assert response.status_code == 200
    action = db_session.query(RecoveryAction).filter_by(id=result.recovery_action_id).one()
    assert action.status == "failed"
    assert action.outcome_status == "payment_failed"


def test_manual_retry_route_preserves_nonrecoverable_guard(client, db_session):
    order = make_failed_order(db_session, "route_nonrecoverable", "CARD_DECLINED")
    response = client.post(f"/payments/order/{order.razorpay_order_id}/retry")
    assert response.status_code == 400
    assert "exhausted" in response.json()["detail"]


def test_abandonment_creates_candidate_without_payment_attempt(client, db_session):
    order = MerchantOrder(
        razorpay_order_id="abandoned_order",
        amount=50000,
        currency="INR",
        customer_id="cust_abandon",
        status="created",
    )
    db_session.add(order)
    db_session.commit()
    response = client.post("/payments/order/abandoned_order/abandon")
    assert response.status_code == 200
    db_session.refresh(order)
    assert order.status == "recoverable"
    assert db_session.query(PaymentAttempt).filter_by(order_id=order.internal_id).count() == 0
    assert db_session.query(RecoveryAction).filter_by(
        order_id=order.internal_id, action_type="abandonment_candidate"
    ).count() == 1


def test_abandonment_can_stop_without_creating_payment_attempt(client, db_session, monkeypatch):
    class StopEngine:
        def decide(self, case):
            return type("Outcome", (), {
                "decision": "STOP",
                "reason": "test stop",
                "recovery_probability": 0.0,
                "expected_recovered_value": 0.0,
                "net_expected_value": -1.0,
                "intervention_cost": 1.0,
                "to_dict": lambda self: {"decision": "STOP", "recovery_probability": 0.0},
            })()

    monkeypatch.setattr("backend.agent.AIDecisionEngine", StopEngine)
    order = MerchantOrder(
        razorpay_order_id="abandoned_stop_order",
        amount=50000,
        currency="INR",
        customer_id="cust_abandon_stop",
        status="created",
    )
    db_session.add(order)
    db_session.commit()
    response = client.post("/payments/order/abandoned_stop_order/abandon")
    assert response.status_code == 200
    db_session.refresh(order)
    assert order.status == "stopped"
    assert db_session.query(PaymentAttempt).filter_by(order_id=order.internal_id).count() == 0


def test_abandonment_recovery_capture_is_assisted_and_linked(client, db_session, monkeypatch):
    class RetryEngine:
        def decide(self, case):
            return type("Outcome", (), {
                "decision": "RETRY",
                "reason": "test recovery recommendation",
                "recovery_probability": 0.8,
                "expected_recovered_value": 40000.0,
                "net_expected_value": 39900.0,
                "intervention_cost": 100.0,
                "to_dict": lambda self: {"decision": "RETRY", "recovery_probability": 0.8},
            })()

    monkeypatch.setattr("backend.agent.AIDecisionEngine", RetryEngine)
    order = MerchantOrder(
        razorpay_order_id="abandoned_capture_order",
        amount=50000,
        currency="INR",
        customer_id="cust_abandon_capture",
        status="created",
    )
    db_session.add(order)
    db_session.commit()
    response = client.post("/payments/order/abandoned_capture_order/abandon")
    assert response.status_code == 200
    pending = db_session.query(RecoveryAction).filter_by(
        order_id=order.internal_id, action_type="recovery_retry_pending"
    ).first()
    assert pending is not None

    payload = {
        "event": "payment.captured",
        "event_id": "abandoned_capture_event",
        "payload": {"payment": {"entity": {
            "id": "abandoned_capture_pay", "order_id": order.razorpay_order_id,
            "amount": 50000, "status": "captured", "method": "upi",
        }}},
    }
    body = json.dumps(payload).encode("utf-8")
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"
    signature = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()
    try:
        capture = client.post("/webhooks/razorpay", content=body, headers={"X-Razorpay-Signature": signature})
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original
    assert capture.status_code == 200
    trace = db_session.query(AgentTrace).filter_by(
        order_id=order.internal_id, event="payment.captured"
    ).one()
    assert trace.webhook_outcome == "RECOVERY_SUCCESS"


def test_historical_retry_without_pending_execution_is_natural_success(client, db_session):
    order = make_failed_order(db_session, "historical_retry_order")
    db_session.add(RecoveryAction(
        order_id=order.internal_id,
        action_type="retry_payment",
        reason="historical record only",
        attempt_number=1,
        status="failed",
        case_status="active",
    ))
    db_session.commit()
    payload = {
        "event": "payment.captured",
        "event_id": "historical_capture",
        "payload": {"payment": {"entity": {
            "id": "historical_pay_2", "order_id": order.razorpay_order_id,
            "amount": 50000, "status": "captured", "method": "card",
        }}},
    }
    body = json.dumps(payload).encode("utf-8")
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"
    signature = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()
    try:
        response = client.post("/webhooks/razorpay", content=body, headers={"X-Razorpay-Signature": signature})
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original
    assert response.status_code == 200
    assert db_session.query(RecoveryAction).filter_by(
        order_id=order.internal_id, action_type="recovery_success"
    ).count() == 0


def test_pending_capture_links_actual_method_and_duplicate_is_idempotent(client, db_session):
    order = make_failed_order(db_session, "method_mismatch_order")
    result = execute_recovery_decision(
        db_session, order, decision="RETRY", attempt_number=1,
        planned_method="upi",
    )
    payload = {
        "event": "payment.captured",
        "event_id": "method_mismatch_capture",
        "payload": {"payment": {"entity": {
            "id": "method_mismatch_pay_2", "order_id": order.razorpay_order_id,
            "amount": 50000, "status": "captured", "method": "card",
        }}},
    }
    body = json.dumps(payload).encode("utf-8")
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"
    signature = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()
    try:
        first = client.post("/webhooks/razorpay", content=body, headers={"X-Razorpay-Signature": signature})
        second = client.post("/webhooks/razorpay", content=body, headers={"X-Razorpay-Signature": signature})
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original
    action = db_session.query(RecoveryAction).filter_by(id=result.recovery_action_id).one()
    assert first.status_code == 200
    assert second.json()["status"] == "duplicate"
    assert action.actual_payment_method == "card"
    assert db_session.query(RecoveryAction).filter_by(
        order_id=order.internal_id, action_type="recovery_success"
    ).count() == 1