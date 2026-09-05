from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.database import Base, engine, get_db
from backend.main import app
from backend.models import MerchantOrder, PaymentAttempt, RecoveryAction, WebhookEvent
from backend.razorpay_client import verify_signature
from backend.services.recovery import (
    compute_dashboard_metrics,
    evaluate_recovery_policy,
    mark_order_recoverable,
    mark_recovery_success,
    register_retry_attempt,
)


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


def test_order_creation_logic(client, monkeypatch):
    def fake_create_razorpay_order(amount_paise: int, currency: str, receipt: str, notes: dict[str, str]):
        return {"id": "order_test_123", "amount": amount_paise, "currency": currency, "receipt": receipt, "notes": notes}

    monkeypatch.setattr("backend.routes.orders.create_razorpay_order", fake_create_razorpay_order)

    response = client.post("/orders", json={"customer_id": "cust_001", "amount": 50000})
    assert response.status_code == 200
    payload = response.json()
    assert payload["razorpay_order_id"] == "order_test_123"
    assert payload["amount"] == 50000
    assert payload["status"] == "created"


def test_recovery_policy(db_session):
    order = MerchantOrder(razorpay_order_id="order_recover_1", amount=50000, currency="INR", customer_id="cust_1", status="created")
    db_session.add(order)
    db_session.commit()
    db_session.refresh(order)

    failed_attempt = PaymentAttempt(
        razorpay_payment_id="pay_fail_1",
        razorpay_order_id=order.razorpay_order_id,
        amount=50000,
        status="failed",
        method="card",
        failure_code="PAYMENT_ERROR",
        failure_description="Declined",
        attempt_number=1,
        order_id=order.internal_id,
    )
    db_session.add(failed_attempt)
    db_session.commit()

    eligible, reason, attempt_number = evaluate_recovery_policy(db_session, order)
    assert eligible is True
    assert attempt_number == 1
    assert "eligible" in reason.lower()


def test_max_retry_limit(db_session, monkeypatch):
    monkeypatch.setattr("backend.config.MAX_RECOVERY_ATTEMPTS", 2)
    order = MerchantOrder(razorpay_order_id="order_recover_2", amount=75000, currency="INR", customer_id="cust_2", status="created")
    db_session.add(order)
    db_session.commit()
    db_session.refresh(order)

    for attempt_number, status in [(1, "failed"), (2, "recovery_retry"), (3, "failed")]:
        db_session.add(
            PaymentAttempt(
                razorpay_payment_id=f"pay_{attempt_number}",
                razorpay_order_id=order.razorpay_order_id,
                amount=75000,
                status=status,
                attempt_number=attempt_number,
                order_id=order.internal_id,
            )
        )
    db_session.commit()

    eligible, reason, attempt_number = evaluate_recovery_policy(db_session, order)
    assert eligible is False
    assert attempt_number == 2
    assert "max" in reason.lower()


def test_already_paid_order_cannot_be_retried(client, db_session):
    order = MerchantOrder(razorpay_order_id="order_paid_retry", amount=50000, currency="INR", customer_id="cust_paid", status="paid")
    db_session.add(order)
    db_session.commit()

    response = client.post("/payments/order/order_paid_retry/retry")
    assert response.status_code == 400
    assert "already paid" in response.json()["detail"].lower()


def test_webhook_signature_validation():
    body = b'{"event":"payment.captured","payload":{"payment":{"entity":{"id":"pay_sig_1","order_id":"order_sig_1"}}}}'
    secret = "webhook_secret"
    signature = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = secret
    try:
        assert verify_signature(body, signature) is True
        assert verify_signature(body, "bad-signature") is False
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original


def test_webhook_idempotency(client, db_session, monkeypatch):
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"

    order = MerchantOrder(razorpay_order_id="order_webhook_id", amount=60000, currency="INR", customer_id="cust_w", status="created")
    db_session.add(order)
    db_session.commit()

    payload = {
        "event": "payment.failed",
        "event_id": "evt_123",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_evt_123",
                    "order_id": "order_webhook_id",
                    "amount": 60000,
                    "method": "card",
                    "error_code": "PAYMENT_ERROR",
                    "error_description": "Declined",
                }
            }
        },
    }
    body = json.dumps(payload).encode("utf-8")
    signature = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()

    try:
        first = client.post("/webhooks/razorpay", content=body, headers={"X-Razorpay-Signature": signature})
        second = client.post("/webhooks/razorpay", content=body, headers={"X-Razorpay-Signature": signature})
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["status"] == "ok"
        assert second.json()["status"] == "duplicate"
        assert db_session.query(WebhookEvent).count() == 1
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original


def test_payment_failed_processing(client, db_session):
    order = MerchantOrder(razorpay_order_id="order_failed_001", amount=40000, currency="INR", customer_id="cust_fail", status="created")
    db_session.add(order)
    db_session.commit()

    payload = {
        "event": "payment.failed",
        "event_id": "evt_failed_1",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_failed_1",
                    "order_id": "order_failed_001",
                    "amount": 40000,
                    "method": "card",
                    "error_code": "PAYMENT_ERROR",
                    "error_description": "Bank declined",
                }
            }
        },
    }
    body = json.dumps(payload).encode("utf-8")
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"
    signature = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()
    try:
        response = client.post("/webhooks/razorpay", content=body, headers={"X-Razorpay-Signature": signature})
        assert response.status_code == 200
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original

    assert db_session.query(PaymentAttempt).filter(PaymentAttempt.razorpay_order_id == "order_failed_001").count() == 1
    order_after = db_session.query(MerchantOrder).filter(MerchantOrder.razorpay_order_id == "order_failed_001").one()
    assert order_after.status == "recoverable"


def test_payment_captured_processing(client, db_session):
    order = MerchantOrder(razorpay_order_id="order_captured_001", amount=90000, currency="INR", customer_id="cust_cap", status="created")
    db_session.add(order)
    db_session.commit()

    payload = {
        "event": "payment.captured",
        "event_id": "evt_captured_1",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_cap_1",
                    "order_id": "order_captured_001",
                    "amount": 90000,
                    "status": "captured",
                    "method": "upi",
                }
            }
        },
    }
    body = json.dumps(payload).encode("utf-8")
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"
    signature = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()
    try:
        response = client.post("/webhooks/razorpay", content=body, headers={"X-Razorpay-Signature": signature})
        assert response.status_code == 200
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original

    updated_order = db_session.query(MerchantOrder).filter(MerchantOrder.razorpay_order_id == "order_captured_001").one()
    assert updated_order.status == "paid"


def test_order_paid_processing(client, db_session):
    order = MerchantOrder(razorpay_order_id="order_paid_001", amount=70000, currency="INR", customer_id="cust_paid_order", status="created")
    db_session.add(order)
    db_session.commit()

    payload = {
        "event": "order.paid",
        "event_id": "evt_order_paid_1",
        "payload": {
            "order": {"entity": {"id": "order_paid_001", "status": "paid"}}
        },
    }
    body = json.dumps(payload).encode("utf-8")
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"
    signature = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()
    try:
        response = client.post("/webhooks/razorpay", content=body, headers={"X-Razorpay-Signature": signature})
        assert response.status_code == 200
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original

    updated_order = db_session.query(MerchantOrder).filter(MerchantOrder.razorpay_order_id == "order_paid_001").one()
    assert updated_order.status == "paid"


def test_recovery_metrics(db_session):
    db_session.add_all([
        MerchantOrder(razorpay_order_id="metric_order_1", amount=10000, currency="INR", customer_id="cust_a", status="paid"),
        MerchantOrder(razorpay_order_id="metric_order_2", amount=20000, currency="INR", customer_id="cust_b", status="paid"),
        MerchantOrder(razorpay_order_id="metric_order_3", amount=30000, currency="INR", customer_id="cust_c", status="paid"),
        MerchantOrder(razorpay_order_id="metric_order_4", amount=40000, currency="INR", customer_id="cust_d", status="exhausted"),
    ])
    db_session.commit()

    db_session.add_all([
        PaymentAttempt(razorpay_payment_id="metric_pay_1", razorpay_order_id="metric_order_1", amount=10000, status="captured", attempt_number=1, order_id=1),
        PaymentAttempt(razorpay_payment_id="metric_pay_2", razorpay_order_id="metric_order_2", amount=20000, status="failed", attempt_number=1, order_id=2),
        PaymentAttempt(razorpay_payment_id="metric_pay_3", razorpay_order_id="metric_order_3", amount=30000, status="captured", attempt_number=2, order_id=3),
    ])
    db_session.commit()

    db_session.add_all([
        RecoveryAction(order_id=2, action_type="recovery_candidate", reason="eligible", attempt_number=1, status="active", case_status="active"),
        RecoveryAction(order_id=2, action_type="retry_payment", reason="retry", attempt_number=1, status="active", case_status="active"),
        RecoveryAction(order_id=2, action_type="recovery_success", reason="paid after retry", attempt_number=2, status="success", case_status="recovered"),
        RecoveryAction(order_id=4, action_type="recovery_exhausted", reason="max attempts", attempt_number=1, status="exhausted", case_status="exhausted"),
    ])
    db_session.commit()

    metrics = compute_dashboard_metrics(db_session)
    assert metrics["total_orders"] == 4
    assert metrics["successful_orders"] == 3
    assert metrics["recovery_cases_history"] == 2
    assert metrics["active_recovery_candidates"] == 0
    assert metrics["recovery_attempts"] == 1
    assert metrics["recovered_orders"] == 1
    assert metrics["recovered_revenue"] == 20000
    assert metrics["recovery_rate"] == 0.5
    assert metrics["revenue_recovery_rate"] == 1 / 3
    assert metrics["exhausted_escalated_cases"] == 1


def test_recovery_case_history_tracks_success_after_retry(db_session):
    order = MerchantOrder(razorpay_order_id="recovery_history_order", amount=50000, currency="INR", customer_id="cust_recover", status="created")
    db_session.add(order)
    db_session.commit()
    db_session.refresh(order)

    db_session.add(PaymentAttempt(
        razorpay_payment_id="pay_fail_history_1",
        razorpay_order_id=order.razorpay_order_id,
        amount=50000,
        status="failed",
        method="card",
        failure_code="PAYMENT_ERROR",
        failure_description="Bank declined",
        attempt_number=1,
        order_id=order.internal_id,
    ))
    db_session.commit()

    mark_order_recoverable(db_session, order)
    register_retry_attempt(db_session, order)

    db_session.add(PaymentAttempt(
        razorpay_payment_id="pay_capture_history_2",
        razorpay_order_id=order.razorpay_order_id,
        amount=50000,
        status="captured",
        method="upi",
        attempt_number=2,
        order_id=order.internal_id,
    ))
    db_session.commit()
    mark_recovery_success(db_session, order, attempt_number=2)

    metrics = compute_dashboard_metrics(db_session)
    assert metrics["recovery_cases_history"] == 1
    assert metrics["active_recovery_candidates"] == 0
    assert metrics["recovery_attempts"] == 1
    assert metrics["recovered_orders"] == 1
    assert metrics["recovered_revenue"] == 50000
    assert metrics["recovery_rate"] == 1.0
    assert metrics["revenue_recovery_rate"] == 1.0


def test_synthetic_baseline_route(client):
    response = client.get("/simulation/baseline")
    assert response.status_code == 200
    payload = response.json()
    assert payload["seed"] == 42
    assert payload["total_orders"] == 1000
    assert "recovery_rate" in payload
    assert "revenue_recovery_rate" in payload
    assert "cases" in payload
    assert len(payload["cases"]) <= 5


def test_synthetic_policy_is_deterministic():
    from backend.services.synthetic_simulation import generate_synthetic_cases, evaluate_synthetic_policy

    first_batch = generate_synthetic_cases(seed=123, count=12)
    second_batch = generate_synthetic_cases(seed=123, count=12)
    assert [case.order_id for case in first_batch] == [case.order_id for case in second_batch]

    first_decisions = [evaluate_synthetic_policy(case)[0] for case in first_batch]
    second_decisions = [evaluate_synthetic_policy(case)[0] for case in second_batch]
    assert first_decisions == second_decisions


def test_frontend_loads_live_public_key():
    source = (Path(__file__).resolve().parent.parent / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "apiFetch('config')" in source
    assert "razorpayKeyId = config.razorpay_key_id" in source
    assert "loadRazorpayConfig()" in source
