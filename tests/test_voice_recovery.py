from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend import config
from backend.services.voice_recovery import (
    _extract_content,
    _valid_message,
    build_voice_facts,
    deterministic_fallback,
    generate_recovery_message,
    sanitize_speech_text,
)


# ---------------------------------------------------------------------------
# build_voice_facts — allowlist & sanitisation
# ---------------------------------------------------------------------------

def test_voice_facts_are_allowlisted_and_do_not_contain_financial_decision_fields():
    facts = build_voice_facts(
        scenario="payment.failed",
        decision="SWITCH_PAYMENT_METHOD",
        recommended_method="card",
        reason="model net EV is 999999",
        customer_history={"total_attempts": 3, "method_success_rates": {"card": 1.0}},
    )
    assert facts["scenario"] == "payment.failed"
    assert facts["decision"] == "SWITCH_PAYMENT_METHOD"
    assert "probability" not in facts
    assert "net_ev" not in facts
    assert facts["customer_history"]["recommended_method_success_rate"] == 1.0


def test_voice_facts_sanitises_unknown_scenario_and_decision():
    facts = build_voice_facts(
        scenario="unknown_event",
        decision="DO_MAGIC",
        recommended_method=None,
        reason="test",
        customer_history=None,
    )
    assert facts["scenario"] == "payment.failed"
    assert facts["decision"] == "STOP"


def test_voice_facts_reason_is_truncated_to_240():
    long_reason = "x" * 500
    facts = build_voice_facts(
        scenario="payment.failed",
        decision="RETRY",
        recommended_method=None,
        reason=long_reason,
        customer_history=None,
    )
    assert len(facts["reason"]) == 240


# ---------------------------------------------------------------------------
# deterministic_fallback — scenario × decision coverage
# ---------------------------------------------------------------------------

def _facts(scenario="payment.failed", decision="RETRY", method=None):
    return build_voice_facts(
        scenario=scenario,
        decision=decision,
        recommended_method=method,
        reason="test",
        customer_history=None,
    )


def test_fallback_retry_with_method_mentions_method():
    msg = deterministic_fallback(_facts(decision="RETRY", method="card"))
    assert "CARD" in msg
    assert "checkout" in msg.lower() or "control" in msg.lower()


def test_fallback_switch_method_mentions_recommended_method():
    msg = deterministic_fallback(_facts(decision="SWITCH_PAYMENT_METHOD", method="upi"))
    assert "UPI" in msg
    # Must not claim system charged the customer
    assert "charge" not in msg.lower()
    assert "debit" not in msg.lower()


def test_fallback_wait_does_not_push_immediate_retry():
    msg = deterministic_fallback(_facts(decision="WAIT"))
    assert "baad" in msg.lower() or "later" in msg.lower() or "thodi" in msg.lower()
    # should not force immediate action
    assert "abhi" not in msg.lower()


def test_fallback_stop_does_not_encourage_unlimited_retries():
    msg = deterministic_fallback(_facts(decision="STOP"))
    # Should not say "retry now" unconditionally but may mention option
    assert "charge" not in msg.lower()
    assert "guaranteed" not in msg.lower()


def test_fallback_payment_captured_is_positive():
    msg = deterministic_fallback(_facts(scenario="payment.captured", decision="RETRY"))
    assert "successfully" in msg.lower() or "complete" in msg.lower()


def test_fallback_checkout_abandoned_has_correct_prefix():
    msg = deterministic_fallback(_facts(scenario="checkout.abandoned", decision="RETRY", method="wallet"))
    assert "checkout" in msg.lower() or "complete nahi" in msg.lower()


def test_fallback_is_deterministic():
    facts = _facts(decision="SWITCH_PAYMENT_METHOD", method="upi")
    assert deterministic_fallback(facts) == deterministic_fallback(facts)


# ---------------------------------------------------------------------------
# _valid_message — validation rules
# ---------------------------------------------------------------------------

def test_valid_message_rejects_forbidden_words():
    assert not _valid_message("Payment guaranteed!")
    assert not _valid_message("We will charge you now.")
    assert not _valid_message("net EV is 99")
    assert not _valid_message("recovery probability is 0.9")


def test_valid_message_rejects_empty():
    assert not _valid_message("")
    assert not _valid_message("   ")


def test_valid_message_rejects_too_long():
    assert not _valid_message("x" * 281)


def test_valid_message_accepts_normal_hinglish():
    assert _valid_message("Aapka payment fail ho gaya. Kripya dobara try karein.")


# ---------------------------------------------------------------------------
# _extract_content — handles reasoning-model empty content
# ---------------------------------------------------------------------------

def test_extract_content_standard_model():
    payload = {"choices": [{"message": {"content": "Dobara try karein.", "role": "assistant"}}]}
    assert _extract_content(payload) == "Dobara try karein."


def test_extract_content_reasoning_model_falls_through_to_reasoning_key():
    payload = {"choices": [{"message": {"content": "", "reasoning": "Aapka payment fail hua.", "role": "assistant"}}]}
    assert _extract_content(payload) == "Aapka payment fail hua."


def test_extract_content_returns_empty_string_on_missing_keys():
    assert _extract_content({}) == ""
    assert _extract_content({"choices": []}) == ""


# ---------------------------------------------------------------------------
# generate_recovery_message — fallback when LLM not configured
# ---------------------------------------------------------------------------

def test_voice_layer_falls_back_when_llm_is_unavailable(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_URL", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    result = generate_recovery_message(
        scenario="payment.failed",
        decision="RETRY",
        recommended_method="card",
        reason="payment failed",
        customer_history=None,
    )
    assert result["source"] == "fallback"
    assert result["message"]
    assert result["provider_configured"] is False


def test_voice_layer_response_contains_diagnostic_fields(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_URL", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    result = generate_recovery_message(
        scenario="payment.failed",
        decision="STOP",
        recommended_method=None,
        reason="test",
        customer_history=None,
    )
    assert "provider_configured" in result
    assert "model" in result
    assert "source" in result
    assert "facts" in result


def test_voice_layer_response_does_not_leak_api_key(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_URL", "")
    monkeypatch.setattr(config, "LLM_API_KEY", "super-secret-key")
    result = generate_recovery_message(
        scenario="payment.failed",
        decision="STOP",
        recommended_method=None,
        reason="test",
        customer_history=None,
    )
    result_str = str(result)
    assert "super-secret-key" not in result_str


# ---------------------------------------------------------------------------
# generate_recovery_message — fallback when LLM returns invalid message
# ---------------------------------------------------------------------------

def test_voice_layer_falls_back_when_llm_returns_invalid_message(monkeypatch):
    monkeypatch.setattr(config, "LLM_API_URL", "https://llm.test/v1/chat/completions")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_MODEL", "test-model")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Payment guaranteed. I will charge you now.", "role": "assistant"}}]
    }
    monkeypatch.setattr("backend.services.voice_recovery.httpx.post", lambda *a, **kw: mock_response)

    result = generate_recovery_message(
        scenario="payment.failed",
        decision="STOP",
        recommended_method=None,
        reason="low value",
        customer_history=None,
    )
    assert result["source"] == "fallback"
    assert "charge" not in result["message"].lower()
    assert result["facts"]["decision"] == "STOP"


def test_voice_layer_falls_back_when_llm_returns_empty_content(monkeypatch):
    """Simulates reasoning model that returns empty content field."""
    monkeypatch.setattr(config, "LLM_API_URL", "https://llm.test/v1/chat/completions")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_MODEL", "test-model")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "", "reasoning": "", "role": "assistant"}}]
    }
    monkeypatch.setattr("backend.services.voice_recovery.httpx.post", lambda *a, **kw: mock_response)

    result = generate_recovery_message(
        scenario="payment.failed",
        decision="RETRY",
        recommended_method="card",
        reason="test",
        customer_history=None,
    )
    assert result["source"] == "fallback"
    assert result["message"]


def test_voice_layer_uses_llm_when_valid_response(monkeypatch):
    """Simulates a good Groq response."""
    monkeypatch.setattr(config, "LLM_API_URL", "https://llm.test/v1/chat/completions")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_MODEL", "groq/compound-mini")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Aapka payment fail hua. CARD se dobara try karein.", "role": "assistant"}}]
    }
    monkeypatch.setattr("backend.services.voice_recovery.httpx.post", lambda *a, **kw: mock_response)

    result = generate_recovery_message(
        scenario="payment.failed",
        decision="SWITCH_PAYMENT_METHOD",
        recommended_method="card",
        reason="better EV",
        customer_history={"total_attempts": 3, "method_success_rates": {"card": 0.9}},
    )
    assert result["source"] == "llm"
    assert "CARD" in result["message"] or "card" in result["message"].lower()
    assert result["provider_configured"] is True


def test_voice_layer_falls_back_on_http_error(monkeypatch):
    import httpx as _httpx
    monkeypatch.setattr(config, "LLM_API_URL", "https://llm.test/v1/chat/completions")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_MODEL", "test-model")

    def raise_http(*a, **kw):
        raise _httpx.ConnectError("connection refused")

    monkeypatch.setattr("backend.services.voice_recovery.httpx.post", raise_http)

    result = generate_recovery_message(
        scenario="payment.failed",
        decision="RETRY",
        recommended_method=None,
        reason="test",
        customer_history=None,
    )
    assert result["source"] == "fallback"
    assert result["message"]


# ---------------------------------------------------------------------------
# LLM cannot modify the underlying agent decision
# ---------------------------------------------------------------------------

def test_llm_cannot_alter_decision_in_facts(monkeypatch):
    """The agent decision (facts.decision) must be immutable regardless of LLM response content."""
    monkeypatch.setattr(config, "LLM_API_URL", "https://llm.test/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config, "LLM_MODEL", "test-model")

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    # LLM generates persuasive text, but the agent already decided RETRY
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Aapka payment fail hua, baad mein try karein.", "role": "assistant"}}]
    }
    monkeypatch.setattr("backend.services.voice_recovery.httpx.post", lambda *a, **kw: mock_response)

    result = generate_recovery_message(
        scenario="payment.failed",
        decision="RETRY",   # agent's actual decision — must not change
        recommended_method="card",
        reason="transient failure",
        customer_history=None,
    )
    # The facts.decision must still be RETRY regardless of what the LLM said
    assert result["facts"]["decision"] == "RETRY"
    # The LLM cannot inject a different decision into the response
    assert "decision" not in result or result.get("decision") is None  # no top-level decision override


# ---------------------------------------------------------------------------
# Frontend UX voice contract regression tests
# ---------------------------------------------------------------------------

def test_frontend_has_single_global_voice_control_and_no_per_step_voice_buttons():
    from pathlib import Path
    html_path = Path("frontend/index.html")
    js_path = Path("frontend/app.js")
    
    assert html_path.exists()
    assert js_path.exists()
    
    html_content = html_path.read_text(encoding="utf-8")
    js_content = js_path.read_text(encoding="utf-8")
    
    # 1. Exactly ONE global voice toggle control in HTML
    assert 'id="voice-toggle-btn"' in html_content
    assert html_content.count('id="voice-toggle-btn"') == 1
    
    # 2. No per-step voice-button class added to agent steps
    assert 'class="voice-button"' not in html_content
    assert 'class="voice-button"' not in js_content
    
    # 3. Exactly one voice-live-banner container for voice output
    assert 'id="voice-live-banner"' in html_content
    assert html_content.count('id="voice-live-banner"') == 1
    
    # 4. JavaScript implements consent gate, mute control, and deduplication
    assert "isVoiceEnabled" in js_content
    assert "lastPlayedKey" in js_content
    assert "updateVoiceUI" in js_content
    assert "stopVoicePlayback" in js_content
    assert "speechSynthesis.cancel()" in js_content


# ---------------------------------------------------------------------------
# Speech Quality & Deterministic Sanitization Tests
# ---------------------------------------------------------------------------

def test_sanitize_speech_text_removes_emojis():
    # Emojis like folded hands, thumbs up, smileys must be stripped
    raw = "Namaste! \U0001F64F Aapka payment retry karein \U0001F44D \u2705 \U0001F60A"
    cleaned = sanitize_speech_text(raw)
    assert "\U0001F64F" not in cleaned
    assert "\U0001F44D" not in cleaned
    assert "\u2705" not in cleaned
    assert "\U0001F60A" not in cleaned
    assert "Namaste! Aapka payment retry karein" == cleaned


def test_sanitize_speech_text_removes_internal_technical_tokens():
    raw = "Agent Step 1: payment.failed recorded. checkout.abandoned processed. Net EV is high."
    cleaned = sanitize_speech_text(raw)
    assert "payment.failed" not in cleaned.lower()
    assert "checkout.abandoned" not in cleaned.lower()
    assert "agent step" not in cleaned.lower()
    assert "net ev" not in cleaned.lower()


def test_sanitize_speech_text_removes_technical_step_numbering():
    raw = "Aap step 1 complete karein."
    cleaned = sanitize_speech_text(raw)
    assert "step 1" not in cleaned.lower()


def test_sanitize_speech_text_strips_markdown_formatting_and_bullets():
    raw = "* Aapka payment fail hua.\n- Kripya **UPI** se try karein: \"checkout\" link."
    cleaned = sanitize_speech_text(raw)
    assert "*" not in cleaned
    assert "**" not in cleaned
    assert '"' not in cleaned
    assert "- " not in cleaned
    assert "Aapka payment fail hua. Kripya UPI se try karein: checkout link." == cleaned


def test_valid_message_rejects_emojis():
    assert not _valid_message("Aapka payment fail hua \U0001F64F")
    assert not _valid_message("Kripya retry karein \u2705")


def test_valid_message_rejects_internal_event_and_step_names():
    assert not _valid_message("Event payment.failed received.")
    assert not _valid_message("Agent Step 1 completed.")
    assert not _valid_message("Net EV calculated.")


def test_fallback_messages_are_speech_safe_and_emoji_free():
    for scenario in ["payment.failed", "checkout.abandoned", "payment.captured", "order.paid"]:
        for decision in ["RETRY", "WAIT", "STOP", "SWITCH_PAYMENT_METHOD"]:
            facts = build_voice_facts(
                scenario=scenario,
                decision=decision,
                recommended_method="upi",
                reason="testing speech safety",
                customer_history=None,
            )
            msg = deterministic_fallback(facts)
            assert _valid_message(msg)
            # Ensure no emojis or markdown artifacts
            assert sanitize_speech_text(msg) == msg


# ---------------------------------------------------------------------------
# Paid Order Safety & Idempotency Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    from backend.database import Base, engine
    from sqlalchemy.orm import Session
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
    from fastapi.testclient import TestClient
    from backend.database import get_db
    from backend.main import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _create_paid_order(db_session, order_id="order_paid_test_123"):
    from backend.models import MerchantOrder
    order = MerchantOrder(
        razorpay_order_id=order_id,
        amount=50000,
        currency="INR",
        customer_id="cust_paid_001",
        status="paid",
    )
    db_session.add(order)
    db_session.commit()
    db_session.refresh(order)
    return order


def test_late_payment_failed_webhook_ignored_for_already_paid_order(client, db_session, monkeypatch):
    import hmac
    import hashlib
    import json
    from backend.models import PaymentAttempt

    order = _create_paid_order(db_session, "order_late_fail_123")

    payload = {
        "event": "payment.failed",
        "event_id": "evt_late_fail_001",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_late_fail_001",
                    "order_id": order.razorpay_order_id,
                    "amount": order.amount,
                    "method": "card",
                    "error_code": "BAD_REQUEST_ERROR",
                    "error_description": "Payment was cancelled",
                }
            }
        }
    }
    body = json.dumps(payload).encode("utf-8")
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"
    sig = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()

    try:
        response = client.post(
            "/webhooks/razorpay",
            content=body,
            headers={"X-Razorpay-Signature": sig},
        )
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original

    assert response.status_code == 200
    assert response.json()["status"] == "already_paid"

    # Verify NO new PaymentAttempt was created
    attempt = db_session.query(PaymentAttempt).filter(
        PaymentAttempt.razorpay_payment_id == "pay_late_fail_001"
    ).first()
    assert attempt is None
    # Order status must remain paid
    db_session.refresh(order)
    assert order.status == "paid"


def test_duplicate_payment_captured_is_idempotent_for_paid_order(client, db_session):
    import hmac
    import hashlib
    import json
    from backend.models import PaymentAttempt

    order = _create_paid_order(db_session, "order_dup_cap_123")
    existing_attempt = PaymentAttempt(
        razorpay_payment_id="pay_already_captured_001",
        razorpay_order_id=order.razorpay_order_id,
        amount=order.amount,
        status="captured",
        method="upi",
        attempt_number=1,
        order_id=order.internal_id,
    )
    db_session.add(existing_attempt)
    db_session.commit()

    payload = {
        "event": "payment.captured",
        "event_id": "evt_dup_cap_001",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_already_captured_001",
                    "order_id": order.razorpay_order_id,
                    "amount": order.amount,
                    "status": "captured",
                    "method": "upi",
                }
            }
        }
    }
    body = json.dumps(payload).encode("utf-8")
    import backend.razorpay_client as razorpay_client
    original = razorpay_client.RAZORPAY_WEBHOOK_SECRET
    razorpay_client.RAZORPAY_WEBHOOK_SECRET = "webhook_secret"
    sig = hmac.new(b"webhook_secret", body, hashlib.sha256).hexdigest()

    try:
        response = client.post(
            "/webhooks/razorpay",
            content=body,
            headers={"X-Razorpay-Signature": sig},
        )
    finally:
        razorpay_client.RAZORPAY_WEBHOOK_SECRET = original

    assert response.status_code == 200
    assert response.json()["status"] == "already_processed"


def test_retry_on_paid_order_is_rejected(client, db_session):
    order = _create_paid_order(db_session, "order_retry_paid_123")

    response = client.post(f"/payments/order/{order.razorpay_order_id}/retry")
    assert response.status_code == 400
    assert "already paid" in response.json()["detail"]


def test_abandon_on_paid_order_returns_already_paid(client, db_session):
    order = _create_paid_order(db_session, "order_abandon_paid_123")

    response = client.post(f"/payments/order/{order.razorpay_order_id}/abandon")
    assert response.status_code == 200
    assert response.json()["status"] == "already_paid"


# ---------------------------------------------------------------------------
# Frontend Contract Tests: Paid Order UI Safety
# ---------------------------------------------------------------------------

def test_frontend_paid_order_ui_contract():
    from pathlib import Path
    js_content = Path("frontend/app.js").read_text(encoding="utf-8")

    # 1. renderOrder renders clear PAID badge and success indicator
    assert "const isPaid = (order.status || '').toLowerCase() === 'paid';" in js_content
    assert "PAID" in js_content
    assert "Payment completed successfully" in js_content

    # 2. renderOrder hides and disables checkout button when paid
    assert "checkoutButton.classList.add('hidden')" in js_content
    assert "checkoutButton.disabled = true" in js_content

    # 3. loadActiveRecoveryState suppresses retry buttons when order is paid
    assert "const orderIsPaid = currentOrder && (currentOrder.status || '').toLowerCase() === 'paid';" in js_content
    assert "!orderIsPaid" in js_content

    # 4. openCheckout guards against opening checkout for paid orders
    assert "alert('This order is already paid.')" in js_content