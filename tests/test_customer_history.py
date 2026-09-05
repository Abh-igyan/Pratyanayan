from __future__ import annotations

import pytest
from types import SimpleNamespace
from sqlalchemy.orm import Session

from backend.database import Base, engine
from backend.agent import RecoveryAgent
from backend.models import MerchantOrder, PaymentAttempt
from backend.services.customer_history import get_customer_payment_history


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


def add_order(db, order_id, customer_id, amount=10000):
    order = MerchantOrder(
        razorpay_order_id=order_id,
        amount=amount,
        currency="INR",
        customer_id=customer_id,
        status="created",
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


def add_attempt(db, order, payment_id, method, status, number=1):
    attempt = PaymentAttempt(
        razorpay_payment_id=payment_id,
        razorpay_order_id=order.razorpay_order_id,
        amount=order.amount,
        status=status,
        method=method,
        failure_code="TIMEOUT" if status == "failed" else None,
        attempt_number=number,
        order_id=order.internal_id,
    )
    db.add(attempt)
    db.commit()
    db.refresh(attempt)
    return attempt


def test_customer_history_aggregates_across_orders_and_methods(db_session):
    first = add_order(db_session, "history_a", "customer_1", 10000)
    second = add_order(db_session, "history_b", "customer_1", 20000)
    add_attempt(db_session, first, "pay_a", "card", "captured")
    add_attempt(db_session, first, "pay_b", "card", "captured", 2)
    add_attempt(db_session, second, "pay_c", "upi", "captured")
    add_attempt(db_session, second, "pay_d", "wallet", "failed", 2)

    history = get_customer_payment_history(db_session, "customer_1")

    assert history.total_attempts == 4
    assert history.successful_attempts == 3
    assert history.failed_attempts == 1
    assert history.success_rate == 0.75
    assert history.attempts_by_method == {"card": 2, "upi": 1, "wallet": 1}
    assert history.successes_by_method == {"card": 2, "upi": 1}
    assert history.failures_by_method == {"wallet": 1}
    assert history.method_success_rates["card"] == 1.0
    assert history.method_success_rates["wallet"] == 0.0
    assert history.most_recent_successful_method == "upi"
    assert history.most_recent_failed_method == "wallet"


def test_current_order_attempt_is_excluded_before_decision(db_session):
    prior = add_order(db_session, "prior_order", "customer_2")
    current = add_order(db_session, "current_order", "customer_2")
    add_attempt(db_session, prior, "prior_card", "card", "captured")
    current_failure = add_attempt(db_session, current, "current_wallet", "wallet", "failed")

    history = get_customer_payment_history(
        db_session,
        "customer_2",
        exclude_order_id=current.internal_id,
        exclude_attempt_id=current_failure.internal_id,
    )
    assert history.total_attempts == 1
    assert history.method_success_rates == {"card": 1.0}

    agent = RecoveryAgent.__new__(RecoveryAgent)
    agent.db = db_session
    case = agent._build_case_dict(current, current_failure, 1, "card", history)
    assert case["customer_method_success_rate"] == 1.0
    assert case["customer_failed_orders"] == 0


def test_successful_payment_updates_future_customer_history(db_session):
    order = add_order(db_session, "feedback_order", "customer_3")
    add_attempt(db_session, order, "feedback_card", "card", "captured")

    before = get_customer_payment_history(db_session, "customer_3")
    assert before.successful_attempts == 1

    later = add_order(db_session, "feedback_order_2", "customer_3")
    add_attempt(db_session, later, "feedback_upi", "upi", "captured")
    after = get_customer_payment_history(db_session, "customer_3")
    assert after.successful_attempts == 2
    assert after.method_success_rates["upi"] == 1.0


def test_customer_histories_are_isolated(db_session):
    first = add_order(db_session, "isolated_a", "customer_a")
    second = add_order(db_session, "isolated_b", "customer_b")
    add_attempt(db_session, first, "isolated_card", "card", "captured")
    add_attempt(db_session, second, "isolated_wallet", "wallet", "failed")

    first_history = get_customer_payment_history(db_session, "customer_a")
    second_history = get_customer_payment_history(db_session, "customer_b")
    assert first_history.method_success_rates == {"card": 1.0}
    assert second_history.method_success_rates == {"wallet": 0.0}


def test_longitudinal_history_changes_actual_agent_plan(db_session, monkeypatch):
    class HistoryAwareEngine:
        def decide(self, case):
            rate = case["customer_method_success_rate"]
            return SimpleNamespace(
                decision="RETRY",
                reason="history-aware test decision",
                recovery_probability=rate,
                expected_recovered_value=rate * case["amount"],
                net_expected_value=rate * case["amount"],
                intervention_cost=0.0,
                to_dict=lambda: {
                    "decision": "RETRY",
                    "recovery_probability": rate,
                    "expected_recovered_value": rate * case["amount"],
                    "net_expected_value": rate * case["amount"],
                },
            )

    def run_customer(customer_id, card_status, upi_status):
        prior_card = add_order(db_session, f"{customer_id}_card", customer_id)
        prior_upi = add_order(db_session, f"{customer_id}_upi", customer_id)
        add_attempt(db_session, prior_card, f"{customer_id}_card_pay", "card", card_status)
        add_attempt(db_session, prior_upi, f"{customer_id}_upi_pay", "upi", upi_status)
        current = add_order(db_session, f"{customer_id}_current", customer_id)
        current_failure = add_attempt(db_session, current, f"{customer_id}_current_pay", "wallet", "failed")
        agent = RecoveryAgent.__new__(RecoveryAgent)
        agent.db = db_session
        agent.engine = HistoryAwareEngine()
        trace = agent.observe_failure(current, current_failure)
        return trace, current_failure

    customer_a_trace, customer_a_attempt = run_customer("history_agent_a", "captured", "failed")
    customer_b_trace, customer_b_attempt = run_customer("history_agent_b", "failed", "captured")

    assert customer_a_trace.planned_method == "card"
    assert customer_b_trace.planned_method == "upi"
    assert customer_a_trace.decision_inputs != customer_b_trace.decision_inputs
    assert customer_a_attempt.method == "wallet"
    assert customer_b_attempt.method == "wallet"
    assert "history_agent_a" not in customer_b_trace.decision_inputs
