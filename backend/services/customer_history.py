from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from backend.models import MerchantOrder, PaymentAttempt


@dataclass(frozen=True)
class CustomerPaymentHistory:
    customer_id: str
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    success_rate: float
    attempts_by_method: dict[str, int]
    successes_by_method: dict[str, int]
    failures_by_method: dict[str, int]
    method_success_rates: dict[str, float]
    customer_average_order_value: float
    customer_total_spend: float
    customer_last_success_age: int
    customer_last_failure_age: int
    most_recent_successful_method: str | None
    most_recent_failed_method: str | None

    def for_method(self, method: str) -> dict[str, Any]:
        attempts = self.attempts_by_method.get(method, 0)
        successes = self.successes_by_method.get(method, 0)
        failures = self.failures_by_method.get(method, 0)
        return {
            "customer_method_attempts": attempts,
            "customer_method_successes": successes,
            "customer_method_failures": failures,
            "customer_method_success_rate": self.method_success_rates.get(method, 0.0),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "failed_attempts": self.failed_attempts,
            "success_rate": self.success_rate,
            "attempts_by_method": dict(self.attempts_by_method),
            "successes_by_method": dict(self.successes_by_method),
            "failures_by_method": dict(self.failures_by_method),
            "method_success_rates": dict(self.method_success_rates),
            "customer_average_order_value": self.customer_average_order_value,
            "customer_total_spend": self.customer_total_spend,
            "customer_last_success_age": self.customer_last_success_age,
            "customer_last_failure_age": self.customer_last_failure_age,
            "most_recent_successful_method": self.most_recent_successful_method,
            "most_recent_failed_method": self.most_recent_failed_method,
        }


def get_customer_payment_history(
    db: Session,
    customer_id: str,
    *,
    exclude_order_id: int | None = None,
    exclude_attempt_id: int | None = None,
) -> CustomerPaymentHistory:
    """Aggregate only persisted observations available before a decision."""
    query = (
        db.query(PaymentAttempt)
        .join(MerchantOrder, PaymentAttempt.order_id == MerchantOrder.internal_id)
        .filter(MerchantOrder.customer_id == customer_id)
    )
    if exclude_order_id is not None:
        query = query.filter(PaymentAttempt.order_id != exclude_order_id)
    if exclude_attempt_id is not None:
        query = query.filter(PaymentAttempt.internal_id != exclude_attempt_id)

    attempts = query.order_by(PaymentAttempt.created_at.asc(), PaymentAttempt.internal_id.asc()).all()
    attempts_by_method: dict[str, int] = {}
    successes_by_method: dict[str, int] = {}
    failures_by_method: dict[str, int] = {}
    for attempt in attempts:
        method = (attempt.method or "unknown").lower()
        attempts_by_method[method] = attempts_by_method.get(method, 0) + 1
        if attempt.status in {"captured", "authorized", "success", "recovered"}:
            successes_by_method[method] = successes_by_method.get(method, 0) + 1
        elif attempt.status in {"failed", "attempt_failed", "recovery_retry"}:
            failures_by_method[method] = failures_by_method.get(method, 0) + 1

    total_attempts = len(attempts)
    successful_attempts = sum(successes_by_method.values())
    failed_attempts = sum(failures_by_method.values())
    successful_amounts = [
        float(attempt.amount)
        for attempt in attempts
        if attempt.status in {"captured", "authorized", "success", "recovered"}
    ]
    method_success_rates = {
        method: successes_by_method.get(method, 0) / count
        for method, count in attempts_by_method.items()
    }
    latest_success = next(
        (attempt for attempt in reversed(attempts) if attempt.status in {"captured", "authorized", "success", "recovered"}),
        None,
    )
    latest_failure = next(
        (attempt for attempt in reversed(attempts) if attempt.status in {"failed", "attempt_failed", "recovery_retry"}),
        None,
    )
    now = datetime.utcnow()
    success_age = (
        max(0, int((now - latest_success.created_at).total_seconds() / 60))
        if latest_success and latest_success.created_at else 0
    )
    failure_age = (
        max(0, int((now - latest_failure.created_at).total_seconds() / 60))
        if latest_failure and latest_failure.created_at else 0
    )
    return CustomerPaymentHistory(
        customer_id=customer_id,
        total_attempts=total_attempts,
        successful_attempts=successful_attempts,
        failed_attempts=failed_attempts,
        success_rate=successful_attempts / total_attempts if total_attempts else 0.0,
        attempts_by_method=attempts_by_method,
        successes_by_method=successes_by_method,
        failures_by_method=failures_by_method,
        method_success_rates=method_success_rates,
        customer_average_order_value=(
            sum(successful_amounts) / len(successful_amounts)
            if successful_amounts else 0.0
        ),
        customer_total_spend=sum(successful_amounts),
        customer_last_success_age=success_age,
        customer_last_failure_age=failure_age,
        most_recent_successful_method=latest_success.method if latest_success else None,
        most_recent_failed_method=latest_failure.method if latest_failure else None,
    )
