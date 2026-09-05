from __future__ import annotations

from backend import config

PAYMENT_METHODS = ["card", "upi", "wallet", "netbanking", "emi"]
CUSTOMER_HISTORY_SEGMENTS = ["strong", "stable", "mixed", "risky"]

TRANSIENT_FAILURE_CODES = (
    "NETWORK_ERROR",
    "TIMEOUT",
    "TEMPORARY_ISSUE",
    "PROCESSING_ERROR",
    "PAYMENT_ERROR",
)

NON_RECOVERABLE_FAILURE_CODES = (
    "AUTHENTICATION_ERROR",
    "INSUFFICIENT_FUNDS",
    "CARD_DECLINED",
    "BANK_DECLINED",
    "INVALID_ACCOUNT",
    "CURRENCY_NOT_ALLOWED",
)

FAILURE_TAXONOMY = {
    "PAYMENT_ERROR": "transient / potentially recoverable",
    "NETWORK_ERROR": "transient / potentially recoverable",
    "TIMEOUT": "transient / potentially recoverable",
    "TEMPORARY_ISSUE": "transient / potentially recoverable",
    "PROCESSING_ERROR": "transient / potentially recoverable",
    "CARD_DECLINED": "non-recoverable",
    "INSUFFICIENT_FUNDS": "non-recoverable",
    "AUTHENTICATION_ERROR": "non-recoverable",
    "BANK_DECLINED": "non-recoverable",
    "INVALID_ACCOUNT": "non-recoverable",
    "CURRENCY_NOT_ALLOWED": "non-recoverable",
}

MAX_RECOVERY_ATTEMPTS = 4
BASELINE_MAX_RECOVERY_INTERVENTIONS = 2
TIME_SINCE_FAILURE_MINUTES_MIN = 0
TIME_SINCE_FAILURE_MINUTES_MAX = 2880  # 48 hours
TIME_SINCE_FAILURE_MINUTES_RANGE = (TIME_SINCE_FAILURE_MINUTES_MIN, TIME_SINCE_FAILURE_MINUTES_MAX)

CUSTOMER_HISTORY_SEMANTICS = {
    "strong": "reliable customer with historically high success rate and lower failure risk",
    "stable": "moderately healthy customer with regular payment behavior",
    "mixed": "customer with a moderate blend of successful and failed payments",
    "risky": "customer with repeated failures and lower historical payment reliability",
}

FAILURE_CODE_ORDER = [
    "PAYMENT_ERROR",
    "NETWORK_ERROR",
    "TIMEOUT",
    "TEMPORARY_ISSUE",
    "PROCESSING_ERROR",
    "CARD_DECLINED",
    "INSUFFICIENT_FUNDS",
    "AUTHENTICATION_ERROR",
    "BANK_DECLINED",
    "INVALID_ACCOUNT",
    "CURRENCY_NOT_ALLOWED",
]


def canonical_failure_codes() -> list[str]:
    return list(FAILURE_CODE_ORDER)


def canonical_customer_history_segment_labels() -> list[str]:
    return list(CUSTOMER_HISTORY_SEGMENTS)
