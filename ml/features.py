from __future__ import annotations

TARGET_COLUMN = "recovery_success"

FEATURE_COLUMNS = [
    "amount",
    "payment_method",
    "failure_code",
    "attempt_number",
    "time_since_failure",
    "customer_total_orders",
    "customer_successful_orders",
    "customer_failed_orders",
    "customer_success_rate",
    "customer_failure_rate",
    "customer_average_order_value",
    "customer_total_spend",
    "customer_last_success_age",
    "customer_last_failure_age",
    "customer_method_attempts",
    "customer_method_successes",
    "customer_method_failures",
    "customer_method_success_rate",
    "customer_history_segment",
    "order_value_percentile_for_customer",
]

CATEGORICAL_FEATURES = [
    "payment_method",
    "failure_code",
    "customer_history_segment",
]

NUMERIC_FEATURES = [
    column for column in FEATURE_COLUMNS if column not in CATEGORICAL_FEATURES
]
