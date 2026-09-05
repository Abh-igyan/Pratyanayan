from __future__ import annotations

import pandas as pd

from backend.services.synthetic_simulation import (
    NON_RECOVERABLE_FAILURE_CODES as BENCHMARK_NON_RECOVERABLE,
    PAYMENT_METHODS as BENCHMARK_PAYMENT_METHODS,
    TRANSIENT_FAILURE_CODES as BENCHMARK_TRANSIENT,
)
from ml.features import FEATURE_COLUMNS, TARGET_COLUMN, CATEGORICAL_FEATURES
from ml.generate_dataset import generate_population
from ml.shared_semantics import enrich_case_features
from ml.synthetic_world import (
    CUSTOMER_HISTORY_SEGMENTS,
    FAILURE_CODE_ORDER,
    MAX_RECOVERY_ATTEMPTS,
    NON_RECOVERABLE_FAILURE_CODES as ML_NON_RECOVERABLE,
    PAYMENT_METHODS as ML_PAYMENT_METHODS,
    TIME_SINCE_FAILURE_MINUTES_RANGE,
    TRANSIENT_FAILURE_CODES as ML_TRANSIENT,
)


def test_generate_population_matches_schema_and_size():
    df = generate_population(n_cases=2000, seeds=[7], customer_count=300)
    assert len(df) == 2000
    assert TARGET_COLUMN in df.columns
    assert set(FEATURE_COLUMNS).issubset(set(df.columns))
    assert "customer_id" in df.columns
    assert df[TARGET_COLUMN].isin([0, 1]).all()


def test_categorical_features_are_declared():
    assert "payment_method" in CATEGORICAL_FEATURES
    assert "failure_code" in CATEGORICAL_FEATURES
    assert "customer_history_segment" in CATEGORICAL_FEATURES


def test_dataset_has_no_target_leakage():
    df = generate_population(n_cases=1500, seeds=[9], customer_count=200)
    target_leakage_columns = {"recovery_probability", "future_recovery_probability", "post_recovery_probability"}
    assert target_leakage_columns.isdisjoint(df.columns)
    assert "recovery_success" in df.columns


def test_shared_synthetic_world_contract_matches_benchmark():
    assert ML_PAYMENT_METHODS == BENCHMARK_PAYMENT_METHODS
    assert ML_NON_RECOVERABLE == BENCHMARK_NON_RECOVERABLE
    assert ML_TRANSIENT == BENCHMARK_TRANSIENT
    assert FAILURE_CODE_ORDER == sorted(FAILURE_CODE_ORDER, key=lambda code: FAILURE_CODE_ORDER.index(code))
    assert CUSTOMER_HISTORY_SEGMENTS == ["strong", "stable", "mixed", "risky"]
    assert MAX_RECOVERY_ATTEMPTS == 4
    assert TIME_SINCE_FAILURE_MINUTES_RANGE == (0, 2880)


def test_feature_generation_is_deterministic_and_ordered():
    case = {
        "customer_id": "feature_customer",
        "amount": 25000,
        "payment_method": "upi",
        "failure_code": "TIMEOUT",
        "attempt_number": 2,
        "time_since_failure": 15,
    }
    first = enrich_case_features(case)
    second = enrich_case_features(case)
    assert first == second
    assert list(first) == [
        "amount", "payment_method", "failure_code", "attempt_number",
        "time_since_failure", "customer_total_orders", "customer_successful_orders",
        "customer_failed_orders", "customer_success_rate", "customer_failure_rate",
        "customer_average_order_value", "customer_total_spend", "customer_last_success_age",
        "customer_last_failure_age", "customer_method_attempts", "customer_method_successes",
        "customer_method_failures", "customer_method_success_rate", "customer_history_segment",
        "order_value_percentile_for_customer",
    ]
    assert set(first).isdisjoint({"recovery_success", "future_recovery_probability"})
