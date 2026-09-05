from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from ml.features import FEATURE_COLUMNS, TARGET_COLUMN
from ml.shared_semantics import (
    calculate_latent_recovery_score,
    convert_score_to_probability,
    enrich_case_features,
    get_customer_profile,
)
from ml.synthetic_world import (
    FAILURE_CODE_ORDER,
    MAX_RECOVERY_ATTEMPTS,
    PAYMENT_METHODS,
    TIME_SINCE_FAILURE_MINUTES_MIN,
    TIME_SINCE_FAILURE_MINUTES_MAX,
)


def _generate_single_case(
    customer_id: str,
    rng: np.random.Generator,
    amount_scale: float,
) -> dict[str, object]:
    profile = get_customer_profile(customer_id)
    
    payment_method = str(
        rng.choice(PAYMENT_METHODS, p=np.asarray(profile["_method_weights"]))
    )
    
    segment = profile["customer_history_segment"]
    if segment == "strong":
        failure_code = rng.choice(["PAYMENT_ERROR", "NETWORK_ERROR", "TIMEOUT", "BAD_REQUEST_ERROR", "TEMPORARY_ISSUE", "CARD_DECLINED"], p=[0.24, 0.18, 0.14, 0.12, 0.16, 0.16])
    elif segment == "risky":
        failure_code = rng.choice(["INSUFFICIENT_FUNDS", "BANK_DECLINED", "CARD_DECLINED", "AUTHENTICATION_ERROR", "PAYMENT_ERROR", "TEMPORARY_ISSUE"], p=[0.22, 0.18, 0.17, 0.15, 0.16, 0.12])
    else:
        failure_code = rng.choice(
            FAILURE_CODE_ORDER,
            p=[0.16, 0.14, 0.10, 0.12, 0.10, 0.09, 0.08, 0.07, 0.07, 0.05, 0.02],
        )

    amount = int(max(2000, float(profile["customer_average_order_value"]) * rng.uniform(0.55, 1.6) * amount_scale))
    attempt_number = int(rng.integers(1, MAX_RECOVERY_ATTEMPTS + 1))
    time_since_failure = int(rng.integers(TIME_SINCE_FAILURE_MINUTES_MIN, TIME_SINCE_FAILURE_MINUTES_MAX + 1))

    base_case = {
        "customer_id": customer_id,
        "amount": amount,
        "payment_method": payment_method,
        "failure_code": failure_code,
        "attempt_number": attempt_number,
        "time_since_failure": time_since_failure,
    }
    
    features = enrich_case_features(base_case)
    
    # Calculate canonical probability with noise for training
    score = calculate_latent_recovery_score(features)
    noise = float(rng.normal(0.0, 0.55))
    probability = convert_score_to_probability(score, noise)
    
    target = int(rng.binomial(1, probability))
    features["recovery_success"] = target
    features["customer_id"] = customer_id
    
    return features


def generate_population(
    n_cases: int = 50000,
    seeds: Sequence[int] | None = None,
    customer_count: int = 3000,
) -> pd.DataFrame:
    if seeds is None:
        seeds = [7, 11, 13, 17, 19, 23]

    all_records: list[dict[str, object]] = []

    for seed in seeds:
        local_rng = np.random.default_rng(seed)
        amount_scale = 1.0 + (seed % 7) * 0.12
        for _ in range(max(1, n_cases // len(seeds))):
            cust_idx = int(local_rng.integers(0, customer_count))
            customer_id = f"cust_{cust_idx:06d}"
            all_records.append(_generate_single_case(customer_id, local_rng, amount_scale))

    if len(all_records) < n_cases:
        extra_seed = max(101, 42 + len(seeds))
        extra_rng = np.random.default_rng(extra_seed)
        while len(all_records) < n_cases:
            cust_idx = int(extra_rng.integers(0, customer_count))
            customer_id = f"cust_{cust_idx:06d}"
            all_records.append(_generate_single_case(customer_id, extra_rng, 1.0))

    df = pd.DataFrame(all_records)
    if "customer_id" not in df.columns:
        df["customer_id"] = ""
    if len(df) > n_cases:
        df = df.sample(n_cases, random_state=42).reset_index(drop=True)
    return df[["customer_id"] + FEATURE_COLUMNS + [TARGET_COLUMN]]
