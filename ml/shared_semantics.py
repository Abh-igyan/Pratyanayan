import math
import random
import hashlib
from typing import Any

from ml.synthetic_world import (
    CUSTOMER_HISTORY_SEGMENTS,
    FAILURE_CODE_ORDER,
    MAX_RECOVERY_ATTEMPTS,
    NON_RECOVERABLE_FAILURE_CODES,
    PAYMENT_METHODS,
    TIME_SINCE_FAILURE_MINUTES_MIN,
    TIME_SINCE_FAILURE_MINUTES_MAX,
    TRANSIENT_FAILURE_CODES,
)


def get_customer_profile(customer_id: str) -> dict[str, Any]:
    seed_val = int(hashlib.md5(customer_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed_val)
    
    segment = rng.choices(
        CUSTOMER_HISTORY_SEGMENTS, 
        weights=[0.35, 0.30, 0.20, 0.15], 
        k=1
    )[0]
    
    if segment == "strong":
        base_success = rng.uniform(0.68, 0.94)
    elif segment == "stable":
        base_success = rng.uniform(0.55, 0.82)
    elif segment == "mixed":
        base_success = rng.uniform(0.40, 0.70)
    else:
        base_success = rng.uniform(0.18, 0.52)
        
    total_orders = rng.randint(3, 90)
    successful_orders = int(total_orders * base_success)
    failed_orders = max(1, total_orders - successful_orders)
    success_rate = successful_orders / total_orders
    failure_rate = failed_orders / total_orders
    average_order_value = rng.uniform(800, 18000)
    total_spend = average_order_value * total_orders
    
    method_weights = [0.32, 0.26, 0.17, 0.15, 0.10]
    if segment == "risky":
        method_weights = [0.24, 0.21, 0.18, 0.19, 0.18]
    
    return {
        "customer_id": customer_id,
        "customer_history_segment": segment,
        "customer_total_orders": total_orders,
        "customer_successful_orders": successful_orders,
        "customer_failed_orders": failed_orders,
        "customer_success_rate": round(max(0.05, min(0.98, success_rate)), 4),
        "customer_failure_rate": round(max(0.02, min(0.95, failure_rate)), 4),
        "customer_average_order_value": round(average_order_value, 2),
        "customer_total_spend": round(total_spend, 2),
        "customer_last_success_age": rng.randint(0, 365),
        "customer_last_failure_age": rng.randint(0, 365),
        "_method_weights": method_weights,
    }


def enrich_case_features(case_dict: dict[str, Any]) -> dict[str, Any]:
    customer_id = case_dict.get("customer_id", "unknown")
    profile = get_customer_profile(customer_id)

    # Live decisions may provide database-backed history. Synthetic training
    # rows continue to use the canonical deterministic profile above.
    if "customer_total_orders" in case_dict:
        profile.update({
            key: case_dict[key]
            for key in (
                "customer_total_orders",
                "customer_successful_orders",
                "customer_failed_orders",
                "customer_success_rate",
                "customer_failure_rate",
                "customer_average_order_value",
                "customer_total_spend",
                "customer_last_success_age",
                "customer_last_failure_age",
                "customer_history_segment",
            )
            if key in case_dict
        })
    
    amount = case_dict.get("amount", 0)
    payment_method = case_dict.get("payment_method", "card")
    failure_code = case_dict.get("failure_code") or "PAYMENT_ERROR"
    attempt_number = case_dict.get("attempt_number", 1)
    time_since_failure = case_dict.get("time_since_failure", 0)
    
    method_seed = int(hashlib.md5(f"{customer_id}_{payment_method}".encode()).hexdigest()[:8], 16)
    mrng = random.Random(method_seed)
    
    customer_method_attempts = int(case_dict.get("customer_method_attempts", mrng.randint(1, 30)))
    customer_method_successes = int(case_dict.get(
        "customer_method_successes",
        max(0, min(customer_method_attempts, customer_method_attempts * profile["customer_success_rate"])),
    ))
    customer_method_failures = int(case_dict.get(
        "customer_method_failures",
        customer_method_attempts - customer_method_successes,
    ))
    customer_method_success_rate = float(case_dict.get(
        "customer_method_success_rate",
        customer_method_successes / max(1, customer_method_attempts),
    ))
    
    order_value_percentile = min(0.99, max(0.01, amount / 150000.0))
    
    return {
        "amount": amount,
        "payment_method": payment_method,
        "failure_code": failure_code,
        "attempt_number": attempt_number,
        "time_since_failure": time_since_failure,
        "customer_total_orders": profile["customer_total_orders"],
        "customer_successful_orders": profile["customer_successful_orders"],
        "customer_failed_orders": profile["customer_failed_orders"],
        "customer_success_rate": profile["customer_success_rate"],
        "customer_failure_rate": profile["customer_failure_rate"],
        "customer_average_order_value": profile["customer_average_order_value"],
        "customer_total_spend": profile["customer_total_spend"],
        "customer_last_success_age": profile["customer_last_success_age"],
        "customer_last_failure_age": profile["customer_last_failure_age"],
        "customer_method_attempts": customer_method_attempts,
        "customer_method_successes": customer_method_successes,
        "customer_method_failures": customer_method_failures,
        "customer_method_success_rate": round(customer_method_success_rate, 4),
        "customer_history_segment": profile["customer_history_segment"],
        "order_value_percentile_for_customer": round(order_value_percentile, 4),
    }


def calculate_latent_recovery_score(features: dict[str, Any]) -> float:
    amount = float(features.get("amount", 0))
    success_rate = float(features.get("customer_success_rate", 0.5))
    failure_rate = float(features.get("customer_failure_rate", 0.5))
    history_segment = str(features.get("customer_history_segment", "stable"))
    payment_method = str(features.get("payment_method", "card"))
    failure_code = str(features.get("failure_code", "PAYMENT_ERROR"))
    failed_orders = int(features.get("customer_failed_orders", 0))
    attempt_number = int(features.get("attempt_number", 1))
    time_since_failure = int(features.get("time_since_failure", 0))
    method_success_rate = float(features.get("customer_method_success_rate", 0.5))
    order_value_percentile = float(features.get("order_value_percentile_for_customer", 0.5))

    base = -2.4
    if amount > 0:
        base += 0.8 * math.log1p(amount / 1000.0)
    base += 0.9 * (success_rate - 0.5)
    base += 0.6 * (0.5 - failure_rate)
    
    if history_segment == "strong":
        base += 0.35
    elif history_segment == "risky":
        base -= 0.25
        
    if payment_method in {"upi", "card"}:
        base += 0.45
        
    if failure_code in TRANSIENT_FAILURE_CODES:
        base += 0.40
    elif failure_code in NON_RECOVERABLE_FAILURE_CODES:
        base -= 0.50
        
    base -= 0.30 * (min(failed_orders, 8) / 8.0)
    
    if attempt_number > 2:
        base -= 0.20
        
    if time_since_failure <= 60 and failure_code in TRANSIENT_FAILURE_CODES:
        base += 0.18
        
    base += 0.15 * method_success_rate
    base += 0.10 * order_value_percentile
    
    return base


def convert_score_to_probability(score: float, noise: float = 0.0) -> float:
    noisy_score = score + noise
    if noisy_score > 20: return 0.9999
    if noisy_score < -20: return 0.0001
    return 1.0 / (1.0 + math.exp(-noisy_score))
