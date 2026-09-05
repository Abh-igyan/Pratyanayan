import math
import random
from ml.shared_semantics import (
    calculate_latent_recovery_score,
    convert_score_to_probability,
    enrich_case_features,
    get_customer_profile,
)

def test_get_customer_profile_is_deterministic():
    p1 = get_customer_profile("cust_001")
    p2 = get_customer_profile("cust_001")
    p3 = get_customer_profile("cust_002")
    
    assert p1 == p2
    assert p1 != p3

def test_enrich_case_features():
    base = {
        "customer_id": "cust_999",
        "amount": 50000,
        "payment_method": "card",
        "failure_code": "NETWORK_ERROR",
        "attempt_number": 1,
        "time_since_failure": 30,
    }
    features = enrich_case_features(base)
    
    assert features["amount"] == 50000
    assert features["payment_method"] == "card"
    assert features["failure_code"] == "NETWORK_ERROR"
    assert "customer_success_rate" in features
    assert "customer_method_success_rate" in features

def test_calculate_latent_recovery_score():
    features = {
        "amount": 50000,
        "customer_success_rate": 0.8,
        "customer_failure_rate": 0.2,
        "customer_history_segment": "strong",
        "payment_method": "card",
        "failure_code": "NETWORK_ERROR",
        "customer_failed_orders": 1,
        "attempt_number": 1,
        "time_since_failure": 30,
        "customer_method_success_rate": 0.9,
        "order_value_percentile_for_customer": 0.7,
    }
    
    score = calculate_latent_recovery_score(features)
    assert isinstance(score, float)
    
    # Check probabilities
    prob = convert_score_to_probability(score)
    assert 0.0 <= prob <= 1.0

def test_fairness_preservation():
    # Ensure no random mutations occur when evaluating score without noise
    features = enrich_case_features({"customer_id": "cust_123", "amount": 10000})
    s1 = calculate_latent_recovery_score(features)
    s2 = calculate_latent_recovery_score(features)
    assert s1 == s2
