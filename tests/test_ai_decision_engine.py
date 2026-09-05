from __future__ import annotations
import math
from backend.services.synthetic_simulation import SyntheticCase
from ml.decision_engine import AIDecisionEngine, DecisionPolicy


def _make_case(amount, failure_code="PAYMENT_ERROR", attempt_number=1, time_since_failure=120, already_paid=False, initial_payment_success=False) -> SyntheticCase:
    return SyntheticCase(
        order_id="test_order",
        amount=amount,
        customer_id="cust_test",
        payment_method="card",
        initial_payment_success=initial_payment_success,
        failure_code=failure_code if not initial_payment_success else None,
        failure_description="desc",
        attempt_number=attempt_number,
        time_since_failure=time_since_failure,
        previous_success_rate=0.7,
        previous_failed_attempts=0,
        customer_history="stable",
        already_paid=already_paid,
        max_attempts=3,
        natural_recovery_probability=0.5,
    )

def test_hard_safety_guards_stop_recovery():
    engine = AIDecisionEngine()
    case = _make_case(50000, failure_code="CARD_DECLINED")
    outcome = engine.decide(case)
    assert outcome.decision == "STOP"

def test_already_paid_stop():
    engine = AIDecisionEngine()
    case = _make_case(12000, already_paid=True, initial_payment_success=True)
    outcome = engine.decide(case)
    assert outcome.decision == "STOP"
    assert "already paid" in outcome.reason

def test_max_attempt_stop():
    engine = AIDecisionEngine()
    case = _make_case(60000, attempt_number=3)
    outcome = engine.decide(case)
    assert outcome.decision == "STOP"
    assert "max attempts" in outcome.reason

def test_deterministic_reproducibility():
    engine_a = AIDecisionEngine()
    engine_b = AIDecisionEngine()
    case = _make_case(70000)
    a = engine_a.decide(case)
    b = engine_b.decide(case)
    assert a.decision == b.decision
    assert a.recovery_probability == b.recovery_probability
    assert a.expected_recovered_value == b.expected_recovered_value

def test_economic_high_value_moderate_prob():
    # Large amount, should easily exceed retry_min_net_value
    engine = AIDecisionEngine()
    case = _make_case(100000, time_since_failure=120)
    outcome = engine.decide(case)
    assert outcome.net_expected_value > engine.policy.retry_min_net_value
    assert outcome.decision == "RETRY"

def test_economic_low_value_stop():
    # Very small amount where cost of intervention exceeds expected return
    engine = AIDecisionEngine(policy=DecisionPolicy(intervention_cost=25.0, retry_min_net_value=10.0))
    case = _make_case(10, time_since_failure=120)
    outcome = engine.decide(case)
    assert outcome.net_expected_value < 0
    assert outcome.decision == "STOP"

def test_repeated_attempts_diminishing_returns():
    engine = AIDecisionEngine(policy=DecisionPolicy(diminishing_return_factor=0.5))
    case1 = _make_case(1000, attempt_number=1)
    outcome1 = engine.decide(case1)
    
    case2 = _make_case(1000, attempt_number=2)
    outcome2 = engine.decide(case2)
    
    # Net EV should drop due to diminishing_return_factor
    assert outcome2.net_expected_value < outcome1.net_expected_value

def test_wait_semantics_for_recent_transient():
    engine = AIDecisionEngine(policy=DecisionPolicy(intervention_cost=5.0, retry_min_net_value=5.0))
    # High value, transient, but very recent (<=30)
    case = _make_case(80000, failure_code="PAYMENT_ERROR", time_since_failure=20)
    outcome = engine.decide(case)
    # Should trigger the wait condition: "strongly positive EV, but recent transient"
    assert outcome.net_expected_value >= 5.0
    assert outcome.decision == "WAIT"
    assert "brief cooling window" in outcome.reason

def test_wait_semantics_for_marginal_ev():
    # Setup exactly to fall between wait_min and retry_min
    engine = AIDecisionEngine(policy=DecisionPolicy(intervention_cost=0.0, wait_min_net_value=10.0, retry_min_net_value=1000000.0))
    
    # We want adjusted_ev to be between 10 and 1000000
    case = _make_case(200, time_since_failure=120)
    outcome = engine.decide(case)
    if 10 <= outcome.net_expected_value < 1000000:
        assert outcome.decision == "WAIT"
        assert "marginal net expected value" in outcome.reason


# ---------------------------------------------------------------------------
# Part 2 additions: EV audit, diminishing returns proof, cost transparency
# ---------------------------------------------------------------------------

def test_ev_is_strictly_less_than_order_amount():
    """EV = probability * amount, never the full amount (probability < 1)."""
    engine = AIDecisionEngine()
    case = _make_case(50000, failure_code="PAYMENT_ERROR", attempt_number=1, time_since_failure=120)
    outcome = engine.decide(case)
    # Probability must be < 1.0, so EV must be < amount
    assert 0.0 < outcome.recovery_probability < 1.0, "probability should be in (0, 1)"
    assert outcome.expected_recovered_value < outcome.order_amount, (
        f"EV ({outcome.expected_recovered_value}) should be < amount ({outcome.order_amount})"
    )
    assert abs(outcome.expected_recovered_value - outcome.recovery_probability * outcome.order_amount) < 1.0


def test_diminishing_returns_ev_strictly_decreasing_across_all_attempts():
    """EV(n+1) < EV(n) for all sequential attempts when factor < 1.

    This holds under otherwise equal conditions (same case, only attempt_number varies).
    Initial payment failure is attempt_number=1 (no intervention yet), so
    the first *intervention* occurs at attempt_number=2 in this model.
    """
    factor = 0.75
    engine = AIDecisionEngine(policy=DecisionPolicy(
        diminishing_return_factor=factor,
        intervention_cost=0.0,  # Zero cost so we isolate the penalty effect
    ))
    # Use a large amount and safe failure code so safety guards don't fire
    prev_net_ev = None
    for attempt_num in range(1, 5):
        case = _make_case(200000, failure_code="PAYMENT_ERROR", attempt_number=attempt_num, time_since_failure=120)
        outcome = engine.decide(case)
        expected_penalty = factor ** max(0, attempt_num - 1)
        # Verify penalty applied correctly
        assert abs(
            outcome.expected_recovered_value * expected_penalty - outcome.net_expected_value
        ) < 1.0, f"Penalty not applied correctly at attempt {attempt_num}"
        if prev_net_ev is not None:
            assert outcome.net_expected_value < prev_net_ev, (
                f"EV did not decrease: attempt {attempt_num} EV={outcome.net_expected_value} "
                f">= previous EV={prev_net_ev}"
            )
        prev_net_ev = outcome.net_expected_value


def test_net_ev_equals_adjusted_ev_minus_intervention_cost():
    """net_expected_value = (probability * amount * penalty) - intervention_cost."""
    cost = 500.0
    factor = 0.80
    engine = AIDecisionEngine(policy=DecisionPolicy(
        intervention_cost=cost,
        diminishing_return_factor=factor,
        retry_min_net_value=-999999,  # always permit RETRY to see raw EV
        wait_min_net_value=-999999,
    ))
    case = _make_case(100000, failure_code="PAYMENT_ERROR", attempt_number=2, time_since_failure=200)
    outcome = engine.decide(case)
    penalty = factor ** max(0, outcome.attempt_number - 1)
    expected_net = outcome.recovery_probability * outcome.order_amount * penalty - cost
    assert abs(outcome.net_expected_value - expected_net) < 1.0, (
        f"net_ev={outcome.net_expected_value:.2f}, expected={expected_net:.2f}"
    )
    assert outcome.intervention_cost == cost


def test_initial_failure_not_counted_as_intervention():
    """attempt_number=1 means the initial failure; penalty at attempt 1 = 1.0 (no reduction)."""
    factor = 0.75
    engine = AIDecisionEngine(policy=DecisionPolicy(
        diminishing_return_factor=factor, intervention_cost=0.0,
        retry_min_net_value=-999999, wait_min_net_value=-999999,
    ))
    case = _make_case(100000, failure_code="PAYMENT_ERROR", attempt_number=1, time_since_failure=200)
    outcome = engine.decide(case)
    # Penalty = factor^(1-1) = 1.0, so net_ev == ev with no reduction
    assert abs(outcome.net_expected_value - outcome.expected_recovered_value) < 1.0, (
        "At attempt 1 the penalty must be 1.0 (initial failure is not an intervention)"
    )


def test_hard_safety_overrides_positive_ev():
    """Hard safety guards must fire even when net_expected_value would be positive."""
    engine = AIDecisionEngine()
    # Already paid: must be STOP regardless of amount
    case = _make_case(10000000, already_paid=True, initial_payment_success=True)
    outcome = engine.decide(case)
    assert outcome.decision == "STOP"
    assert "already paid" in outcome.reason

    # Non-recoverable failure: must be STOP
    case2 = _make_case(10000000, failure_code="INSUFFICIENT_FUNDS")
    outcome2 = engine.decide(case2)
    assert outcome2.decision == "STOP"
    assert "non-recoverable" in outcome2.reason
