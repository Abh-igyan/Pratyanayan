from __future__ import annotations

from types import SimpleNamespace

from backend.services.synthetic_journey import (
    JourneyWorld,
    PreRealizedAttempt,
    JourneyState,
    _available_methods,
    generate_world,
    run_agent_policy,
    run_baseline_policy,
)
from ml.synthetic_world import MAX_RECOVERY_ATTEMPTS, PAYMENT_METHODS


def make_world(*, natural=False, outcomes=None):
    outcomes = outcomes or {}
    pre_realized = {}
    for attempt in range(1, MAX_RECOVERY_ATTEMPTS + 1):
        pre_realized[attempt] = {}
        for method in PAYMENT_METHODS:
            success, failure = outcomes.get((attempt, method), (False, "TIMEOUT"))
            pre_realized[attempt][method] = PreRealizedAttempt(method, success, failure)
    return JourneyWorld(
        order_id="journey_test",
        amount=10000,
        customer_id="customer_test",
        initial_method="card",
        initial_failure_code="TIMEOUT",
        pre_realized=pre_realized,
        natural_late_success=natural,
    )


class FakeEngine:
    def __init__(self, scores=None):
        self.scores = scores or {}
        self.calls = []

    def decide(self, case):
        self.calls.append(dict(case))
        score = self.scores.get(case["payment_method"], 100.0)
        return SimpleNamespace(
            decision="RETRY" if score > 0 else "STOP",
            net_expected_value=score,
        )


def test_world_is_deterministic_and_has_bounded_attempt_slots():
    first = generate_world(42, 3)
    second = generate_world(42, 3)
    assert first == second
    assert set(first[0].pre_realized) == set(range(1, MAX_RECOVERY_ATTEMPTS + 1))


def test_agent_tracks_switch_and_actual_method_without_future_outcome_input():
    world = make_world(outcomes={(1, "upi"): (True, "")})
    engine = FakeEngine({"upi": 200.0, "card": 100.0})

    result = run_agent_policy(world, engine)

    assert result["recovery_attributed_orders"] == 1
    assert result["switches"] == 1
    assert result["state"].current_method == "upi"
    assert all("pre_realized" not in call for call in engine.calls)


def test_agent_replans_after_failures_and_does_not_repeat_exhausted_methods():
    outcomes = {
        (1, "card"): (False, "TIMEOUT"),
        (1, "upi"): (False, "TIMEOUT"),
        (2, "card"): (False, "TIMEOUT"),
        (2, "upi"): (False, "TIMEOUT"),
        (3, "wallet"): (False, "TIMEOUT"),
        (4, "netbanking"): (False, "TIMEOUT"),
    }
    world = make_world(outcomes=outcomes)
    engine = FakeEngine({"card": 100.0, "upi": 90.0, "wallet": 80.0, "netbanking": 70.0, "emi": 60.0})

    result = run_agent_policy(world, engine)
    state = result["state"]

    assert result["interventions"] <= MAX_RECOVERY_ATTEMPTS
    assert len(state.attempted_methods) >= 2
    assert max(state.failure_streak_by_method.values()) <= 2
    assert state.remaining_recovery_budget == MAX_RECOVERY_ATTEMPTS - result["interventions"]


def test_wait_is_not_an_immediate_payment_intervention():
    world = make_world()

    class WaitEngine(FakeEngine):
        def decide(self, case):
            self.calls.append(dict(case))
            return SimpleNamespace(decision="WAIT", net_expected_value=100.0)

    result = run_agent_policy(world, WaitEngine())
    assert result["interventions"] == 0
    assert result["state"].terminal_state == "WAITING"


def test_natural_success_is_separate_from_intervention_recovery():
    world = make_world(natural=True)
    result = run_agent_policy(world, FakeEngine({method: -1.0 for method in PAYMENT_METHODS}))
    assert result["recovery_attributed_orders"] == 0
    assert result["natural_success_count"] == 1
    assert result["intervention_attributed_recovered_revenue"] == 0.0
    assert result["natural_success_revenue"] == world.amount


def test_baseline_and_ai_receive_the_same_world_object():
    world = make_world(outcomes={(1, "card"): (True, "")})
    before = world.pre_realized[1]["card"]
    run_baseline_policy(world)
    run_agent_policy(world, FakeEngine({"card": 100.0}))
    assert world.pre_realized[1]["card"] == before


def test_all_methods_with_repeated_failures_have_no_viable_option():
    state = JourneyState(
        order_id="exhausted_methods",
        amount=10000,
        initial_method="card",
        current_method="card",
        attempted_methods=set(PAYMENT_METHODS),
        failure_streak_by_method={method: 2 for method in PAYMENT_METHODS},
    )
    assert _available_methods(state) == []