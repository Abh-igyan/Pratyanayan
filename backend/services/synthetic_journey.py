from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from backend import config
from ml.decision_engine import AIDecisionEngine
from ml.shared_semantics import (
    calculate_latent_recovery_score,
    convert_score_to_probability,
    enrich_case_features,
)
from ml.synthetic_world import (
    MAX_RECOVERY_ATTEMPTS,
    BASELINE_MAX_RECOVERY_INTERVENTIONS,
    NON_RECOVERABLE_FAILURE_CODES,
    PAYMENT_METHODS,
    TRANSIENT_FAILURE_CODES,
)


@dataclass(frozen=True)
class PreRealizedAttempt:
    method: str
    success: bool
    failure_code: str


@dataclass
class JourneyState:
    order_id: str
    amount: float
    initial_method: str
    current_method: str
    attempted_methods: set[str] = field(default_factory=set)
    failure_streak_by_method: dict[str, int] = field(default_factory=dict)
    total_recovery_interventions: int = 0
    method_switches: int = 0
    remaining_recovery_budget: int = MAX_RECOVERY_ATTEMPTS
    last_failure_code: str | None = None
    time_since_failure: int = 5
    last_action: str | None = None
    pending_action: str | None = None
    terminal_state: str | None = None
    recovery_attributed: bool = False
    natural_success: bool = False
    actual_payment_method_when_known: str | None = None


@dataclass
class JourneyWorld:
    order_id: str
    amount: float
    customer_id: str
    initial_method: str
    initial_failure_code: str
    pre_realized: dict[int, dict[str, PreRealizedAttempt]]
    natural_late_success: bool

    def new_state(self) -> JourneyState:
        return JourneyState(
            order_id=self.order_id,
            amount=self.amount,
            initial_method=self.initial_method,
            current_method=self.initial_method,
            last_failure_code=self.initial_failure_code,
        )


def generate_world(seed: int, num_journeys: int) -> list[JourneyWorld]:
    rng = random.Random(seed)
    journeys: list[JourneyWorld] = []
    for index in range(num_journeys):
        order_id = f"sim_order_{seed}_{index}"
        customer_id = f"sim_cust_{rng.randint(1000, 9999)}"
        amount = rng.uniform(100, 20000)
        initial_method = rng.choice(PAYMENT_METHODS)
        failure_codes = NON_RECOVERABLE_FAILURE_CODES if rng.random() < 0.15 else TRANSIENT_FAILURE_CODES
        initial_failure_code = rng.choice(failure_codes)
        natural_late_success = rng.random() < 0.10

        pre_realized: dict[int, dict[str, PreRealizedAttempt]] = {}
        for attempt in range(1, MAX_RECOVERY_ATTEMPTS + 1):
            pre_realized[attempt] = {}
            for method in PAYMENT_METHODS:
                case_dict = {
                    "order_id": order_id,
                    "amount": amount,
                    "customer_id": customer_id,
                    "payment_method": method,
                    "failure_code": initial_failure_code,
                    "attempt_number": attempt,
                    "time_since_failure": 5,
                }
                score = calculate_latent_recovery_score(enrich_case_features(case_dict))
                if method == initial_method:
                    score -= 0.5 * attempt
                probability = convert_score_to_probability(score)
                method_rng = random.Random(
                    int(hashlib.md5(f"{order_id}_{attempt}_{method}".encode()).hexdigest()[:8], 16)
                )
                success = method_rng.random() < probability
                failure_code = initial_failure_code
                if not success:
                    failure_codes = TRANSIENT_FAILURE_CODES if method_rng.random() < 0.8 else NON_RECOVERABLE_FAILURE_CODES
                    failure_code = method_rng.choice(failure_codes)
                pre_realized[attempt][method] = PreRealizedAttempt(method, success, failure_code)

        journeys.append(JourneyWorld(
            order_id=order_id,
            amount=amount,
            customer_id=customer_id,
            initial_method=initial_method,
            initial_failure_code=initial_failure_code,
            pre_realized=pre_realized,
            natural_late_success=natural_late_success,
        ))
    return journeys


def _base_result(state: JourneyState, journey: JourneyWorld) -> dict[str, Any]:
    natural_revenue = journey.amount if state.natural_success else 0.0
    return {
        "recovered": state.recovery_attributed or state.natural_success,
        "revenue": (journey.amount if state.recovery_attributed else natural_revenue),
        "recovered_revenue": journey.amount if state.recovery_attributed else natural_revenue,
        "intervention_attributed_recovered_revenue": journey.amount if state.recovery_attributed else 0.0,
        "natural_success_revenue": natural_revenue,
        "recovery_attributed_orders": int(state.recovery_attributed),
        "natural_success_count": int(state.natural_success),
        "interventions": state.total_recovery_interventions,
        "method_switches": state.method_switches,
        "switches": state.method_switches,
        "stopped": state.terminal_state == "STOPPED",
        "exhausted": state.terminal_state == "EXHAUSTED",
        "state": state,
    }


def run_baseline_policy(journey: JourneyWorld) -> dict[str, Any]:
    state = journey.new_state()
    baseline_max = BASELINE_MAX_RECOVERY_INTERVENTIONS
    for attempt in range(1, baseline_max + 1):
        state.total_recovery_interventions += 1
        state.remaining_recovery_budget = baseline_max - state.total_recovery_interventions
        state.last_action = "RETRY"
        state.pending_action = "RETRY"
        state.attempted_methods.add(state.current_method)
        outcome = journey.pre_realized[attempt][state.current_method]
        state.actual_payment_method_when_known = outcome.method
        if outcome.success:
            state.recovery_attributed = True
            state.terminal_state = "RECOVERED"
            return _base_result(state, journey)
        state.last_failure_code = outcome.failure_code
        state.failure_streak_by_method[state.current_method] = state.failure_streak_by_method.get(state.current_method, 0) + 1
    state.terminal_state = "EXHAUSTED"
    if journey.natural_late_success:
        state.natural_success = True
        state.terminal_state = "NATURAL_SUCCESS"
    return _base_result(state, journey)


def _available_methods(state: JourneyState) -> list[str]:
    return [
        method for method in PAYMENT_METHODS
        if state.failure_streak_by_method.get(method, 0) < 2
        or method not in state.attempted_methods
    ]


def run_agent_policy(journey: JourneyWorld, engine: AIDecisionEngine) -> dict[str, Any]:
    state = journey.new_state()
    max_interventions = min(MAX_RECOVERY_ATTEMPTS, int(config.MAX_RECOVERY_ATTEMPTS))
    state.remaining_recovery_budget = max_interventions

    for _cycle in range(1, (max_interventions * 2) + 1):
        if state.total_recovery_interventions >= max_interventions:
            state.terminal_state = "EXHAUSTED"
            break
        methods = _available_methods(state)
        if not methods:
            state.terminal_state = "STOPPED"
            break

        evaluations: list[tuple[str, Any]] = []
        for method in methods:
            case_dict = {
                "amount": journey.amount,
                "customer_id": journey.customer_id,
                "payment_method": method,
                "failure_code": state.last_failure_code,
                "attempt_number": state.total_recovery_interventions + 1,
                "time_since_failure": state.time_since_failure,
                "max_attempts": max_interventions,
                "already_paid": False,
            }
            evaluations.append((method, engine.decide(case_dict)))
        best_method, best_outcome = max(evaluations, key=lambda item: item[1].net_expected_value)

        if best_outcome.decision == "STOP":
            state.last_action = "STOP"
            state.terminal_state = "STOPPED"
            break
        if best_outcome.decision == "WAIT":
            state.last_action = "WAIT"
            state.pending_action = "WAIT"
            state.time_since_failure = max(state.time_since_failure, 60)
            state.terminal_state = "WAITING"
            continue

        state.total_recovery_interventions += 1
        state.remaining_recovery_budget = max_interventions - state.total_recovery_interventions
        state.last_action = "SWITCH_PAYMENT_METHOD" if best_method != state.current_method else "RETRY"
        state.pending_action = state.last_action
        if best_method != state.current_method:
            state.method_switches += 1
        state.current_method = best_method
        state.attempted_methods.add(best_method)
        outcome = journey.pre_realized[state.total_recovery_interventions][best_method]
        state.actual_payment_method_when_known = outcome.method
        if outcome.success:
            state.recovery_attributed = True
            state.terminal_state = "RECOVERED"
            return _base_result(state, journey)
        state.last_failure_code = outcome.failure_code
        state.failure_streak_by_method[best_method] = state.failure_streak_by_method.get(best_method, 0) + 1
        if outcome.failure_code in NON_RECOVERABLE_FAILURE_CODES:
            state.terminal_state = "STOPPED"
            break

    if state.terminal_state is None:
        state.terminal_state = "EXHAUSTED"
    if state.terminal_state in {"STOPPED", "EXHAUSTED", "WAITING"} and journey.natural_late_success:
        state.natural_success = True
        state.terminal_state = "NATURAL_SUCCESS"
    return _base_result(state, journey)


def _new_metrics() -> dict[str, Any]:
    return {
        "recovered_revenue": 0.0,
        "intervention_attributed_recovered_revenue": 0.0,
        "natural_success_revenue": 0.0,
        "interventions": 0,
        "switches": 0,
        "recovered": 0,
        "recovery_attributed_orders": 0,
        "natural_success_count": 0,
        "exhausted": 0,
        "stopped": 0,
        "total_revenue_at_risk": 0.0,
    }


def _add_result(metrics: dict[str, Any], result: dict[str, Any], amount: float) -> None:
    metrics["total_revenue_at_risk"] += amount
    for key in (
        "recovered_revenue", "intervention_attributed_recovered_revenue",
        "natural_success_revenue", "interventions", "switches", "recovered",
        "recovery_attributed_orders", "natural_success_count", "exhausted", "stopped",
    ):
        metrics[key] += result[key]


def run_batch_simulation(seed: int, num_journeys: int) -> dict[str, Any]:
    journeys = generate_world(seed, num_journeys)
    engine = AIDecisionEngine()
    baseline = _new_metrics()
    ai = _new_metrics()
    for journey in journeys:
        _add_result(baseline, run_baseline_policy(journey), journey.amount)
        _add_result(ai, run_agent_policy(journey, engine), journey.amount)
    for metrics in (baseline, ai):
        metrics["recovery_rate"] = metrics["recovery_attributed_orders"] / max(1, num_journeys)
        metrics["revenue_recovery_rate"] = metrics["intervention_attributed_recovered_revenue"] / max(1.0, metrics["total_revenue_at_risk"])
    return {"seed": seed, "journeys": num_journeys, "baseline": baseline, "ai": ai}
