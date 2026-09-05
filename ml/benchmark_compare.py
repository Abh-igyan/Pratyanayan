from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

from backend.services.synthetic_simulation import (
    SyntheticCase,
    _simulate_ground_truth_outcome,
    _simulate_policy_outcome,
    evaluate_synthetic_policy,
    generate_synthetic_cases,
)
from ml.decision_engine import AIDecisionEngine


def _run_policy_suite(
    cases: list[SyntheticCase],
    policy_fn,
    realized_outcomes: dict[str, bool],
) -> dict[str, Any]:
    initial_failures = 0
    total_at_risk_revenue = 0
    recovery_attempts = 0
    recovered_orders = 0
    recovered_revenue = 0
    unnecessary_interventions = 0
    exhausted_cases = 0
    retry_count = 0
    wait_count = 0
    stop_count = 0
    rows: list[dict[str, Any]] = []

    for case in cases:
        if case.initial_payment_success:
            continue
        initial_failures += 1
        total_at_risk_revenue += case.amount

        decision, reason = policy_fn(case)

        if decision == "RETRY":
            retry_count += 1
            recovery_attempts += 1
        elif decision == "WAIT":
            wait_count += 1
            recovery_attempts += 1
        else:
            stop_count += 1
            if case.attempt_number >= case.max_attempts:
                exhausted_cases += 1

        if decision == "STOP":
            final_outcome = "not_recovered"
            recovered_amount = 0
            successful = False
            unnecessarily = False
        else:
            # Both policies must observe the same realized world outcome
            # for each case; policy actions must not change RNG alignment.
            recovered = realized_outcomes[case.order_id]
            if recovered:
                final_outcome = "recovered"
                recovered_amount = case.amount
                successful = True
                unnecessarily = False
            else:
                final_outcome = "not_recovered"
                recovered_amount = 0
                successful = False
                unnecessarily = (
                    decision in {"RETRY", "WAIT"}
                    and case.failure_code in {"CARD_DECLINED", "INSUFFICIENT_FUNDS", "AUTHENTICATION_ERROR", "BANK_DECLINED", "INVALID_ACCOUNT", "CURRENCY_NOT_ALLOWED"}
                )

        if unnecessarily:
            unnecessary_interventions += 1
        if recovered_amount > 0:
            recovered_orders += 1
            recovered_revenue += recovered_amount

        rows.append(
            {
                **asdict(case),
                "decision": decision,
                "reason": reason,
                "final_outcome": final_outcome,
                "recovered_amount": recovered_amount,
                "successful": successful,
                "unnecessarily_intervened": unnecessarily,
            }
        )

    recovery_rate = (recovered_orders / initial_failures) if initial_failures else 0.0
    revenue_recovery_rate = (recovered_revenue / total_at_risk_revenue) if total_at_risk_revenue else 0.0
    intervention_rate = (recovery_attempts / initial_failures) if initial_failures else 0.0
    return {
        "initial_failed_orders": initial_failures,
        "total_at_risk_revenue": total_at_risk_revenue,
        "retry": retry_count,
        "wait": wait_count,
        "stop": stop_count,
        "recovery_attempts": recovery_attempts,
        "recovered_orders": recovered_orders,
        "recovered_revenue": recovered_revenue,
        "recovery_rate": recovery_rate,
        "revenue_recovery_rate": revenue_recovery_rate,
        "intervention_rate": intervention_rate,
        "unnecessary_interventions": unnecessary_interventions,
        "exhausted_cases": exhausted_cases,
        "rows": rows[:3],
    }


def compare_policies(seed: int = 42, count: int = 1000) -> dict[str, Any]:
    cases = generate_synthetic_cases(seed=seed, count=count)
    outcome_rng = random.Random(seed + 1)
    realized_outcomes = {
        case.order_id: _simulate_ground_truth_outcome(outcome_rng, case)
        for case in cases
        if not case.initial_payment_success
    }
    baseline = _run_policy_suite(
        cases,
        evaluate_synthetic_policy,
        realized_outcomes,
    )

    engine = AIDecisionEngine()

    def ai_policy(case: SyntheticCase):
        outcome = engine.decide(case)
        return outcome.decision, outcome.reason

    ai = _run_policy_suite(cases, ai_policy, realized_outcomes)

    return {
        "baseline": baseline,
        "ai": ai,
        "ai_minus_baseline_recovered_revenue": ai["recovered_revenue"] - baseline["recovered_revenue"],
        "ai_minus_baseline_recovery_rate": ai["recovery_rate"] - baseline["recovery_rate"],
        "additional_interventions_required": max(0, ai["recovery_attempts"] - baseline["recovery_attempts"]),
        "recovered_revenue_per_intervention": (
            ai["recovered_revenue"] / ai["recovery_attempts"] if ai["recovery_attempts"] else 0.0
        ),
    }
