from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

from backend import config
from ml.shared_semantics import (
    calculate_latent_recovery_score,
    convert_score_to_probability,
    enrich_case_features,
    get_customer_profile,
)
from ml.synthetic_world import (
    NON_RECOVERABLE_FAILURE_CODES,
    PAYMENT_METHODS,
    TRANSIENT_FAILURE_CODES,
)


FAILURE_DESCRIPTIONS = {
    "PAYMENT_ERROR": "Temporary payment gateway issue. Retry may succeed.",
    "NETWORK_ERROR": "Network timeout during payment authorization.",
    "TIMEOUT": "Payment attempt timed out before completion.",
    "BAD_REQUEST_ERROR": "Payment request was rejected by the provider.",
    "TEMPORARY_ISSUE": "A temporary processing issue prevented payment completion.",
    "PROCESSING_ERROR": "Payment processing failed temporarily.",
    "CARD_DECLINED": "Card was declined by the issuer.",
    "INSUFFICIENT_FUNDS": "Funds unavailable in the source account.",
    "AUTHENTICATION_ERROR": "Authentication step failed for the payment instrument.",
    "BANK_DECLINED": "Bank declined the payment request.",
    "INVALID_ACCOUNT": "The account or instrument could not be used.",
    "CURRENCY_NOT_ALLOWED": "The payment method does not support this currency.",
}


@dataclass
class SyntheticCase:
    order_id: str
    amount: int
    customer_id: str
    payment_method: str

    # Initial payment state
    initial_payment_success: bool

    # Failure information is populated only for failed payments.
    failure_code: str | None
    failure_description: str | None

    # Context available to the decision-maker.
    attempt_number: int
    time_since_failure: int
    previous_success_rate: float
    previous_failed_attempts: int
    customer_history: str

    # Current state / policy configuration.
    already_paid: bool
    max_attempts: int

    # Ground truth for the synthetic world.
    # This is deliberately generated independently of the policy.
    natural_recovery_probability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _failure_code_and_description(
    rng: random.Random,
) -> tuple[str, str]:
    # Include both recoverable/transient and non-recoverable failures.
    codes = [
        "PAYMENT_ERROR",
        "NETWORK_ERROR",
        "TIMEOUT",
        "BAD_REQUEST_ERROR",
        "TEMPORARY_ISSUE",
        "PROCESSING_ERROR",
        "CARD_DECLINED",
        "INSUFFICIENT_FUNDS",
        "AUTHENTICATION_ERROR",
        "BANK_DECLINED",
        "INVALID_ACCOUNT",
        "CURRENCY_NOT_ALLOWED",
    ]

    code = rng.choice(codes)
    return code, FAILURE_DESCRIPTIONS[code]


def generate_synthetic_cases(
    seed: int = 42,
    count: int = 1000,
) -> list[SyntheticCase]:
    rng = random.Random(seed)

    cases: list[SyntheticCase] = []

    for idx in range(count):
        customer_id = f"cust_{idx % 250:03d}"
        
        # Pull deterministic background features
        profile = get_customer_profile(customer_id)
        
        amount = rng.randint(5000, 150000)
        method = rng.choice(PAYMENT_METHODS)

        previous_success_rate = profile["customer_success_rate"]
        previous_failed_attempts = profile["customer_failed_orders"]
        customer_history = profile["customer_history_segment"]

        # Most orders succeed initially; a meaningful minority fail.
        initial_payment_success = rng.random() < 0.65

        if initial_payment_success:
            failure_code = None
            failure_description = None
            attempt_number = 1
            time_since_failure = 0
            already_paid = True
            natural_recovery_probability = 0.0

        else:
            failure_code, failure_description = (
                _failure_code_and_description(rng)
            )

            max_attempts = int(config.MAX_RECOVERY_ATTEMPTS)
            attempt_number = rng.randint(1, max(1, max_attempts))
            time_since_failure = rng.randint(0, 1440)
            already_paid = False

            # We build the base case and enrich it to calculate the exact canonical score
            base_case = {
                "customer_id": customer_id,
                "amount": amount,
                "payment_method": method,
                "failure_code": failure_code,
                "attempt_number": attempt_number,
                "time_since_failure": time_since_failure,
            }
            features = enrich_case_features(base_case)
            score = calculate_latent_recovery_score(features)
            
            # The benchmark environment has NO random noise added to the score.
            # It maps the latent score purely to a probability.
            natural_recovery_probability = convert_score_to_probability(score, noise=0.0)

        order_id = (
            f"synthetic_order_{idx:04d}_"
            f"{rng.randint(10000, 99999)}"
        )

        cases.append(
            SyntheticCase(
                order_id=order_id,
                amount=amount,
                customer_id=customer_id,
                payment_method=method,
                initial_payment_success=initial_payment_success,
                failure_code=failure_code,
                failure_description=failure_description,
                attempt_number=attempt_number,
                time_since_failure=time_since_failure,
                previous_success_rate=previous_success_rate,
                previous_failed_attempts=previous_failed_attempts,
                customer_history=customer_history,
                already_paid=already_paid,
                max_attempts=int(config.MAX_RECOVERY_ATTEMPTS),
                natural_recovery_probability=(
                    natural_recovery_probability
                ),
            )
        )

    return cases


# ---------------------------------------------------------------------------
# Deterministic baseline policy
# ---------------------------------------------------------------------------

def _decision_priority(case: SyntheticCase) -> str:
    if case.already_paid:
        return "STOP"

    if case.attempt_number >= case.max_attempts:
        return "STOP"

    if case.failure_code in NON_RECOVERABLE_FAILURE_CODES:
        return "STOP"

    if (
        case.previous_failed_attempts >= 3
        and case.previous_success_rate <= 0.35
    ):
        return "STOP"

    if (
        case.time_since_failure <= 30
        and case.failure_code in TRANSIENT_FAILURE_CODES
    ):
        return "WAIT"

    if (
        case.customer_history == "strong"
        and case.previous_success_rate >= 0.75
    ):
        return "RETRY"

    if case.failure_code in TRANSIENT_FAILURE_CODES:
        return "RETRY"

    if (
        case.previous_success_rate >= 0.60
        and case.previous_failed_attempts <= 1
    ):
        return "RETRY"

    if case.time_since_failure <= 120:
        return "WAIT"

    return "RETRY"


def evaluate_synthetic_policy(
    case: SyntheticCase,
) -> tuple[str, str]:
    decision = _decision_priority(case)

    if decision == "STOP":
        if case.already_paid:
            return "STOP", "order is already paid"

        if case.attempt_number >= case.max_attempts:
            return "STOP", "maximum attempt limit reached"

        if case.failure_code in NON_RECOVERABLE_FAILURE_CODES:
            return "STOP", "failure is classified as non-recoverable"

        if (
            case.previous_failed_attempts >= 3
            and case.previous_success_rate <= 0.35
        ):
            return (
                "STOP",
                "customer has repeated failures and low success rate",
            )

        return "STOP", "case is ineligible for recovery"

    if decision == "WAIT":
        return (
            "WAIT",
            "failure is recent; use a short cooling window",
        )

    if (
        case.customer_history == "strong"
        and case.previous_success_rate >= 0.75
    ):
        return (
            "RETRY",
            "strong customer history and good success rate",
        )

    if case.failure_code in TRANSIENT_FAILURE_CODES:
        return (
            "RETRY",
            "failure appears transient and potentially recoverable",
        )

    if (
        case.previous_success_rate >= 0.60
        and case.previous_failed_attempts <= 1
    ):
        return (
            "RETRY",
            "customer has reliable payment history",
        )

    return "RETRY", "default deterministic retry policy"


# ---------------------------------------------------------------------------
# Synthetic environment / ground truth
# ---------------------------------------------------------------------------

def _simulate_ground_truth_outcome(
    rng: random.Random,
    case: SyntheticCase,
) -> bool:
    if case.initial_payment_success:
        return True

    return rng.random() < case.natural_recovery_probability


def _simulate_policy_outcome(
    rng: random.Random,
    case: SyntheticCase,
    decision: str,
) -> tuple[str, int, bool, bool]:
    if case.initial_payment_success:
        return ("already_paid", 0, False, False)

    if decision == "STOP":
        return ("not_recovered", 0, False, False)

    recovered = _simulate_ground_truth_outcome(rng, case)

    if recovered:
        return ("recovered", case.amount, True, False)

    unnecessary = (
        decision in {"RETRY", "WAIT"}
        and case.failure_code in NON_RECOVERABLE_FAILURE_CODES
    )

    return ("not_recovered", 0, False, unnecessary)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_synthetic_simulation(
    seed: int = 42,
    count: int = 1000,
) -> dict[str, Any]:

    cases = generate_synthetic_cases(
        seed=seed,
        count=count,
    )

    outcome_rng = random.Random(seed + 1)

    rows: list[dict[str, Any]] = []

    retry_count = 0
    wait_count = 0
    stop_count = 0

    recovery_attempts = 0
    recovered_orders = 0
    recovered_revenue = 0

    initial_failures = 0
    total_at_risk_revenue = 0

    unnecessary_interventions = 0

    for case in cases:
        if case.initial_payment_success:
            rows.append(
                {
                    **case.to_dict(),
                    "decision": "NONE",
                    "reason": "initial payment succeeded",
                    "recovery_attempt": 0,
                    "final_outcome": "initial_success",
                    "recovered_amount": 0,
                    "successful": True,
                    "unnecessarily_intervened": False,
                }
            )
            continue

        initial_failures += 1
        total_at_risk_revenue += case.amount

        decision, reason = evaluate_synthetic_policy(case)

        if decision == "RETRY":
            retry_count += 1
            recovery_attempts += 1
        elif decision == "WAIT":
            wait_count += 1
            recovery_attempts += 1
        else:
            stop_count += 1

        (
            final_outcome,
            recovered_amount,
            successful,
            unnecessarily_intervened,
        ) = _simulate_policy_outcome(
            outcome_rng,
            case,
            decision,
        )

        if unnecessarily_intervened:
            unnecessary_interventions += 1

        if recovered_amount > 0:
            recovered_orders += 1
            recovered_revenue += recovered_amount

        rows.append(
            {
                **case.to_dict(),
                "decision": decision,
                "reason": reason,
                "recovery_attempt": (
                    1 if decision in {"RETRY", "WAIT"} else 0
                ),
                "final_outcome": final_outcome,
                "recovered_amount": recovered_amount,
                "successful": successful,
                "unnecessarily_intervened": (
                    unnecessarily_intervened
                ),
            }
        )

    recovery_candidates = initial_failures
    recovery_rate = (recovered_orders / recovery_candidates if recovery_candidates else 0.0)
    revenue_recovery_rate = (recovered_revenue / total_at_risk_revenue if total_at_risk_revenue else 0.0)

    return {
        "seed": seed,
        "total_orders": count,
        "initial_successes": count - initial_failures,
        "initial_failures": initial_failures,
        "recovery_candidates": recovery_candidates,
        "retry": retry_count,
        "wait": wait_count,
        "stop": stop_count,
        "recovery_attempts": recovery_attempts,
        "recovered_orders": recovered_orders,
        "recovered_revenue": recovered_revenue,
        "total_at_risk_revenue": total_at_risk_revenue,
        "recovery_rate": round(recovery_rate, 4),
        "revenue_recovery_rate": round(revenue_recovery_rate, 4),
        "exhausted_cases": sum(
            1
            for case in cases
            if (
                not case.initial_payment_success
                and case.attempt_number >= case.max_attempts
            )
        ),
        "unnecessary_interventions": unnecessary_interventions,
        "cases": rows[:5],
    }