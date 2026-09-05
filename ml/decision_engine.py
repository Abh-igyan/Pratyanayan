from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from catboost import CatBoostClassifier

from backend.services.synthetic_simulation import (
    NON_RECOVERABLE_FAILURE_CODES,
    TRANSIENT_FAILURE_CODES,
    SyntheticCase,
)
from ml.features import FEATURE_COLUMNS
from ml.synthetic_world import MAX_RECOVERY_ATTEMPTS, TIME_SINCE_FAILURE_MINUTES_MAX

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "model" / "catboost_recovery_model.cbm"


@dataclass(frozen=True)
class DecisionPolicy:
    intervention_cost: float = 25.0
    wait_cost: float = 0.0
    retry_min_net_value: float = 50.0
    wait_min_net_value: float = 10.0
    diminishing_return_factor: float = 0.75
    max_wait_minutes: int = 120
    max_attempts: int = MAX_RECOVERY_ATTEMPTS

    def as_dict(self) -> dict[str, float | int]:
        return {
            "intervention_cost": self.intervention_cost,
            "wait_cost": self.wait_cost,
            "retry_min_net_value": self.retry_min_net_value,
            "wait_min_net_value": self.wait_min_net_value,
            "diminishing_return_factor": self.diminishing_return_factor,
            "max_wait_minutes": self.max_wait_minutes,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True)
class DecisionOutcome:
    decision: str
    reason: str
    recovery_probability: float
    order_amount: float
    expected_recovered_value: float
    intervention_cost: float
    net_expected_value: float
    attempt_number: int
    action_taken: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "recovery_probability": float(self.recovery_probability),
            "order_amount": float(self.order_amount),
            "expected_recovered_value": float(self.expected_recovered_value),
            "intervention_cost": float(self.intervention_cost),
            "net_expected_value": float(self.net_expected_value),
            "attempt_number": int(self.attempt_number),
            "action_taken": bool(self.action_taken),
        }


class AIDecisionEngine:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, policy: DecisionPolicy | None = None):
        self.model_path = Path(model_path)
        self.model = CatBoostClassifier()
        self.model.load_model(str(self.model_path))
        self.policy = policy or DecisionPolicy()

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _case_to_feature_row(case: SyntheticCase | Mapping[str, Any]) -> dict[str, Any]:
        from ml.shared_semantics import enrich_case_features

        if isinstance(case, Mapping):
            base_case = dict(case)
        else:
            base_case = case.to_dict()

        features = enrich_case_features(base_case)
        return {column: features.get(column, 0) for column in FEATURE_COLUMNS}

    def predict_case(self, case: SyntheticCase | Mapping[str, Any]) -> dict[str, float]:
        row = self._case_to_feature_row(case)
        frame = pd.DataFrame([row])
        probability = float(self.model.predict_proba(frame)[0, 1])
        return {
            "recovery_probability": probability,
        }

    def decide(self, case: SyntheticCase | Mapping[str, Any]) -> DecisionOutcome:
        if isinstance(case, Mapping):
            amount = float(case.get("amount", 0))
            failure_code = str(case.get("failure_code") or "PAYMENT_ERROR")
            attempt_number = int(case.get("attempt_number", 1))
            time_since_failure = int(case.get("time_since_failure", 0))
            already_paid = bool(case.get("already_paid", False))
            max_attempts = min(int(case.get("max_attempts", MAX_RECOVERY_ATTEMPTS)), MAX_RECOVERY_ATTEMPTS)
        else:
            amount = float(case.amount)
            failure_code = str(case.failure_code or "PAYMENT_ERROR")
            attempt_number = int(case.attempt_number)
            time_since_failure = int(case.time_since_failure)
            already_paid = bool(case.already_paid)
            max_attempts = min(int(case.max_attempts), MAX_RECOVERY_ATTEMPTS)

        prediction = self.predict_case(case)
        probability = prediction["recovery_probability"]
        
        # Economic Reasoning
        expected_recovered_value = probability * amount
        
        # Account for diminishing returns on repeated attempts
        attempt_penalty = self.policy.diminishing_return_factor ** max(0, attempt_number - 1)
        adjusted_ev = expected_recovered_value * attempt_penalty
        
        net_expected_value = adjusted_ev - self.policy.intervention_cost

        # Default Outcome constructor helper
        def make_outcome(
            decision: str,
            reason: str,
            action_taken: bool,
            cost: float | None = None,
        ) -> DecisionOutcome:
            applied_cost = self.policy.intervention_cost if cost is None else cost
            return DecisionOutcome(
                decision=decision,
                reason=reason,
                recovery_probability=probability,
                order_amount=amount,
                expected_recovered_value=expected_recovered_value,
                intervention_cost=applied_cost,
                net_expected_value=adjusted_ev - applied_cost,
                attempt_number=attempt_number,
                action_taken=action_taken,
            )

        # 1. HARD SAFETY GUARDS
        if already_paid:
            return make_outcome("STOP", "already paid; hard safety guard override", False)
        if failure_code in NON_RECOVERABLE_FAILURE_CODES:
            return make_outcome("STOP", "non-recoverable failure code; hard safety guard override", False)
        if attempt_number >= max_attempts:
            return make_outcome("STOP", "max attempts reached; hard safety guard override", False)

        # 2. ECONOMIC REASONING
        if net_expected_value >= self.policy.retry_min_net_value:
            # If strongly positive EV, consider if WAIT is better
            if time_since_failure <= 30 and failure_code in TRANSIENT_FAILURE_CODES:
                return make_outcome("WAIT", "strongly positive EV, but recent transient failure justifies brief cooling window", True, self.policy.wait_cost)
            return make_outcome("RETRY", "strongly positive net expected value justifies immediate retry", True)
            
        elif net_expected_value >= self.policy.wait_min_net_value:
            # Marginal/uncertain EV
            return make_outcome("WAIT", "marginal net expected value; defer intervention to avoid immediate cost", True, self.policy.wait_cost)
            
        else:
            return make_outcome("STOP", "negative or insufficient net expected value does not justify cost", False)

