from __future__ import annotations

from fastapi import APIRouter

from backend.services.synthetic_simulation import run_synthetic_simulation

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.get("/baseline")
def run_baseline_simulation() -> dict[str, object]:
    return run_synthetic_simulation(seed=42, count=1000)
