from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import RAZORPAY_KEY_ID
from backend.database import Base, engine, ensure_recovery_audit_columns
from backend.routes.dashboard import router as dashboard_router
from backend.routes.orders import router as orders_router
from backend.routes.payments import router as payments_router
from backend.routes.simulation import router as simulation_router
from backend.routes.webhooks import router as webhooks_router

app = FastAPI(title="Revenue Recovery", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(orders_router)
app.include_router(payments_router)
app.include_router(webhooks_router)
app.include_router(dashboard_router)
app.include_router(simulation_router)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


@app.on_event("startup")
def startup_event() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_recovery_audit_columns()


@app.get("/config")
def get_config() -> dict[str, str]:
    return {"razorpay_key_id": RAZORPAY_KEY_ID}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
