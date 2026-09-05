from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.config import DATABASE_URL

Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def ensure_recovery_audit_columns() -> None:
    """Add audit columns to an existing local database without data loss."""
    inspector = inspect(engine)
    if "recovery_actions" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("recovery_actions")}
    additions = {
        "recovery_probability": "VARCHAR(50)",
        "expected_recovered_value": "VARCHAR(50)",
        "outcome_status": "VARCHAR(50)",
        "error": "TEXT",
        "planned_method": "VARCHAR(50)",
        "actual_payment_method": "VARCHAR(50)",
        "payment_attempt_id": "INTEGER",
    }
    with engine.begin() as connection:
        for name, column_type in additions.items():
            if name not in existing:
                connection.execute(text(
                    f"ALTER TABLE recovery_actions ADD COLUMN {name} {column_type}"
                ))
        if "agent_traces" in inspector.get_table_names():
            trace_columns = {column["name"] for column in inspector.get_columns("agent_traces")}
            if "actual_payment_method" not in trace_columns:
                connection.execute(text(
                    "ALTER TABLE agent_traces ADD COLUMN actual_payment_method VARCHAR(50)"
                ))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
