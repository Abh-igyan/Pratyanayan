from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from backend.models import MerchantOrder, PaymentAttempt
from backend.razorpay_client import fetch_order, fetch_payment, list_payments_for_order


def verify_order_and_payment_state(db: Session, order: MerchantOrder) -> dict[str, Any]:
    if not order.razorpay_order_id:
        return {"paid": False, "status": "missing_order_id", "payment_id": None}

    try:
        razorpay_order = fetch_order(order.razorpay_order_id)
    except Exception as exc:  # pragma: no cover - network integration case
        return {"paid": False, "status": "verification_failed", "error": str(exc), "payment_id": None}

    order_status = razorpay_order.get("status")
    amount_paid = int(razorpay_order.get("amount_paid", 0) or 0)
    paid = order_status == "paid" or amount_paid >= order.amount

    payment_items = list_payments_for_order(order.razorpay_order_id)
    latest_payment = None
    for item in payment_items:
        if not latest_payment or item.get("created_at", 0) >= latest_payment.get("created_at", 0):
            latest_payment = item

    payment_id = latest_payment.get("id") if latest_payment else None
    method = latest_payment.get("method") if latest_payment else None
    status = latest_payment.get("status") if latest_payment else "unknown"

    if paid:
        order.status = "paid"
        order.updated_at = datetime.utcnow()
        db.commit()

    return {
        "paid": paid,
        "status": order_status or status,
        "payment_id": payment_id,
        "method": method,
        "amount_paid": amount_paid,
        "order_data": razorpay_order,
    }


def make_payment_attempt_record(db: Session, order: MerchantOrder, payment_payload: dict[str, Any]) -> PaymentAttempt:
    attempt = PaymentAttempt(
        razorpay_payment_id=payment_payload.get("id"),
        razorpay_order_id=order.razorpay_order_id,
        amount=int(payment_payload.get("amount") or order.amount),
        status=payment_payload.get("status") or "unknown",
        method=payment_payload.get("method"),
        failure_code=payment_payload.get("error_code") or payment_payload.get("error", {}).get("code"),
        failure_description=payment_payload.get("error_description") or payment_payload.get("error", {}).get("description"),
        attempt_number=1,
        order_id=order.internal_id,
    )

    existing_attempts = db.query(PaymentAttempt).filter(PaymentAttempt.razorpay_order_id == order.razorpay_order_id).count()
    if existing_attempts:
        attempt.attempt_number = existing_attempts + 1

    db.add(attempt)
    db.commit()
    db.refresh(attempt)
    return attempt


def upsert_payment_from_payload(db: Session, order: MerchantOrder, payment_payload: dict[str, Any]) -> PaymentAttempt:
    payment_id = payment_payload.get("id")
    existing = db.query(PaymentAttempt).filter(PaymentAttempt.razorpay_payment_id == payment_id).first()
    if existing:
        if payment_payload.get("status"):
            existing.status = payment_payload.get("status")
        if payment_payload.get("method"):
            existing.method = payment_payload.get("method")
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing
    return make_payment_attempt_record(db, order, payment_payload)


def resolve_final_payment_state(db: Session, order: MerchantOrder, payment_id: str | None = None) -> dict[str, Any]:
    if payment_id:
        payment = fetch_payment(payment_id)
        return {
            "paid": payment.get("status") == "captured",
            "status": payment.get("status"),
            "payment": payment,
        }
    return verify_order_and_payment_state(db, order)
