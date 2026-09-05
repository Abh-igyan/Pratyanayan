from __future__ import annotations

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import MerchantOrder, PaymentAttempt, RecoveryAction, WebhookEvent
from backend.razorpay_client import verify_signature
from backend.services.payment_verification import upsert_payment_from_payload
from backend.services.recovery import mark_order_recoverable
from backend.services.recovery import mark_recovery_success

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/razorpay")
async def razorpay_webhook(request: Request, x_razorpay_signature: str | None = Header(default=None, alias="X-Razorpay-Signature"), db: Session = Depends(get_db)):
    raw_body = await request.body()
    if not x_razorpay_signature:
        raise HTTPException(status_code=400, detail="Missing Razorpay signature")

    if not verify_signature(raw_body, x_razorpay_signature):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    payload = json.loads(raw_body.decode("utf-8"))
    event_type = payload.get("event")
    event_payload = payload.get("payload", {})
    payment_entity = event_payload.get("payment", {}).get("entity") or event_payload.get("payment", {})
    order_entity = event_payload.get("order", {}).get("entity") or event_payload.get("order", {})
    razorpay_event_id = payload.get("event_id") or payment_entity.get("id") or order_entity.get("id") or f"evt-{datetime.utcnow().timestamp()}"

    existing = db.query(WebhookEvent).filter(WebhookEvent.razorpay_event_id == razorpay_event_id).first()
    if existing:
        return {"status": "duplicate", "event_id": razorpay_event_id}

    webhook = WebhookEvent(
        razorpay_event_id=razorpay_event_id,
        event_type=event_type,
        payload=json.dumps(payload),
        received_at=datetime.utcnow(),
        processing_status="received",
    )
    db.add(webhook)
    db.commit()
    db.refresh(webhook)

    if event_type == "payment.failed":
        order = db.query(MerchantOrder).filter(MerchantOrder.razorpay_order_id == payment_entity.get("order_id")).first()
        if order:
            if order.status == "paid":
                logger.info("Order %s already paid; ignoring late payment.failed webhook", order.razorpay_order_id)
                webhook.processing_status = "processed"
                webhook.processed_at = datetime.utcnow()
                db.commit()
                return {"status": "already_paid", "event_id": razorpay_event_id}

            record = PaymentAttempt(
                razorpay_payment_id=payment_entity.get("id"),
                razorpay_order_id=order.razorpay_order_id,
                amount=int(payment_entity.get("amount") or order.amount),
                status="failed",
                method=payment_entity.get("method"),
                failure_code=payment_entity.get("error_code"),
                failure_description=payment_entity.get("error_description"),
                attempt_number=(db.query(PaymentAttempt).filter(PaymentAttempt.razorpay_order_id == order.razorpay_order_id).count() + 1),
                order_id=order.internal_id,
            )
            db.add(record)
            db.commit()
            
            # Close pending retries that failed
            db.query(RecoveryAction).filter(
                RecoveryAction.order_id == order.internal_id,
                RecoveryAction.action_type == "recovery_retry_pending",
                RecoveryAction.status == "pending",
            ).update({
                "status": "failed",
                "case_status": "active",
                "outcome_status": "payment_failed",
                "completed_at": datetime.utcnow(),
            })
            db.commit()
            
            logger.info("payment.failed for order=%s payment_id=%s -> created failed attempt #%s", order.razorpay_order_id, payment_entity.get("id"), record.attempt_number)
            
            # Delegate to RecoveryAgent
            from backend.agent import RecoveryAgent
            agent = RecoveryAgent(db)
            agent.observe_failure(order, record)
        else:
            logger.warning("payment.failed webhook matched no local order for razorpay_order_id=%s", payment_entity.get("order_id"))

    elif event_type == "payment.captured":
        order = db.query(MerchantOrder).filter(MerchantOrder.razorpay_order_id == payment_entity.get("order_id")).first()
        if order:
            existing_attempt = db.query(PaymentAttempt).filter(PaymentAttempt.razorpay_payment_id == payment_entity.get("id")).first()
            if order.status == "paid" and existing_attempt and existing_attempt.status == "captured":
                logger.info("Payment %s for order %s already captured and paid; idempotent return", existing_attempt.razorpay_payment_id, order.razorpay_order_id)
                webhook.processing_status = "processed"
                webhook.processed_at = datetime.utcnow()
                db.commit()
                return {"status": "already_processed", "event_id": razorpay_event_id}

            attempt = upsert_payment_from_payload(db, order, payment_entity)
            
            # Delegate to RecoveryAgent
            from backend.agent import RecoveryAgent
            agent = RecoveryAgent(db)
            agent.verify_captured(order, attempt)
        else:
            logger.warning("payment.captured webhook matched no local order for razorpay_order_id=%s", payment_entity.get("order_id"))

    elif event_type == "order.paid":
        merchant_order = db.query(MerchantOrder).filter(MerchantOrder.razorpay_order_id == order_entity.get("id")).first()
        if merchant_order:
            if merchant_order.status == "paid":
                logger.info("Order %s already paid; idempotent order.paid webhook", merchant_order.razorpay_order_id)
                webhook.processing_status = "processed"
                webhook.processed_at = datetime.utcnow()
                db.commit()
                return {"status": "already_processed", "event_id": razorpay_event_id}

            # Delegate to RecoveryAgent
            from backend.agent import RecoveryAgent
            agent = RecoveryAgent(db)
            agent.verify_order_paid(merchant_order)

    webhook.processing_status = "processed"
    webhook.processed_at = datetime.utcnow()
    db.commit()
    return {"status": "ok", "event_id": razorpay_event_id}
