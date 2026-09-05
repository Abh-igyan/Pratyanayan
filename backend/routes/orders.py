from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import MerchantOrder
from backend.razorpay_client import build_order_notes, create_razorpay_order
from backend.schemas import OrderCreateRequest, OrderResponse

router = APIRouter(prefix="/orders", tags=["orders"])


@router.post("", response_model=OrderResponse)
def create_order(payload: OrderCreateRequest, db: Session = Depends(get_db)):
    order = MerchantOrder(
        razorpay_order_id="",
        amount=payload.amount,
        currency=payload.currency,
        customer_id=payload.customer_id,
        status="created",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(order)
    db.commit()
    db.refresh(order)

    receipt = f"order-{order.internal_id}-{uuid.uuid4().hex[:10]}"
    notes = build_order_notes(payload.customer_id, order.internal_id)
    try:
        razorpay_response = create_razorpay_order(payload.amount, payload.currency, receipt, notes)
    except Exception as exc:
        db.delete(order)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Unable to create Razorpay order: {exc}") from exc

    order.razorpay_order_id = razorpay_response["id"]
    order.status = "created"
    order.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(order)

    return OrderResponse(
        internal_id=order.internal_id,
        razorpay_order_id=order.razorpay_order_id,
        amount=order.amount,
        currency=order.currency,
        customer_id=order.customer_id,
        status=order.status,
    )


@router.get("/{internal_id}")
def get_order(internal_id: int, db: Session = Depends(get_db)):
    order = db.query(MerchantOrder).filter(MerchantOrder.internal_id == internal_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return {
        "internal_id": order.internal_id,
        "razorpay_order_id": order.razorpay_order_id,
        "amount": order.amount,
        "currency": order.currency,
        "customer_id": order.customer_id,
        "status": order.status,
    }


@router.get("")
def list_orders(db: Session = Depends(get_db)):
    orders = db.query(MerchantOrder).order_by(MerchantOrder.created_at.desc()).all()
    return [{
        "internal_id": order.internal_id,
        "razorpay_order_id": order.razorpay_order_id,
        "amount": order.amount,
        "currency": order.currency,
        "customer_id": order.customer_id,
        "status": order.status,
        "created_at": order.created_at.isoformat(),
    } for order in orders]
