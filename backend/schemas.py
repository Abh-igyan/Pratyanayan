from __future__ import annotations

from pydantic import BaseModel, Field


class OrderCreateRequest(BaseModel):
    customer_id: str = Field(..., min_length=1)
    amount: int = Field(..., gt=0)
    currency: str = "INR"


class OrderResponse(BaseModel):
    internal_id: int
    razorpay_order_id: str
    amount: int
    currency: str
    customer_id: str
    status: str


class PaymentVerificationRequest(BaseModel):
    payment_id: str
    order_id: str
    signature: str


class RetryPaymentRequest(BaseModel):
    order_id: str


class DashboardMetrics(BaseModel):
    total_orders: int
    total_expected_revenue: int
    successful_orders: int
    failed_payment_attempts: int
    recovery_candidates: int
    recovery_attempts: int
    recovered_orders: int
    recovered_revenue: int
    recovery_rate: float
    revenue_recovery_rate: float
    exhausted_escalated_cases: int
