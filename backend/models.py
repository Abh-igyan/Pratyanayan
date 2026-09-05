from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, Float
from sqlalchemy.orm import relationship

from backend.database import Base


class MerchantOrder(Base):
    __tablename__ = "merchant_orders"

    internal_id = Column(Integer, primary_key=True, index=True)
    razorpay_order_id = Column(String(255), unique=True, index=True, nullable=False)
    amount = Column(Integer, nullable=False)
    currency = Column(String(10), default="INR", nullable=False)
    customer_id = Column(String(255), nullable=False)
    status = Column(String(50), default="created", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    payment_attempts = relationship("PaymentAttempt", back_populates="order")
    recovery_actions = relationship("RecoveryAction", back_populates="order")
    agent_traces = relationship("AgentTrace", back_populates="order")


class PaymentAttempt(Base):
    __tablename__ = "payment_attempts"

    internal_id = Column(Integer, primary_key=True, index=True)
    razorpay_payment_id = Column(String(255), nullable=True, index=True)
    razorpay_order_id = Column(String(255), nullable=False, index=True)
    amount = Column(Integer, nullable=False)
    status = Column(String(50), default="pending", nullable=False)
    method = Column(String(50), nullable=True)
    failure_code = Column(String(100), nullable=True)
    failure_description = Column(String(255), nullable=True)
    attempt_number = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    order_id = Column(ForeignKey("merchant_orders.internal_id"), nullable=False)

    order = relationship("MerchantOrder", back_populates="payment_attempts")


class RecoveryAction(Base):
    __tablename__ = "recovery_actions"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(ForeignKey("merchant_orders.internal_id"), nullable=False)
    action_type = Column(String(50), nullable=False)
    reason = Column(String(255), nullable=False)
    attempt_number = Column(Integer, default=1, nullable=False)
    status = Column(String(50), default="active", nullable=False)
    case_status = Column(String(50), nullable=True)
    recovery_probability = Column(String(50), nullable=True)
    expected_recovered_value = Column(String(50), nullable=True)
    outcome_status = Column(String(50), nullable=True)
    error = Column(Text, nullable=True)
    planned_method = Column(String(50), nullable=True)
    actual_payment_method = Column(String(50), nullable=True)
    payment_attempt_id = Column(ForeignKey("payment_attempts.internal_id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    order = relationship("MerchantOrder", back_populates="recovery_actions")


class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey("merchant_orders.internal_id"), index=True)
    razorpay_order_id = Column(String(100), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Workflow Steps
    event = Column(String(50))
    diagnosis = Column(Text)
    decision_inputs = Column(Text)  # JSON dump of case_dict
    
    # Evaluation
    probability = Column(Float, nullable=True)
    expected_value = Column(Float, nullable=True)
    net_ev = Column(Float, nullable=True)
    
    # Plan
    selected_action = Column(String(50))
    planned_method = Column(String(50), nullable=True)
    actual_payment_method = Column(String(50), nullable=True)
    
    # Execution & Verification
    execution_result = Column(Text, nullable=True)
    webhook_outcome = Column(String(50), nullable=True)
    recovered_revenue = Column(Float, nullable=True)
    
    order = relationship("MerchantOrder", back_populates="agent_traces")



class WebhookEvent(Base):
    __tablename__ = "webhook_events"

    id = Column(Integer, primary_key=True, index=True)
    razorpay_event_id = Column(String(255), unique=True, index=True, nullable=False)
    event_type = Column(String(100), nullable=False)
    payload = Column(Text, nullable=False)
    received_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    processing_status = Column(String(50), default="queued", nullable=False)
