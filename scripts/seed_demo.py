from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from backend.database import SessionLocal, engine
from backend.models import MerchantOrder, PaymentAttempt, RecoveryAction
from backend.database import Base


def seed_demo_data() -> None:
    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()
    try:
        orders = [
            MerchantOrder(razorpay_order_id="order_demo_1", amount=50000, currency="INR", customer_id="cust_100", status="paid"),
            MerchantOrder(razorpay_order_id="order_demo_2", amount=25000, currency="INR", customer_id="cust_200", status="recoverable"),
            MerchantOrder(razorpay_order_id="order_demo_3", amount=75000, currency="INR", customer_id="cust_300", status="recovered"),
            MerchantOrder(razorpay_order_id="order_demo_4", amount=15000, currency="INR", customer_id="cust_400", status="exhausted"),
        ]
        db.add_all(orders)
        db.commit()

        db.add_all([
            PaymentAttempt(razorpay_payment_id="pay_demo_1", razorpay_order_id="order_demo_1", amount=50000, status="captured", method="card", attempt_number=1, order_id=orders[0].internal_id),
            PaymentAttempt(razorpay_payment_id="pay_demo_2", razorpay_order_id="order_demo_2", amount=25000, status="failed", method="netbanking", failure_code="PAYMENT_ERROR", failure_description="Bank declined", attempt_number=1, order_id=orders[1].internal_id),
            PaymentAttempt(razorpay_payment_id="pay_demo_3", razorpay_order_id="order_demo_3", amount=75000, status="captured", method="upi", attempt_number=2, order_id=orders[2].internal_id),
            PaymentAttempt(razorpay_payment_id="pay_demo_4", razorpay_order_id="order_demo_4", amount=15000, status="failed", method="card", failure_code="PAYMENT_ERROR", failure_description="Customer closed checkout", attempt_number=1, order_id=orders[3].internal_id),
        ])
        db.commit()

        db.add_all([
            RecoveryAction(order_id=orders[1].internal_id, action_type="recovery_candidate", reason="payment failed and order not paid", attempt_number=1, status="eligible", created_at=datetime.utcnow()),
            RecoveryAction(order_id=orders[2].internal_id, action_type="retry_payment", reason="recovered successfully", attempt_number=1, status="success", created_at=datetime.utcnow()),
            RecoveryAction(order_id=orders[3].internal_id, action_type="recovery_exhausted", reason="max retries reached", attempt_number=2, status="exhausted", created_at=datetime.utcnow()),
        ])
        db.commit()
        print("Seeded demo records successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_demo_data()
