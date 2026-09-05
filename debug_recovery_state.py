from backend.database import SessionLocal
from backend.models import MerchantOrder, PaymentAttempt, RecoveryAction


def inspect_order(razorpay_order_id: str):
    db = SessionLocal()
    try:
        order = db.query(MerchantOrder).filter(MerchantOrder.razorpay_order_id == razorpay_order_id).first()
        if not order:
            print(f"No order found for razorpay_order_id={razorpay_order_id}")
            return

        print("\nORDER")
        print(f"internal_id={order.internal_id}")
        print(f"razorpay_order_id={order.razorpay_order_id}")
        print(f"status={order.status}")
        print(f"amount={order.amount}")

        print("\nPAYMENT ATTEMPTS")
        attempts = db.query(PaymentAttempt).filter(PaymentAttempt.razorpay_order_id == razorpay_order_id).order_by(PaymentAttempt.attempt_number).all()
        if not attempts:
            print("No payment attempts found")
        for a in attempts:
            print({
                "internal_id": a.internal_id,
                "razorpay_payment_id": a.razorpay_payment_id,
                "attempt_number": a.attempt_number,
                "status": a.status,
                "method": a.method,
                "failure_code": a.failure_code,
                "failure_description": a.failure_description,
                "amount": a.amount,
            })

        print("\nRECOVERY ACTIONS")
        actions = db.query(RecoveryAction).filter(RecoveryAction.order_id == order.internal_id).order_by(RecoveryAction.created_at).all()
        if not actions:
            print("No recovery actions found")
        for a in actions:
            print({
                "id": a.id,
                "action_type": a.action_type,
                "status": a.status,
                "case_status": a.case_status,
                "attempt_number": a.attempt_number,
                "reason": a.reason,
                "created_at": str(a.created_at),
            })

        print("\nDASHBOARD SUMMARY")
        from backend.services.recovery import compute_dashboard_metrics
        print(compute_dashboard_metrics(db))
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python debug_recovery_state.py <razorpay_order_id>")
        sys.exit(1)
    inspect_order(sys.argv[1])
