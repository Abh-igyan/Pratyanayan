from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx

from backend.config import RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, RAZORPAY_WEBHOOK_SECRET

BASE_URL = "https://api.razorpay.com/v1"


def _headers() -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _auth() -> tuple[str, str]:
    return RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET


def create_razorpay_order(amount_paise: int, currency: str, receipt: str, notes: dict[str, str]) -> dict[str, Any]:
    if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
        raise RuntimeError("Razorpay test credentials are not configured.")

    payload = {
        "amount": amount_paise,
        "currency": currency,
        "receipt": receipt,
        "notes": notes,
    }
    response = httpx.post(f"{BASE_URL}/orders", json=payload, auth=_auth(), headers=_headers(), timeout=15.0)
    response.raise_for_status()
    return response.json()


def fetch_order(razorpay_order_id: str) -> dict[str, Any]:
    response = httpx.get(f"{BASE_URL}/orders/{razorpay_order_id}", auth=_auth(), headers=_headers(), timeout=15.0)
    response.raise_for_status()
    return response.json()


def fetch_payment(payment_id: str) -> dict[str, Any]:
    response = httpx.get(f"{BASE_URL}/payments/{payment_id}", auth=_auth(), headers=_headers(), timeout=15.0)
    response.raise_for_status()
    return response.json()


def list_payments_for_order(razorpay_order_id: str, count: int = 10) -> list[dict[str, Any]]:
    response = httpx.get(
        f"{BASE_URL}/payments",
        params={"order_id": razorpay_order_id, "count": count},
        auth=_auth(),
        headers=_headers(),
        timeout=15.0,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("items", [])


def verify_signature(raw_body: bytes, signature: str) -> bool:
    if not RAZORPAY_WEBHOOK_SECRET:
        return False
    expected = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def verify_payment_signature(order_id: str, payment_id: str, signature: str) -> bool:
    if not RAZORPAY_KEY_SECRET:
        return False
    payload = f"{order_id}|{payment_id}".encode("utf-8")
    expected = hmac.new(RAZORPAY_KEY_SECRET.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def build_order_notes(customer_id: str, internal_id: int) -> dict[str, str]:
    return {
        "customer_id": customer_id,
        "internal_order_id": str(internal_id),
    }
