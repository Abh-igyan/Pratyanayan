import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./revenue_recovery.db")
# Initial payment failure is not an intervention; this is the hard recovery cap.
MAX_RECOVERY_ATTEMPTS = min(4, int(os.getenv("MAX_RECOVERY_ATTEMPTS", "4")))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8000")
LLM_API_URL = os.getenv("LLM_API_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "groq/compound-mini")

if not RAZORPAY_KEY_ID:
    print("Warning: RAZORPAY_KEY_ID is not set. Razorpay API calls will fail until configured.")
if not RAZORPAY_KEY_SECRET:
    print("Warning: RAZORPAY_KEY_SECRET is not set. Razorpay API calls will fail until configured.")
if not RAZORPAY_WEBHOOK_SECRET:
    print("Warning: RAZORPAY_WEBHOOK_SECRET is not set. Webhook verification will fail until configured.")
