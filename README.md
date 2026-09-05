# AI Revenue Recovery Agent

Stateful AI-powered revenue recovery for Razorpay merchants. When a payment fails, the system automatically evaluates whether and how to recover the sale using a CatBoost ML model, an economic expected-value decision engine, and a stateful recovery agent.

> **Full system design documentation:** [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)

---

## What this system does

1. Merchant creates a Razorpay order and customer pays via Razorpay Standard Checkout.
2. Payment fails — Razorpay sends a `payment.failed` webhook.
3. The Recovery Agent:
   - Observes: checks DB-backed attempt count and order status.
   - Diagnoses: classifies failure as transient, repeated, or method-specific degradation.
   - Evaluates: runs CatBoost + EV decision engine across all available payment methods.
   - Plans: picks best method (RETRY / SWITCH_PAYMENT_METHOD / WAIT / STOP).
   - Executes: creates a bounded `RecoveryAction` in the database.
4. Merchant dashboard shows AI decision, EV, recovery probability, and agent trace.
5. A `payment.captured` webhook closes the loop with recovery attribution (RECOVERY_SUCCESS vs NATURAL_SUCCESS).

This is merchant revenue recovery — not a refund or customer-money recovery system.

---

## 1. Problem being solved

Many merchants lose revenue when a customer starts a checkout flow, a payment fails, and the merchant does not promptly retry or recover the sale. This system uses AI to make that recovery intelligent:

- Not every failure is recoverable — hard safety guards prevent wasteful retries.
- Repeated attempts have diminishing economic value — the EV model accounts for this.
- Different payment methods have different per-customer success rates — the agent switches methods when beneficial.
- Customer history across prior orders informs each decision.

---

## 2. Why this is revenue recovery rather than customer-money/refund recovery

This system focuses on merchant revenue that was expected but not yet realized:

- A valid order exists.
- Payment was attempted but failed or not yet captured.
- The merchant may retry payment against the same Razorpay order.
- Recovered revenue is recognized as a successful sale recovery.

It does not represent a refund or a customer dispute.

---

## 3. Architecture

```
backend/
  agent.py              — RecoveryAgent: Observe→Diagnose→Evaluate→Plan→Execute→Verify
  config.py             — Environment config (MAX_RECOVERY_ATTEMPTS, LLM settings)
  models.py             — SQLAlchemy ORM: MerchantOrder, PaymentAttempt, RecoveryAction, AgentTrace
  razorpay_client.py    — Razorpay API client (order creation, signature verification)
  routes/
    orders.py           — POST /orders, GET /orders
    payments.py         — candidates, trace, voice-recovery, simulation, abandon
    webhooks.py         — HMAC-verified payment.failed / payment.captured / order.paid
    dashboard.py        — GET /dashboard/metrics
    simulation.py       — GET /simulation/baseline
  services/
    customer_history.py — get_customer_payment_history() — cross-order longitudinal history
    recovery.py         — evaluate_recovery_policy(), mark_recovery_success()
    recovery_executor.py — bounded idempotent RecoveryAction executor
    voice_recovery.py   — Groq LLM / deterministic Hinglish fallback

ml/
  decision_engine.py    — AIDecisionEngine: EV = prob × amount × 0.75^(n-1) − cost
  shared_semantics.py   — enrich_case_features(), calculate_latent_recovery_score()
  generate_dataset.py   — synthetic training data (50,000 cases)
  train.py              — CatBoost + LogisticRegression training pipeline
  evaluate.py           — ROC-AUC, PR-AUC, Brier, calibration, business metrics
  benchmark_compare.py  — Baseline vs AI capability comparison

frontend/
  index.html            — Merchant dashboard
  app.js                — Dashboard JS (metrics, candidates, trace, voice, simulation)

tests/                  — 89 pytest tests
docs/
  SYSTEM_DESIGN.md      — Complete canonical system design document
```

---

## 4. Expected Value economics

The decision engine computes:

```
probability              = CatBoost.predict_proba(features)[1]
expected_recovered_value = probability × amount
attempt_penalty          = 0.75 ^ max(0, attempt_number − 1)
adjusted_ev              = expected_recovered_value × attempt_penalty
net_expected_value       = adjusted_ev − 25   (intervention_cost)

if net_ev >= 50 → RETRY (or WAIT if recent transient failure)
if net_ev >= 10 → WAIT
if net_ev < 10  → STOP
```

EV is always less than the order amount (probability < 1). Repeated attempts face diminishing returns. The initial payment failure is attempt_number=1 and carries no penalty — only actual interventions are penalized.

---

## 5. Hard safety rules

These override economics unconditionally:

- `already_paid` → STOP immediately.
- Non-recoverable failure codes (CARD_DECLINED, INSUFFICIENT_FUNDS, AUTHENTICATION_ERROR, BANK_DECLINED, INVALID_ACCOUNT, CURRENCY_NOT_ALLOWED) → STOP immediately.
- `attempt_number >= MAX_RECOVERY_ATTEMPTS (4)` → STOP immediately.

---

## 6. Razorpay payment/order lifecycle

1. Merchant creates an order through the backend.
2. Backend creates a Razorpay order via the Orders API.
3. Frontend opens Razorpay Standard Checkout with the Razorpay order_id.
4. Customer completes or fails the payment.
5. Razorpay sends webhook events to the backend (HMAC-SHA256 verified).
6. Backend verifies the signature and updates local state.
7. The Recovery Agent evaluates the failed payment and records its decision.
8. A valid retry reuses the same Razorpay order — no new order is created.

Amounts are in paise and currency is INR.

---

## 7. Benchmark results

Capability comparison over 1,000 synthetic cases (seed=42, deterministic):

| Metric | Baseline (deterministic) | AI Agent | Lift |
|--------|-------------------------|---------|------|
| Failed orders | 361 | 361 | — |
| Recovered orders | 100 | 107 | +7 |
| Recovered revenue | Rs.72,292 | Rs.77,902 | +Rs.5,610 |
| Recovery rate | 27.70% | 29.64% | +1.94pp |
| Unnecessary interventions | 0 | 0 | 0 |

> This is a **capability comparison** (AI agent has more interventions + EV-based selection), not a pure ML quality benchmark. See [SYSTEM_DESIGN.md §12](docs/SYSTEM_DESIGN.md#12-synthetic-benchmark) for details.

---

## 8. LLM / Voice layer

The system generates Hinglish (Hindi-English) recovery messages via Groq and renders them as text alongside real-time browser audio playback (`window.speechSynthesis` / `hi-IN`). The global **"Enable Agent Voice"** control on the dashboard enables spoken customer recovery guidance with deterministic speech sanitization (emojis and internal technical tokens stripped). The LLM is **communication-only** — it does not make financial decisions or change the agent's recovery action.

- Spoken prompt engineering enforces natural conversational speech without emojis, bullet points, or internal jargon.
- Deterministic sanitization (`sanitize_speech_text`) cleans emojis, markdown artifacts, and technical tokens (`payment.failed`, `net EV`, `step` prefixes).
- Valid LLM responses are filtered for forbidden content ("guaranteed", "charge", probability/EV values).
- Falls back to deterministic scenario × decision templates when LLM is unavailable.

---

## 9. Data model

Core tables: `MerchantOrder`, `PaymentAttempt`, `RecoveryAction`, `AgentTrace`, `WebhookEvent`.

See [SYSTEM_DESIGN.md §15](docs/SYSTEM_DESIGN.md#15-database-and-data-model) for the full ER diagram.

---

## 10. How to run the project

Create a local `.env` file:

```env
RAZORPAY_KEY_ID=rzp_test_xxxxxxxxxxxxxxxxx
RAZORPAY_KEY_SECRET=your_test_secret
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret
DATABASE_URL=sqlite:///./revenue_recovery.db
MAX_RECOVERY_ATTEMPTS=4
FRONTEND_URL=http://localhost:8000
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
LLM_API_KEY=your_groq_key
LLM_MODEL=groq/compound-mini
```

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the backend:

```bash
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Open the dashboard at `http://localhost:8000/`.

Run automated tests:

```bash
.venv\Scripts\python.exe -m pytest tests/ -v
```

Train the ML model:

```bash
.venv\Scripts\python.exe -m ml.train
```

---

## 11. Razorpay Test Mode setup

1. Log in to the Razorpay Dashboard.
2. Create or use a Test Mode account.
3. Copy the test Key ID and Key Secret into `.env`.
4. Set a webhook secret in the Razorpay Dashboard and copy it into `RAZORPAY_WEBHOOK_SECRET`.
5. Configure a public webhook URL (tunneling service) pointing to `/webhooks/razorpay`.

---

## Notes

- No Razorpay secret or LLM API key is ever sent to the frontend.
- The UI only receives the public Razorpay Key ID via `GET /config`.
- `MAX_RECOVERY_ATTEMPTS` is hard-capped at 4 in `config.py` regardless of `.env` setting.
- The `intervention_cost = 25` (paise) is a simplified internal accounting assumption, not a real Razorpay fee.
- See [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md) for the complete technical specification.
