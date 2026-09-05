from __future__ import annotations

import logging
from typing import Any

import httpx

from backend import config

logger = logging.getLogger(__name__)

ALLOWED_SCENARIOS = {"payment.failed", "checkout.abandoned", "payment.captured", "order.paid"}
ALLOWED_DECISIONS = {"RETRY", "WAIT", "STOP", "SWITCH_PAYMENT_METHOD"}

import re

# Comprehensive regex matching emojis, pictographs, symbols, dingbats, variation selectors, zero-width joiners
EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u27bf\ufe00-\ufe0f\u200d\u20e3]",
    flags=re.UNICODE,
)

# Technical or internal tokens that must never be vocalized by TTS
FORBIDDEN_SPEECH_PATTERNS = [
    re.compile(r"(?i)\b(payment\.failed|checkout\.abandoned|payment\.captured|order\.paid)\b"),
    re.compile(r"(?i)\b(recovery_success|natural_success|payment_failed)\b"),
    re.compile(r"(?i)\b(switch_payment_method|try_method)\b"),
    re.compile(r"(?i)\b(net[\s_]*ev|expected[\s_]*value|recovery[\s_]*probability|attempt[\s_]*penalty|latent[\s_]*score)\b"),
    re.compile(r"(?i)\bagent\s+step(\s+\d+)?\b"),
]

FORBIDDEN_WORDS = (
    "guarantee",
    "guaranteed",
    "charge",
    "debit",
    "net ev",
    "probability",
    "payment.failed",
    "checkout.abandoned",
    "recovery_success",
    "agent step",
)


def sanitize_speech_text(text: str) -> str:
    """Deterministically sanitize text for natural Text-to-Speech (TTS).

    Strips:
    - Emojis (e.g. folded hands, checkmarks, smileys)
    - Markdown symbols, asterisks, bullet points, brackets, quotes
    - Internal event codes (payment.failed, checkout.abandoned, etc.)
    - Internal decision tokens and metric jargon (net EV, probability, etc.)
    - Unnatural technical 'step' prefixes
    """
    if not text:
        return ""

    # 1. Strip emojis and unicode symbols
    cleaned = EMOJI_PATTERN.sub("", text)

    # 2. Remove markdown artifacts and bullets
    cleaned = re.sub(r"[\*\#\_\[\]\(\)\{\}\<\>\~\|\`\"]", "", cleaned)
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned, flags=re.MULTILINE)

    # 3. Strip internal technical tokens
    for pattern in FORBIDDEN_SPEECH_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # 4. Remove technical 'step' occurrences that cause awkward pronunciation (e.g. 'sa-tep')
    cleaned = re.sub(r"(?i)\bstep\s*\d+\b", "", cleaned)

    # 5. Normalize whitespace and punctuation
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)

    return cleaned


def build_voice_facts(
    *,
    scenario: str,
    decision: str,
    recommended_method: str | None,
    reason: str,
    customer_history: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the allowlisted, sanitised fact bundle sent to the LLM.

    The LLM sees *only* this bundle – no probability, no EV, no net_ev,
    no internal model terms, no financial decision fields.
    """
    # Sanitize reason to remove any internal model abbreviations before passing to LLM
    clean_reason = sanitize_speech_text(reason)[:240]
    return {
        "scenario": scenario if scenario in ALLOWED_SCENARIOS else "payment.failed",
        "decision": decision if decision in ALLOWED_DECISIONS else "STOP",
        "recommended_method": recommended_method,
        "reason": clean_reason,
        "customer_history": {
            "total_attempts": int((customer_history or {}).get("total_attempts", 0)),
            "recommended_method_success_rate": float(
                (customer_history or {}).get("method_success_rates", {}).get(recommended_method, 0.0)
            ) if recommended_method else 0.0,
        },
    }


def deterministic_fallback(facts: dict[str, Any]) -> str:
    """Scenario × decision specific Hinglish fallback messages.

    Each branch is communication-only — no financial decisions, no guarantees,
    no implication that payment was auto-charged, no emojis, no internal jargon.
    """
    decision = facts.get("decision", "STOP")
    scenario = facts.get("scenario", "payment.failed")
    method = facts.get("recommended_method")
    method_upper = method.upper() if method else None

    # Recovery success
    if scenario in {"payment.captured", "order.paid"}:
        msg = "Aapka payment successfully complete ho gaya! Transaction ke liye shukriya."
        return sanitize_speech_text(msg)

    # Scenario prefix
    if scenario == "checkout.abandoned":
        prefix = "Aapka checkout complete nahi hua."
    else:
        prefix = "Aapka payment complete nahi ho paaya."

    # Decision-specific messages
    if decision == "STOP":
        msg = (
            f"{prefix} Agar aap chahein to baad mein dobara try kar sakte hain — "
            "koi bhi payment tabhi hogi jab aap khud confirm karein."
        )
    elif decision == "WAIT":
        msg = (
            f"{prefix} Thodi der baad dobara try karna better ho sakta hai. "
            "Checkout aapke haath mein hai — jab ready hon tab continue karein."
        )
    elif decision == "SWITCH_PAYMENT_METHOD" and method_upper:
        msg = (
            f"{prefix} {method_upper} se try karna ek better option ho sakta hai. "
            "Payment tabhi hogi jab aap khud checkout continue karein."
        )
    elif decision == "RETRY" and method_upper:
        msg = (
            f"{prefix} Aap {method_upper} se dobara try kar sakte hain. "
            "Checkout aapke control mein hai."
        )
    else:
        # Generic RETRY without method
        msg = f"{prefix} Aap chahein to checkout dobara continue kar sakte hain."

    return sanitize_speech_text(msg)


def _valid_message(message: str) -> bool:
    if not message or not message.strip():
        return False
    # Must be reasonably concise for spoken audio
    if len(message) > 280:
        return False
    # No emojis allowed
    if EMOJI_PATTERN.search(message):
        return False
    # No forbidden terms allowed
    lowered = message.lower()
    return not any(word in lowered for word in FORBIDDEN_WORDS)


def _extract_content(payload: dict) -> str:
    """Extract text content from a chat completion response.

    Handles standard models (content field) and reasoning models
    that may put output in a 'reasoning' field when 'content' is empty.
    """
    try:
        message = payload["choices"][0]["message"]
        content = message.get("content") or ""
        if content.strip():
            return content.strip()
        # Reasoning models may use 'reasoning' key
        reasoning = message.get("reasoning") or ""
        return reasoning.strip()
    except (KeyError, IndexError, TypeError):
        return ""


def generate_recovery_message(
    *,
    scenario: str,
    decision: str,
    recommended_method: str | None,
    reason: str,
    customer_history: dict[str, Any] | None,
) -> dict[str, Any]:
    """Generate a Hinglish customer recovery message.

    Returns a dict with:
      message       – the final Hinglish text
      source        – "llm" or "fallback"
      facts         – the sanitised fact bundle (no secrets, no EV/probability)
      provider_configured – bool, safe to expose
      model         – model name, safe to expose
    """
    facts = build_voice_facts(
        scenario=scenario,
        decision=decision,
        recommended_method=recommended_method,
        reason=reason,
        customer_history=customer_history,
    )
    fallback = deterministic_fallback(facts)

    provider_configured = bool(config.LLM_API_URL and config.LLM_API_KEY)
    model_name = config.LLM_MODEL or "none"

    base_meta = {
        "facts": facts,
        "provider_configured": provider_configured,
        "model": model_name,
    }

    if not provider_configured:
        logger.info(
            "voice_recovery: provider not configured — using fallback. "
            "Set LLM_API_URL and LLM_API_KEY to enable Groq."
        )
        return {"message": fallback, "source": "fallback", **base_meta}

    logger.info(
        "voice_recovery: requesting LLM message. provider_configured=True model=%s scenario=%s decision=%s",
        model_name, facts["scenario"], facts["decision"],
    )

    system = (
        "You are a friendly, helpful customer voice assistant for an Indian payment gateway. "
        "Speak ONE short, polite, natural Hinglish sentence for Text-to-Speech (TTS) playback. "
        "STRICT VOICE RULES:\n"
        "- Conversational speech only: do NOT write display text, headings, or bullet points.\n"
        "- NEVER use any emojis or emoticons (no folded hands, smileys, icons, or pictographs).\n"
        "- NEVER use quotation marks, asterisks, or markdown symbols.\n"
        "- NEVER speak internal event names (no 'payment.failed', no 'checkout.abandoned').\n"
        "- NEVER speak internal action codes or technical jargon (no 'RETRY', 'WAIT', 'STOP', 'SWITCH_PAYMENT_METHOD', no 'step', no 'agent step', no 'EV', no 'probability').\n"
        "- Keep it under 240 characters. Explain simply what happened and remind the customer they are in full control to retry when ready."
    )
    try:
        response = httpx.post(
            config.LLM_API_URL,
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(facts)},
                ],
                "temperature": 0.2,
                "max_tokens": 80,
            },
            timeout=10.0,
        )
        response.raise_for_status()
        payload = response.json()
        message = _extract_content(payload)
        logger.info(
            "voice_recovery: LLM responded. status=%s message_empty=%s",
            response.status_code, not bool(message),
        )
        sanitized_message = sanitize_speech_text(message)
        if _valid_message(sanitized_message):
            return {"message": sanitized_message, "source": "llm", **base_meta}
        logger.warning(
            "voice_recovery: LLM message failed validation (empty or forbidden content) — using fallback."
        )
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "voice_recovery: LLM HTTP error status=%s — using fallback.", exc.response.status_code
        )
    except httpx.HTTPError as exc:
        logger.warning("voice_recovery: LLM network error %s — using fallback.", type(exc).__name__)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("voice_recovery: LLM response parse error %s — using fallback.", type(exc).__name__)

    return {"message": fallback, "source": "fallback", **base_meta}
