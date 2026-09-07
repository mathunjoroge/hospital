"""
departments/api/ai_audit.py
────────────────────────────
AI Governance & Audit Logging Module for HIMS.

Every AI/NLP inference call (LLM, RDKit, FHIR AI, etc.) should go through this
module to ensure:
1. Full audit trail of who triggered what AI capability, when, and with what input.
2. Clear disclosure in API responses when offline/fallback mode is active.
3. Input validation and length limits to prevent prompt injection / abuse.
4. Structured log format suitable for SIEM / compliance export.

Usage:
    from departments.api.ai_audit import log_ai_call, validate_ai_input, AIMode

    # At the start of an AI handler:
    clean_input = validate_ai_input(raw_text)

    # After inference:
    log_ai_call(
        feature='clinical_chatbot',
        mode=AIMode.OFFLINE_FALLBACK,
        user_id=current_user.id,
        input_summary=clean_input[:120],
        output_summary=response[:120],
        latency_ms=elapsed
    )
"""

import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from flask_login import current_user

logger = logging.getLogger("HIMS.AIAudit")

# ────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────

MAX_INPUT_LENGTH = 4000  # Characters — reject inputs longer than this
MAX_INPUT_WORDS = 600  # Word-level guard against verbose prompt injection
AUDIT_TRUNCATE_LENGTH = 200  # Max chars stored in audit log per field


class AIMode(str, Enum):
    """Enum to distinguish live LLM calls from offline fallback, RDKit, rules, etc."""

    LIVE_LLM = "live_llm"
    OFFLINE_FALLBACK = "offline_fallback"
    RULE_BASED = "rule_based"
    RDKIT = "rdkit"
    FHIR_TRANSFORM = "fhir_transform"
    DHIS2_AGGREGATE = "dhis2_aggregate"


# ────────────────────────────────────────────────────────
# Input Validation
# ────────────────────────────────────────────────────────


class AIInputValidationError(ValueError):
    """Raised when AI input fails validation checks."""

    pass


def validate_ai_input(text: str, feature: str = "unspecified") -> str:
    """
    Validate and sanitize user input before passing to any AI model.

    Enforces:
    - Non-empty input
    - Maximum character length (MAX_INPUT_LENGTH)
    - Maximum word count (MAX_INPUT_WORDS)
    - Strips leading/trailing whitespace

    Args:
        text: Raw user input string.
        feature: AI feature name for error context.

    Returns:
        Sanitized input string.

    Raises:
        AIInputValidationError: If input fails any validation rule.
    """
    if not text or not isinstance(text, str):
        raise AIInputValidationError(f"[{feature}] Input must be a non-empty string.")

    clean = text.strip()

    if len(clean) == 0:
        raise AIInputValidationError(f"[{feature}] Input cannot be blank.")

    if len(clean) > MAX_INPUT_LENGTH:
        raise AIInputValidationError(
            f"[{feature}] Input exceeds maximum length of {MAX_INPUT_LENGTH} characters "
            f"(received {len(clean)}). Please shorten your query."
        )

    word_count = len(clean.split())
    if word_count > MAX_INPUT_WORDS:
        raise AIInputValidationError(
            f"[{feature}] Input exceeds {MAX_INPUT_WORDS} words (received {word_count}). "
            f"Please provide a more concise query."
        )

    return clean


# ────────────────────────────────────────────────────────
# Audit Logging
# ────────────────────────────────────────────────────────


def log_ai_call(
    feature: str,
    mode: AIMode,
    input_summary: str,
    output_summary: str,
    user_id: Optional[int] = None,
    latency_ms: Optional[float] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> None:
    """
    Write a structured AI audit log entry.

    All AI inference calls in HIMS must call this function after completing
    (or failing) inference. The log entry is written to the HIMS.AIAudit logger
    in a structured format suitable for parsing by log aggregators (ELK, Splunk, etc.).

    Args:
        feature: Name of the AI feature/capability (e.g. 'clinical_chatbot', 'cancer_risk').
        mode: AIMode enum indicating live LLM, offline fallback, RDKit, etc.
        input_summary: Truncated, sanitized summary of the input (max AUDIT_TRUNCATE_LENGTH chars).
        output_summary: Truncated summary of the output or response.
        user_id: ID of the authenticated user who triggered the call. Auto-resolved from
                 current_user if not provided and in a Flask request context.
        latency_ms: Wall-clock inference time in milliseconds.
        error: Error message if the call failed (do NOT include stack traces or raw exceptions).
        metadata: Optional additional structured metadata (e.g. molecule SMILES, patient context flags).
    """
    # Auto-resolve user_id from Flask request context if not provided
    if user_id is None:
        try:
            user_id = (
                current_user.id
                if current_user and current_user.is_authenticated
                else None
            )
        except RuntimeError:
            user_id = None  # Not in request context

    # Truncate fields for audit log
    input_truncated = (input_summary or "")[:AUDIT_TRUNCATE_LENGTH]
    output_truncated = (output_summary or "")[:AUDIT_TRUNCATE_LENGTH]

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feature": feature,
        "mode": mode.value if isinstance(mode, AIMode) else str(mode),
        "user_id": user_id,
        "input_summary": input_truncated,
        "output_summary": output_truncated,
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "error": error,
        "metadata": metadata or {},
    }

    if error:
        logger.warning(
            "AI_AUDIT | %(timestamp)s | feature=%(feature)s | mode=%(mode)s | "
            "user=%(user_id)s | latency=%(latency_ms)sms | ERROR: %(error)s | "
            "input=%(input_summary)r",
            entry,
        )
    else:
        logger.info(
            "AI_AUDIT | %(timestamp)s | feature=%(feature)s | mode=%(mode)s | "
            "user=%(user_id)s | latency=%(latency_ms)sms | "
            "input=%(input_summary)r | output=%(output_summary)r",
            entry,
        )


# ────────────────────────────────────────────────────────
# Response Disclosure Helpers
# ────────────────────────────────────────────────────────


def add_disclosure(response_dict: dict, mode: AIMode, feature: str) -> dict:
    """
    Attach a standardized disclosure block to any AI API response dict.

    This ensures the UI and downstream consumers always know:
    - Whether the response came from a live LLM, offline fallback, or deterministic engine.
    - The feature that generated it.
    - A human-readable disclaimer appropriate to the mode.

    Args:
        response_dict: The response dictionary to augment.
        mode: The AIMode used to generate this response.
        feature: The AI feature name.

    Returns:
        The augmented response_dict with an '_ai_disclosure' key added.
    """
    disclaimers = {
        AIMode.LIVE_LLM: "Response generated by a live large language model. Clinical outputs must be reviewed by a qualified healthcare professional before use.",
        AIMode.OFFLINE_FALLBACK: "⚠️ Offline mode: Response generated by rule-based fallback (no AI backend configured). Output may be generic. Configure NVIDIA_API_KEY or GEMINI_API_KEY for AI-powered responses.",
        AIMode.RULE_BASED: "Response generated by a deterministic rule-based system. No LLM or probabilistic model was used.",
        AIMode.RDKIT: "Molecular properties computed deterministically by RDKit cheminformatics engine. Results are reproducible and not AI-estimated.",
        AIMode.FHIR_TRANSFORM: "Resource generated by deterministic HL7 FHIR R4 transformation from HIMS database records.",
        AIMode.DHIS2_AGGREGATE: "Data generated by deterministic monthly aggregate queries against HIMS database. No AI or estimation involved.",
    }

    response_dict["_ai_disclosure"] = {
        "feature": feature,
        "mode": mode.value if isinstance(mode, AIMode) else str(mode),
        "disclaimer": disclaimers.get(
            mode, "AI-generated output. Use with clinical judgment."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return response_dict


# ────────────────────────────────────────────────────────
# Timing Context Manager
# ────────────────────────────────────────────────────────


class AITimer:
    """
    Context manager to measure AI inference latency in milliseconds.

    Usage:
        with AITimer() as t:
            result = model.predict(input)
        latency_ms = t.elapsed_ms
    """

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    @property
    def elapsed_ms(self) -> float:
        return getattr(self, "_elapsed_ms", 0.0)

    @elapsed_ms.setter
    def elapsed_ms(self, value: float):
        self._elapsed_ms = value
