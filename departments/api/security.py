"""
departments/api/security.py
────────────────────────────
Security & Password Governance Module for HIMS.
Enforces statutory password complexity rules for user accounts.
"""

import re
from typing import Tuple


MIN_PASSWORD_LENGTH = 8


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """
    Validate password against enterprise complexity policy:
    1. Minimum 8 characters in length
    2. At least one uppercase letter (A-Z)
    3. At least one lowercase letter (a-z)
    4. At least one numeric digit (0-9)
    5. At least one special symbol (!@#$%^&*()_+-=[]{}|;:,.<>?)

    Args:
        password: Raw password string to validate.

    Returns:
        Tuple of (is_valid: bool, error_or_success_message: str)
    """
    if not password or not isinstance(password, str):
        return False, "Password cannot be empty."

    if len(password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."

    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter (A-Z)."

    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter (a-z)."

    if not re.search(r"\d", password):
        return False, "Password must contain at least one numeric digit (0-9)."

    if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]", password):
        return False, "Password must contain at least one special character (!@#$%^&*...)."

    return True, "Password meets complexity requirements."
