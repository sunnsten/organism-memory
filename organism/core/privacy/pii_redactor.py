from __future__ import annotations

import re
from typing import List, Protocol, Tuple, runtime_checkable


@runtime_checkable
class PIIRedactor(Protocol):
    """Protocol for PII detection and redaction."""

    def redact(self, text: str) -> str:
        """
        Redact PII from text.

        Args:
            text: Input text that may contain PII.

        Returns:
            Text with PII replaced by placeholder tokens.
        """
        ...

    def detect(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Detect PII spans in text without redacting.

        Args:
            text: Input text.

        Returns:
            List of (pii_type, matched_text, start, end) tuples.
        """
        ...


class RegexPIIRedactor:
    """
    Regex-based PII redactor.

    Detects and masks:
    - Email addresses
    - Phone numbers (international and local formats)
    - Credit card numbers
    - Social security numbers
    - IP addresses

    This is a baseline implementation. For better accuracy,
    swap with a NER-based redactor (e.g. Presidio, spaCy NER).
    """

    # Patterns: (pii_type, regex_pattern, replacement)
    # Order matters: more specific patterns MUST come before greedy ones
    # (credit_card and SSN before phone, IP before phone)
    _PATTERNS: List[Tuple[str, re.Pattern, str]] = [
        (
            "email",
            re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
            ),
            "[EMAIL]",
        ),
        (
            "credit_card",
            re.compile(
                r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
            ),
            "[CREDIT_CARD]",
        ),
        (
            "ssn",
            re.compile(
                r"\b\d{3}-\d{2}-\d{4}\b"
            ),
            "[SSN]",
        ),
        (
            "ip_address",
            re.compile(
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
            ),
            "[IP]",
        ),
        (
            "phone",
            re.compile(
                r"(?<!\d)"  # negative lookbehind: not preceded by digit
                r"(?:\+?\d{1,3}[-.\s]?)?"
                r"(?:\(?\d{2,4}\)?[-.\s]?)?"
                r"\d{3,4}[-.\s]?\d{2,4}"
                r"(?!\d)"  # negative lookahead: not followed by digit
            ),
            "[PHONE]",
        ),
    ]

    def redact(self, text: str) -> str:
        """Redact all detected PII with placeholder tokens."""
        result = text
        for pii_type, pattern, replacement in self._PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def detect(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Detect PII spans without redacting.

        Returns:
            List of (pii_type, matched_text, start, end).
        """
        findings: List[Tuple[str, str, int, int]] = []
        for pii_type, pattern, _ in self._PATTERNS:
            for match in pattern.finditer(text):
                findings.append((
                    pii_type,
                    match.group(),
                    match.start(),
                    match.end(),
                ))
        findings.sort(key=lambda x: x[2])
        return findings


__all__ = ["PIIRedactor", "RegexPIIRedactor"]
