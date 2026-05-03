"""Pre-processing helpers that reduce LLM token cost before analysis."""

from __future__ import annotations  # Keep modern type hints available.

import os  # Allow log cost limits to be tuned from environment variables.
from dataclasses import dataclass  # Use a small typed OCR result container.
from io import BytesIO  # Open image bytes without writing temporary files.
from pathlib import Path  # Read local screenshot files when uploaded.
from statistics import mean  # Estimate OCR confidence when available.


DEFAULT_MAX_LOG_LINES = 400  # Keep enough evidence while avoiding thousands of log lines.
DEFAULT_LOG_CONTEXT_RADIUS = 25  # Include nearby lines before and after each detected error.
MIN_OCR_TEXT_CHARS = 30  # Treat very tiny OCR output as unreliable.
MIN_OCR_CONFIDENCE = 45.0  # Tesseract confidence threshold for clear enough screenshots.
ERROR_KEYWORDS = (  # Keywords that usually indicate high-signal log regions.
    "error",
    "exception",
    "failed",
    "failure",
    "fatal",
    "critical",
    "traceback",
    "stacktrace",
    "stack trace",
    "caused by",
    "panic",
    "abort",
    "timeout",
    "denied",
    "unhandled",
)


@dataclass
class OCRResult:
    """Result of trying local OCR before sending screenshots to the LLM."""

    text: str = ""  # Text extracted from the screenshot.
    confidence: float | None = None  # Average OCR confidence when Tesseract provides it.
    usable: bool = False  # True when OCR text is clear enough to use instead of image input.
    reason: str = ""  # Short explanation for debugging/fallback behavior.


def reduce_log_lines(
    log_text: str,
    max_lines: int | None = None,
    context_radius: int | None = None,
) -> str:
    """Keep only relevant windows from very large logs."""

    if not log_text:  # Empty logs need no processing.
        return ""  # Return an empty string unchanged.

    max_lines = max_lines or _env_int("MAX_LOG_LINES_FOR_LLM", DEFAULT_MAX_LOG_LINES)  # Let .env tune line cap.
    context_radius = context_radius or _env_int("LOG_CONTEXT_RADIUS", DEFAULT_LOG_CONTEXT_RADIUS)  # Tune context window.
    lines = log_text.splitlines()  # Split once so line counting and slicing are cheap.
    if len(lines) <= max_lines:  # Short logs are already cost-safe.
        return log_text  # Preserve the original evidence exactly.

    lowered_lines = [line.lower() for line in lines]  # Normalize log lines for keyword matching.
    focus_indexes = [  # Locate lines that look like failures or stack traces.
        index  # Keep the original zero-based line index.
        for index, line in enumerate(lowered_lines)  # Inspect every log line.
        if any(keyword in line for keyword in ERROR_KEYWORDS)  # Match known error terms.
    ]

    if not focus_indexes:  # If no obvious error exists, sample the beginning and end.
        head_count = max_lines // 2  # Keep the startup/config context.
        tail_count = max_lines - head_count  # Keep the final failure/end context.
        kept = lines[:head_count] + ["... log truncated: middle lines removed ..."] + lines[-tail_count:]  # Compact log.
        return _log_truncation_header(len(lines), len(kept), "head/tail sampling") + "\n".join(kept)  # Add metadata.

    selected_indexes: set[int] = set()  # Store deduplicated line indexes to keep.
    for focus_index in focus_indexes:  # Build a context window around each error-like line.
        window_start = max(0, focus_index - context_radius)  # Avoid negative indexes.
        window_end = min(len(lines), focus_index + context_radius + 1)  # Avoid going past the log.
        selected_indexes.update(range(window_start, window_end))  # Add surrounding evidence lines.
        if len(selected_indexes) >= max_lines:  # Stop once we have enough signal.
            break  # Prevent huge prompts when the log has many repeated errors.

    ordered_indexes = sorted(selected_indexes)[:max_lines]  # Keep stable line order and enforce the cap.
    compacted_lines = [f"{index + 1}: {lines[index]}" for index in ordered_indexes]  # Preserve original line numbers.
    return _log_truncation_header(len(lines), len(compacted_lines), "error-window extraction") + "\n".join(compacted_lines)


def _log_truncation_header(total_lines: int, kept_lines: int, method: str) -> str:
    """Describe how a large log was reduced before the LLM call."""

    return (  # Include enough metadata for the LLM and user to understand missing lines.
        f"[Log preprocessing: original_lines={total_lines}, "
        f"kept_lines={kept_lines}, method={method}]\n"
    )


def _env_int(name: str, default: int) -> int:
    """Read a positive integer environment setting with a safe fallback."""

    try:  # Environment variables arrive as strings and may be invalid.
        value = int(os.getenv(name, str(default)))  # Convert configured value to an integer.
        return value if value > 0 else default  # Reject zero and negative caps.
    except ValueError:  # Invalid values should not break incident analysis.
        return default  # Fall back to the safe default.


def ocr_image_file(path: str | Path) -> OCRResult:
    """Run free local OCR against an uploaded screenshot file."""

    image_bytes = Path(path).read_bytes()  # Read the screenshot bytes from disk.
    return ocr_image_bytes(image_bytes)  # Reuse the byte-based OCR implementation.


def ocr_image_bytes(image_bytes: bytes) -> OCRResult:
    """Run local Tesseract OCR and report whether the text is clear enough."""

    try:  # Keep OCR optional so the app works even without local OCR dependencies.
        from PIL import Image  # type: ignore  # Pillow opens common image formats.
        import pytesseract  # type: ignore  # Python wrapper for the free Tesseract OCR engine.
        from pytesseract import Output  # type: ignore  # Structured OCR output with confidence values.
    except ImportError as exc:  # Missing Python OCR packages should fall back to image input.
        return OCRResult(reason=f"OCR package missing: {exc}")  # Explain why OCR was skipped.

    try:  # OCR can fail on damaged images or when Tesseract is not installed.
        image = Image.open(BytesIO(image_bytes))  # Load image from memory.
        data = pytesseract.image_to_data(image, output_type=Output.DICT)  # Extract words and confidence scores.
        words = [word.strip() for word in data.get("text", []) if word and word.strip()]  # Keep real OCR words.
        confidences = [float(value) for value in data.get("conf", []) if _is_valid_confidence(value)]  # Clean scores.
        text = " ".join(words).strip()  # Convert OCR words into readable text.
        confidence = mean(confidences) if confidences else None  # Average confidence if scores exist.
    except Exception as exc:  # Tesseract binary errors and image decode errors land here.
        return OCRResult(reason=f"OCR failed: {exc}")  # Tell caller to use the image fallback.

    usable_by_length = len("".join(text.split())) >= MIN_OCR_TEXT_CHARS  # Require meaningful text volume.
    usable_by_confidence = confidence is None or confidence >= MIN_OCR_CONFIDENCE  # Require clear enough OCR.
    usable = bool(text and usable_by_length and usable_by_confidence)  # Final decision for text-only usage.
    reason = "usable OCR text" if usable else "OCR text was too short or low confidence"  # Explain decision.
    return OCRResult(text=text, confidence=confidence, usable=usable, reason=reason)  # Return OCR details.


def _is_valid_confidence(value) -> bool:
    """Return True when a Tesseract confidence value is numeric and non-negative."""

    try:  # Tesseract can return strings such as "-1" for non-word regions.
        return float(value) >= 0  # Keep only real word confidence values.
    except (TypeError, ValueError):  # Ignore malformed confidence entries.
        return False  # Invalid values do not contribute to the average.
