"""LLM-based fact checking utilities using GPT or Ollama.

This module is designed to consume a daily news feed summary string and
evaluate whether a claim is likely fact, fake, or uncertain.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
import ollama
from openai import OpenAI

try:
    # Works when imported as package: NewsSummarizer.LLMFactCheck
    from .NewsSearch import build_llm_factcheck_context, fetch_latest_news_by_topic
except ImportError:
    # Works when run directly: python NewsSummarizer/LLMFactCheck.py
    from NewsSearch import build_llm_factcheck_context, fetch_latest_news_by_topic


def ensure_ollama_model_available(model: str) -> None:
    """Validate Ollama service reachability and model availability."""
    try:
        response = ollama.list()
    except Exception as exc:
        raise ConnectionError(
            "Could not connect to Ollama. Ensure Ollama is running locally."
        ) from exc

    # ollama.list() can return either a dict-like response or a typed object.
    if isinstance(response, dict):
        models = response.get("models", [])
    else:
        models = getattr(response, "models", [])

    installed_model_names = []
    for item in models:
        if isinstance(item, dict):
            installed_model_names.append(item.get("model") or item.get("name") or "")
        else:
            installed_model_names.append(
                getattr(item, "model", "") or getattr(item, "name", "")
            )

    installed_model_names = [name for name in installed_model_names if name]
    if model not in installed_model_names:
        # Some clients return only canonical "name" field; accept both exact and
        # tag-normalized forms (e.g., "phi4-mini" vs "phi4-mini:latest").
        normalized_target = model if ":" in model else f"{model}:latest"
        normalized_installed = {
            name if ":" in name else f"{name}:latest" for name in installed_model_names
        }
        if normalized_target in normalized_installed:
            return

        available = ", ".join(installed_model_names) if installed_model_names else "none"
        raise ValueError(
            f"Ollama model '{model}' not found. Available models: {available}. "
            f"Run `ollama pull {model}` to install it."
        )


def _build_factcheck_prompt(claim: str, daily_feed_summary: str) -> str:
    """Build a structured prompt for fake-vs-fact evaluation."""
    return f"""
You are a strict fact-checking assistant.
Use ONLY the evidence in DAILY_FEED_SUMMARY to evaluate the CLAIM.
If the summary does not provide enough evidence, return "uncertain".

Return your answer as valid JSON with this exact structure:
{{
  "verdict": "fact | fake | uncertain",
  "confidence": 0-100,
  "reasoning": "short explanation",
  "evidence_points": ["point 1", "point 2"]
}}

CLAIM:
{claim}

DAILY_FEED_SUMMARY:
{daily_feed_summary}
""".strip()


def _safe_json_parse(text: str) -> Dict[str, Any]:
    """Parse model JSON output with graceful fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "verdict": "uncertain",
            "confidence": 0,
            "reasoning": (
                "Model output was not valid JSON. Raw output is returned in "
                "evidence_points."
            ),
            "evidence_points": [text],
        }


def _fact_check_with_gpt(
    claim: str,
    daily_feed_summary: str,
    model: str = "gpt-5-nano",
) -> Dict[str, Any]:
    """Run fact-checking with OpenAI GPT models."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY in environment. Add it to your .env file."
        )

    client = OpenAI(api_key=api_key)
    prompt = _build_factcheck_prompt(claim, daily_feed_summary)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a reliable fact-checking assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    return _safe_json_parse(content)


def _fact_check_with_ollama(
    claim: str,
    daily_feed_summary: str,
    model: str = "phi4-mini:latest",
) -> Dict[str, Any]:
    """Run fact-checking with local Ollama models via ollama library."""
    ensure_ollama_model_available(model)
    prompt = _build_factcheck_prompt(claim, daily_feed_summary)

    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a reliable fact-checking assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0},
        format="json",
    )
    model_output = response["message"]["content"]

    return _safe_json_parse(model_output)


def fact_check_from_daily_summary(
    claim: str,
    daily_feed_summary: str,
    *,
    provider: str = "gpt",
    model: str | None = None,
) -> Dict[str, Any]:
    """Fact-check a claim using a selected LLM provider.

    Args:
        claim: Statement to verify.
        daily_feed_summary: The daily summary/context text from your news feed.
        provider: "gpt" or "ollama".
        model: Optional model override for the selected provider.

    Returns:
        Dictionary with keys:
        - verdict: "fact", "fake", or "uncertain"
        - confidence: 0-100
        - reasoning: brief explanation
        - evidence_points: list of evidence bullets
    """
    provider_normalized = provider.strip().lower()

    if provider_normalized == "gpt":
        return _fact_check_with_gpt(
            claim=claim,
            daily_feed_summary=daily_feed_summary,
            model=model or "gpt-4o-mini",
        )

    if provider_normalized == "ollama":
        return _fact_check_with_ollama(
            claim=claim,
            daily_feed_summary=daily_feed_summary,
            model=model or "phi4-mini:latest",
        )

    raise ValueError("Unsupported provider. Use 'gpt' or 'ollama'.")


if __name__ == "__main__":
    sample_topic = "West Bengal election poll BJP"
    sample_claim = "The Bengal election poll winner is BJP."
    sample_articles = fetch_latest_news_by_topic(sample_topic, max_articles=5)
    sample_daily_summary = build_llm_factcheck_context(sample_topic, sample_articles)

    # Switch provider between "gpt" and "ollama"
    result = fact_check_from_daily_summary(
        claim=sample_claim,
        daily_feed_summary=sample_daily_summary,
        provider="ollama",
    )
    print(json.dumps(result, indent=2))
