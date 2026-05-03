"""LLM-based fact checking utilities using GPT or Ollama.

This module is designed to consume a daily news feed summary string and
evaluate whether a claim is likely fact, fake, or uncertain.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
import ollama
from openai import OpenAI

DEBUG_LOG_PATH = Path(__file__).resolve().parents[1] / "debug-6bf439.log"


def _debug_log(hypothesis_id: str, location: str, message: str, data: Dict[str, Any]) -> None:
    # region agent log
    payload = {
        "sessionId": "6bf439",
        "runId": "llmfactcheck-runtime-1",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # endregion


try:
    # Works when imported as package: NewsLensFact.LLMFactCheck
    from .NewsSearch import build_llm_factcheck_context, fetch_latest_news_by_topic
except ImportError:
    # Works when run directly: python NewsLensFact/LLMFactCheck.py
    from NewsSearch import build_llm_factcheck_context, fetch_latest_news_by_topic


def ensure_ollama_model_available(model: str) -> None:
    """Validate Ollama service reachability and model availability."""
    _debug_log("H1", "ensure_ollama_model_available:start", "checking ollama model", {"model": model})
    try:
        response = ollama.list()
    except Exception as exc:
        _debug_log("H1", "ensure_ollama_model_available:error", "ollama list failed", {"error_type": type(exc).__name__, "error": str(exc)})
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
    _debug_log("H1", "ensure_ollama_model_available:list", "ollama models discovered", {"count": len(installed_model_names), "models": installed_model_names[:20]})
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
    timeout_sec = float(os.getenv("OLLAMA_TIMEOUT_SEC", "45"))
    _debug_log(
        "H2",
        "_fact_check_with_ollama:pre_chat",
        "starting ollama chat request",
        {
            "model": model,
            "claim_len": len(claim),
            "summary_len": len(daily_feed_summary),
            "prompt_len": len(prompt),
            "timeout_sec": timeout_sec,
        },
    )
    start = time.time()
    client = ollama.Client(timeout=timeout_sec)
    try:
        response = client.chat(
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
    except Exception as exc:
        _debug_log(
            "H2",
            "_fact_check_with_ollama:error",
            "ollama chat failed",
            {
                "elapsed_sec": round(time.time() - start, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise TimeoutError(
            f"Ollama request failed or timed out after {timeout_sec}s. "
            "Increase OLLAMA_TIMEOUT_SEC or verify Ollama server/model health."
        ) from exc
    _debug_log(
        "H2",
        "_fact_check_with_ollama:post_chat",
        "ollama chat completed",
        {"elapsed_sec": round(time.time() - start, 3)},
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


def _print_sources_for_certain_verdict(
    result: Dict[str, Any], articles: list[Dict[str, str]]
) -> None:
    """Print source names and URLs when verdict is not uncertain."""
    verdict = str(result.get("verdict", "")).strip().lower()
    if verdict == "uncertain":
        return

    print("\nEvidence sources used:")
    seen_pairs: set[tuple[str, str]] = set()
    for article in articles:
        source = (article.get("source") or "Unknown source").strip()
        url = (article.get("url") or "").strip()
        key = (source, url)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if url:
            print(f"- {source}: {url}")
        else:
            print(f"- {source}: URL not available")


if __name__ == "__main__":
    print("=== FactLens Interactive Fact Check ===")
    user_topic = input("Enter topic to fetch latest news: ").strip()
    user_claim = input("Enter claim to verify: ").strip()
    provider = input("Choose provider (gpt/ollama) [ollama]: ").strip().lower() or "ollama"
    model_override = input("Optional model override (press Enter for default): ").strip() or None

    if not user_topic:
        raise ValueError("Topic cannot be empty.")
    if not user_claim:
        raise ValueError("Claim cannot be empty.")
    if provider not in {"gpt", "ollama"}:
        raise ValueError("Invalid provider. Use 'gpt' or 'ollama'.")

    _debug_log("H3", "__main__:start", "script execution started", {"topic": user_topic, "provider": provider})
    try:
        sample_articles = fetch_latest_news_by_topic(user_topic, max_articles=5)
        _debug_log("H3", "__main__:articles", "news fetched", {"article_count": len(sample_articles)})
        sample_daily_summary = build_llm_factcheck_context(user_topic, sample_articles)
        _debug_log("H3", "__main__:summary", "summary built", {"summary_len": len(sample_daily_summary)})

        result = fact_check_from_daily_summary(
            claim=user_claim,
            daily_feed_summary=sample_daily_summary,
            provider=provider,
            model=model_override,
        )
        _debug_log("H4", "__main__:result", "fact check completed", {"keys": list(result.keys())})
        print(json.dumps(result, indent=2))
        _print_sources_for_certain_verdict(result, sample_articles)
    except Exception as exc:
        _debug_log("H4", "__main__:error", "script failed", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
