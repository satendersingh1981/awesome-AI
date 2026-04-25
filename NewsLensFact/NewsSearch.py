"""Utilities for fetching topic-based news for LLM fact-checking workflows."""

from __future__ import annotations

import os
from typing import Dict, List

import requests
from dotenv import load_dotenv

NEWS_API_URL = "https://newsapi.org/v2/everything"


def fetch_latest_news_by_topic(
    topic: str,
    *,
    max_articles: int = 5,
    language: str = "en",
    sort_by: str = "publishedAt",
) -> List[Dict[str, str]]:
    """Fetch latest news articles for a topic from NewsAPI.

    Returns a list of normalized article dictionaries that are easy to pass
    to downstream LLM pipelines.
    """
    load_dotenv()
    api_key = os.getenv("NEWS_ORG_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing NEWS_ORG_API_KEY in environment. Add it to your .env file."
        )

    params = {
        "q": topic,
        "language": language,
        "sortBy": sort_by,
        "pageSize": max_articles,
        "apiKey": api_key,
    }

    response = requests.get(NEWS_API_URL, params=params, timeout=20)
    response.raise_for_status()

    payload = response.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"NewsAPI returned error: {payload}")

    normalized_articles: List[Dict[str, str]] = []
    for article in payload.get("articles", []):
        normalized_articles.append(
            {
                "title": article.get("title") or "",
                "source": (article.get("source") or {}).get("name") or "",
                "published_at": article.get("publishedAt") or "",
                "url": article.get("url") or "",
                "description": article.get("description") or "",
                "content": article.get("content") or "",
            }
        )

    return normalized_articles


def build_llm_factcheck_context(topic: str, articles: List[Dict[str, str]]) -> str:
    """Create compact text context to feed into an LLM for fake/fact checks."""
    if not articles:
        return f"No recent articles found for topic: {topic}"

    lines = [
        (
            "You are a fact-checking assistant. Use the following recent news "
            f"snippets about '{topic}' as evidence."
        ),
        "",
    ]

    for idx, article in enumerate(articles, start=1):
        lines.extend(
            [
                f"Article {idx}:",
                f"Title: {article['title']}",
                f"Source: {article['source']}",
                f"Published: {article['published_at']}",
                f"Description: {article['description']}",
                f"Content: {article['content']}",
                f"URL: {article['url']}",
                "",
            ]
        )

    return "\n".join(lines).strip()


if __name__ == "__main__":
    sample_topic = "Riots in India"
    latest_articles = fetch_latest_news_by_topic(sample_topic, max_articles=3)
    llm_context = build_llm_factcheck_context(sample_topic, latest_articles)
    print(llm_context)