"""External service integrations for Jira and source repositories."""

from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests

from .preprocessing import ocr_image_bytes, reduce_log_lines


MAX_ATTACHMENT_CHARS = 40_000
DEFAULT_CONTEXT_RADIUS = 25


@dataclass
class AttachmentContent:
    filename: str
    mime_type: str = "application/octet-stream"
    text_content: str = ""
    content_base64: str = ""
    is_image: bool = False


@dataclass
class JiraTicket:
    key: str
    summary: str = ""
    description: str = ""
    comments: list[str] = field(default_factory=list)
    attachments: list[AttachmentContent] = field(default_factory=list)


@dataclass
class RepositorySnippet:
    provider: str
    repository: str
    ref: str
    path: str
    start_line: int
    end_line: int
    content: str


class JiraClient:
    def __init__(
        self,
        base_url: str,
        email: str = "",
        api_token: str = "",
        bearer_token: str = "",
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.api_token = api_token
        self.bearer_token = bearer_token
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "JiraClient":
        base_url = os.getenv("JIRA_BASE_URL", "")
        if not base_url:
            raise ValueError("Set JIRA_BASE_URL before fetching Jira tickets.")
        return cls(
            base_url=base_url,
            email=os.getenv("JIRA_EMAIL", ""),
            api_token=os.getenv("JIRA_API_TOKEN", ""),
            bearer_token=os.getenv("JIRA_BEARER_TOKEN", ""),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def _auth(self):
        if self.email and self.api_token and not self.bearer_token:
            return (self.email, self.api_token)
        return None

    def fetch_ticket(self, issue_key: str) -> JiraTicket:
        url = f"{self.base_url}/rest/api/3/issue/{quote(issue_key)}"
        params = {"fields": "summary,description,comment,attachment"}
        response = requests.get(
            url,
            headers=self._headers(),
            auth=self._auth(),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        fields = payload.get("fields", {})

        comments = []
        for comment in fields.get("comment", {}).get("comments", []):
            body_text = extract_jira_document_text(comment.get("body"))
            if body_text:
                author = comment.get("author", {}).get("displayName", "Unknown")
                comments.append(f"{author}: {body_text}")

        attachments = [
            self._download_attachment(attachment)
            for attachment in fields.get("attachment", [])
            if attachment.get("content")
        ]
        attachments = [attachment for attachment in attachments if attachment is not None]

        return JiraTicket(
            key=payload.get("key", issue_key),
            summary=fields.get("summary", ""),
            description=extract_jira_document_text(fields.get("description")),
            comments=comments,
            attachments=attachments,
        )

    def _download_attachment(self, attachment: dict[str, Any]) -> AttachmentContent | None:
        filename = attachment.get("filename", "attachment")
        mime_type = attachment.get("mimeType") or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        response = requests.get(
            attachment["content"],
            headers=self._headers(),
            auth=self._auth(),
            timeout=self.timeout,
        )
        response.raise_for_status()

        if mime_type.startswith("image/"):
            ocr_result = ocr_image_bytes(response.content)
            if ocr_result.usable:
                return AttachmentContent(
                    filename=f"{filename} (OCR text)",
                    mime_type="text/plain",
                    text_content=f"[OCR confidence={ocr_result.confidence}]\n{ocr_result.text}",
                )
            return AttachmentContent(
                filename=filename,
                mime_type=mime_type,
                content_base64=base64.b64encode(response.content).decode("ascii"),
                text_content=f"OCR fallback reason: {ocr_result.reason}",
                is_image=True,
            )

        if mime_type.startswith("text/") or filename.lower().endswith(
            (".log", ".txt", ".json", ".xml", ".yaml", ".yml", ".csv", ".md")
        ):
            text = reduce_log_lines(response.content.decode("utf-8", errors="replace"))
            return AttachmentContent(
                filename=filename,
                mime_type=mime_type,
                text_content=text[:MAX_ATTACHMENT_CHARS],
            )

        return AttachmentContent(
            filename=filename,
            mime_type=mime_type,
            text_content=f"Attachment downloaded but not text/image readable by this tool: {filename}",
        )


def extract_jira_document_text(node: Any) -> str:
    """Extract readable text from Jira Cloud ADF, plain strings, or nested JSON."""

    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "\n".join(filter(None, (extract_jira_document_text(item) for item in node)))
    if not isinstance(node, dict):
        return str(node)

    parts: list[str] = []
    if "text" in node:
        parts.append(str(node["text"]))
    for child in node.get("content", []):
        child_text = extract_jira_document_text(child)
        if child_text:
            parts.append(child_text)
    return " ".join(parts).strip()


class GitHubRepositoryClient:
    provider = "github"

    def __init__(self, token: str = "", api_url: str = "https://api.github.com", timeout: int = 30):
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "GitHubRepositoryClient":
        return cls(
            token=os.getenv("GITHUB_TOKEN", ""),
            api_url=os.getenv("GITHUB_API_URL", "https://api.github.com"),
        )

    def fetch_file_snippet(
        self,
        repository: str,
        path: str,
        ref: str = "main",
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> RepositorySnippet:
        url = f"{self.api_url}/repos/{repository}/contents/{quote(path, safe='/')}"
        headers = {"Accept": "application/vnd.github.raw"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = requests.get(url, headers=headers, params={"ref": ref}, timeout=self.timeout)
        response.raise_for_status()
        return _to_snippet(self.provider, repository, ref, path, response.text, start_line, end_line)


class GitLabRepositoryClient:
    provider = "gitlab"

    def __init__(self, token: str = "", api_url: str = "https://gitlab.com/api/v4", timeout: int = 30):
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "GitLabRepositoryClient":
        return cls(
            token=os.getenv("GITLAB_TOKEN", ""),
            api_url=os.getenv("GITLAB_API_URL", "https://gitlab.com/api/v4"),
        )

    def fetch_file_snippet(
        self,
        repository: str,
        path: str,
        ref: str = "main",
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> RepositorySnippet:
        project = quote(repository, safe="")
        file_path = quote(path, safe="")
        url = f"{self.api_url}/projects/{project}/repository/files/{file_path}/raw"
        headers = {}
        if self.token:
            headers["PRIVATE-TOKEN"] = self.token
        response = requests.get(url, headers=headers, params={"ref": ref}, timeout=self.timeout)
        response.raise_for_status()
        return _to_snippet(self.provider, repository, ref, path, response.text, start_line, end_line)


def _to_snippet(
    provider: str,
    repository: str,
    ref: str,
    path: str,
    content: str,
    start_line: int | None,
    end_line: int | None,
) -> RepositorySnippet:
    lines = content.splitlines()
    total = len(lines)

    if start_line is None and end_line is None:
        start = 1
        end = min(total, 200)
    else:
        focus_start = max(1, start_line or end_line or 1)
        focus_end = max(focus_start, end_line or focus_start)
        start = max(1, focus_start - DEFAULT_CONTEXT_RADIUS)
        end = min(total, focus_end + DEFAULT_CONTEXT_RADIUS)

    numbered = [f"{line_number}: {lines[line_number - 1]}" for line_number in range(start, end + 1)]
    return RepositorySnippet(
        provider=provider,
        repository=repository,
        ref=ref,
        path=path,
        start_line=start,
        end_line=end,
        content="\n".join(numbered),
    )
