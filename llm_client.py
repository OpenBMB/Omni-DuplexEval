#!/usr/bin/env python3
"""OpenAI-compatible client used by the DuplexEval evaluation scripts."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


class OpenAILLMClient:
    """Small wrapper around an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.environ.get("DUPLEXEVAL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("DUPLEXEVAL_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
        )
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "Missing API key. Set DUPLEXEVAL_API_KEY or OPENAI_API_KEY before running evaluation."
            )
        if not self.base_url:
            raise ValueError(
                "Missing API base URL. Set DUPLEXEVAL_BASE_URL or OPENAI_BASE_URL before running evaluation."
            )

        self.base_url = self.base_url.rstrip("/")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "DuplexEval/1.0",
            "Accept": "application/json",
        }

    @staticmethod
    def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalized.append({"role": role, "content": content})
        return normalized

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": self._normalize_messages(messages),
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code != 200:
                return {"error": response.text, "status_code": response.status_code}
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": "Request timeout"}
        except Exception as exc:
            return {"error": str(exc)}


def extract_message_text(response: Dict[str, Any]) -> str:
    """Extract text from an OpenAI-compatible chat-completion response."""

    try:
        return str(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
    except Exception:
        return ""
