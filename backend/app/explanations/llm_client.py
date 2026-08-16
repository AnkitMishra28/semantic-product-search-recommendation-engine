"""LLM Provider abstraction and clients for OpenAI / Gemini / Mock."""

from abc import ABC, abstractmethod
import json
import logging
import os
from typing import Any, Dict, Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract interface for LLM completion providers."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        """Execute chat completion and return raw response text."""
        pass


class OpenAILLMClient(BaseLLMClient):
    """Standard OpenAI-compatible chat completion client via standard HTTP."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 8.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        """Call chat completions API."""
        if not self.is_configured():
            raise ValueError("OpenAI API key is not configured or is empty.")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,  # Low temperature for strict factual consistency
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                return resp_data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            logger.error(f"OpenAI API HTTP Error {e.code}: {err_body}")
            raise RuntimeError(f"OpenAI API HTTP Error {e.code}: {err_body}") from e
        except Exception as e:
            logger.error(f"Failed to communicate with LLM provider: {str(e)}")
            raise RuntimeError(f"Failed to communicate with LLM provider: {str(e)}") from e


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for deterministic testing and offline evaluation."""

    def __init__(self, canned_response: Optional[str] = None) -> None:
        self.canned_response = canned_response

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        if self.canned_response:
            return self.canned_response

        # Generate a mock valid JSON response
        return json.dumps({
            "product_id": "B08N6PZR6Y",
            "summary": "Verified match for requested specifications based on catalog attributes.",
            "reasons": [
                {
                    "type": "constraint_match",
                    "label": "Brand",
                    "text": "Manufactured by JETech.",
                    "evidence": "JETech",
                    "is_matched": True,
                },
                {
                    "type": "constraint_match",
                    "label": "Budget",
                    "text": "Listed price is within the specified budget.",
                    "evidence": "$19.99",
                    "is_matched": True,
                },
            ],
            "semantic_match_score": 0.88,
            "grounded": True,
            "warnings": [],
            "generation_method": "llm",
        })
