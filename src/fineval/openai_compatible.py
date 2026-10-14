"""Small OpenAI-compatible client for deterministic structured judgments."""

from __future__ import annotations

import json
from typing import Any, Mapping

import requests


def parse_json_content(content: str) -> dict[str, Any]:
    """Parse plain or fenced JSON returned by a chat-completions server."""
    value = str(content or "").strip()
    if value.startswith("```"):
        first_newline = value.find("\n")
        if first_newline < 0:
            raise ValueError("unterminated fenced JSON response")
        value = value[first_newline + 1:]
        if value.rstrip().endswith("```"):
            value = value.rstrip()[:-3].rstrip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("structured response must be a JSON object")
    return parsed


def chat_payload(
    model: str,
    prompt: str,
    schema: Mapping[str, Any],
    *,
    max_tokens: int = 900,
    seed: int = 2026,
    disable_thinking: bool = False,
) -> dict[str, Any]:
    """Build the frozen zero-temperature JSON-schema request."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "icr_structured_judgment",
                "strict": True,
                "schema": dict(schema),
            },
        },
    }
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def generate_json(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    schema: Mapping[str, Any],
    *,
    timeout: int = 600,
    max_tokens: int = 900,
    seed: int = 2026,
    disable_thinking: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Request one auditable judgment from an OpenAI-compatible endpoint."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers=headers,
        json=chat_payload(
            model,
            prompt,
            schema,
            max_tokens=max_tokens,
            seed=seed,
            disable_thinking=disable_thinking,
        ),
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.json()
    content = raw["choices"][0]["message"]["content"]
    return parse_json_content(content), raw
