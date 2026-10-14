import json

from fineval.openai_compatible import chat_payload, generate_json, parse_json_content


SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}


def test_parse_json_content_accepts_plain_and_fenced_objects():
    assert parse_json_content('{"label": "+1"}') == {"label": "+1"}
    assert parse_json_content('```json\n{"label": "0"}\n```') == {"label": "0"}


def test_chat_payload_freezes_schema_seed_and_disables_thinking():
    payload = chat_payload(
        "teacher", "prompt", SCHEMA, max_tokens=321, disable_thinking=True,
    )
    assert payload["temperature"] == 0
    assert payload["seed"] == 2026
    assert payload["max_tokens"] == 321
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["json_schema"]["schema"] == SCHEMA


def test_generate_json_calls_versioned_chat_endpoint(monkeypatch):
    observed = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": json.dumps({"label": "-1"})}}],
                "usage": {"completion_tokens": 4},
            }

    def fake_post(url, **kwargs):
        observed.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("fineval.openai_compatible.requests.post", fake_post)
    judgment, raw = generate_json(
        "http://127.0.0.1:8000/v1/", "secret", "teacher", "prompt", SCHEMA,
        timeout=12,
    )
    assert judgment == {"label": "-1"}
    assert raw["usage"]["completion_tokens"] == 4
    assert observed["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert observed["headers"]["Authorization"] == "Bearer secret"
    assert observed["timeout"] == 12
