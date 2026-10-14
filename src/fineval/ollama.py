"""Small deterministic Ollama client with structured-output support."""

import json
import urllib.request


LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["+2", "+1", "0", "-1", "-2"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "decisive_span": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["label", "confidence", "decisive_span", "rationale"],
}


def generate_json(
    model: str,
    prompt: str,
    schema: dict = LABEL_SCHEMA,
    *,
    seed: int = 2026,
    timeout: int = 600,
    num_gpu: int | None = None,
) -> tuple[dict, dict]:
    last_error = None
    for attempt in range(2):
        options = {
            "temperature": 0,
            "seed": seed + attempt,
            "num_predict": 900,
        }
        if num_gpu is not None:
            options["num_gpu"] = num_gpu
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": schema,
            # Structured rationales occasionally exceed 500 tokens; truncating
            # a closing quote/bracket loses the entire checkpointed example.
            "options": options,
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.load(response)
        try:
            return json.loads(raw["response"]), raw
        except json.JSONDecodeError as error:
            last_error = error
    raise last_error
