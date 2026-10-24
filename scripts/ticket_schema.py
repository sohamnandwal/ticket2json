import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3.1:latest"

TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "issue_category": {
            "type": "string",
            "enum": ["hardware", "software", "network"],
        },
        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
        "sentiment": {
            "type": "string",
            "enum": ["angry", "neutral", "panicked"],
        },
        "extracted_entities": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "issue_category",
        "urgency",
        "sentiment",
        "extracted_entities",
    ],
    "additionalProperties": False,
}

TICKET_INSTRUCTIONS = (
    "Classify the IT support ticket and extract its concrete entities. "
    "Choose the closest issue category from hardware, software, or network; "
    "urgency from low, medium, or high; and sentiment from angry, neutral, "
    "or panicked. Use neutral when the wording does not clearly show another "
    "sentiment. Return only the requested structured result."
)


class OllamaError(RuntimeError):
    pass


def validate_ticket(value):
    if not isinstance(value, dict) or set(value) != set(TICKET_SCHEMA["required"]):
        raise ValueError("Output must contain exactly the four required fields.")
    if not isinstance(value["issue_category"], str) or value["issue_category"] not in {
        "hardware",
        "software",
        "network",
    }:
        raise ValueError("issue_category must be hardware, software, or network.")
    if not isinstance(value["urgency"], str) or value["urgency"] not in {
        "low",
        "medium",
        "high",
    }:
        raise ValueError("urgency must be low, medium, or high.")
    if not isinstance(value["sentiment"], str) or value["sentiment"] not in {
        "angry",
        "neutral",
        "panicked",
    }:
        raise ValueError("sentiment must be angry, neutral, or panicked.")
    entities = value["extracted_entities"]
    if not isinstance(entities, list) or any(
        not isinstance(entity, str) for entity in entities
    ):
        raise ValueError("extracted_entities must be an array of strings.")
    return value


def call_ollama_json(
    prompt,
    schema,
    model=DEFAULT_MODEL,
    host=OLLAMA_HOST,
    timeout=180,
    temperature=0,
):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {"temperature": temperature},
    }
    request = Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except json.JSONDecodeError as error:
        raise OllamaError(f"Ollama returned invalid API JSON: {error}") from error
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise OllamaError(f"Ollama returned HTTP {error.code}: {details}") from error
    except URLError as error:
        raise OllamaError(
            f"Cannot reach Ollama at {host}: {error.reason}. "
            "Start Ollama and confirm the model is installed."
        ) from error
    except TimeoutError as error:
        raise OllamaError(f"Ollama request timed out after {timeout} seconds.") from error

    if not isinstance(result, dict):
        raise OllamaError("Ollama returned an invalid API response.")
    response_text = result.get("response")
    if not isinstance(response_text, str):
        raise OllamaError("Ollama response did not contain generated text.")
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise OllamaError(f"Ollama returned invalid JSON: {error}") from error
    return parsed


def call_ollama(prompt, model=DEFAULT_MODEL, host=OLLAMA_HOST, timeout=180):
    try:
        return validate_ticket(
            call_ollama_json(
                prompt,
                TICKET_SCHEMA,
                model=model,
                host=host,
                timeout=timeout,
            )
        )
    except (ValueError, TypeError) as error:
        raise OllamaError(f"Ollama output did not match the ticket schema: {error}") from error


def build_prompt(ticket_text):
    return f"{TICKET_INSTRUCTIONS}\n\nIT support ticket:\n{ticket_text}"
