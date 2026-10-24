import argparse
import json
import sys
from pathlib import Path

from ticket_schema import (
    DEFAULT_MODEL,
    OllamaError,
    build_prompt,
    call_ollama,
    call_ollama_json,
)

ROOT = Path(__file__).resolve().parent.parent


def read_tickets(path):
    tickets = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        text = (
            item
            if isinstance(item, str)
            else item.get("text")
            if isinstance(item, dict)
            else None
        )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number}: expected non-empty text.")
        tickets.append(text.strip())
    return tickets


def generate_tickets(count, model, host):
    generated = []
    remaining = count
    while remaining:
        amount = min(remaining, 20)
        batch_schema = {
            "type": "object",
            "properties": {
                "tickets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": amount,
                    "maxItems": amount,
                }
            },
            "required": ["tickets"],
            "additionalProperties": False,
        }
        prompt = (
            f"Generate exactly {amount} distinct, realistic, concise, unstructured "
            "IT support ticket messages. Include a varied mix of hardware, software, "
            "and network issues; vary urgency and emotional wording. Do not provide "
            "labels or solutions. Return the messages in the requested structure."
        )
        response = call_ollama_json(
            prompt,
            batch_schema,
            model=model,
            host=host,
            temperature=0.7,
        )
        tickets = response.get("tickets") if isinstance(response, dict) else None
        if (
            not isinstance(tickets, list)
            or len(tickets) != amount
            or any(
                not isinstance(ticket, str) or not ticket.strip()
                for ticket in tickets
            )
        ):
            raise OllamaError(
                f"Ollama did not generate exactly {amount} non-empty ticket strings."
            )
        generated.extend(ticket.strip() for ticket in tickets)
        remaining -= amount
    return generated


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for item in items:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate optional synthetic tickets and label them with Ollama."
    )
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data" / "raw_prompts.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "curated_dataset.jsonl"
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=0,
        help="Generate and add this many synthetic tickets.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    if args.generate < 0:
        parser.error("--generate must be zero or greater.")

    try:
        tickets = read_tickets(args.input)
        if args.generate:
            generated = generate_tickets(args.generate, args.model, args.host)
            tickets.extend(generated)
            write_jsonl(args.input, [{"text": text} for text in tickets])
        if not tickets:
            raise ValueError(f"No tickets found in {args.input}.")

        examples = []
        for index, text in enumerate(tickets, 1):
            labels = call_ollama(
                build_prompt(text), model=args.model, host=args.host
            )
            examples.append(
                {
                    "prompt": build_prompt(text),
                    "completion": json.dumps(labels, ensure_ascii=False),
                }
            )
            print(f"Curated {index}/{len(tickets)}", file=sys.stderr)
        write_jsonl(args.output, examples)
    except (OllamaError, OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
