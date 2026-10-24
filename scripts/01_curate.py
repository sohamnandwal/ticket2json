import argparse
import json
import sys
from pathlib import Path

from ticket_schema import (
    DEFAULT_MODEL,
    OllamaError,
    TICKET_SCHEMA,
    build_prompt,
    call_ollama_json,
    validate_ticket,
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


def curate_tickets(tickets, model, host, batch_size):
    examples = []
    for offset in range(0, len(tickets), batch_size):
        batch = tickets[offset : offset + batch_size]
        label_schema = {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": TICKET_SCHEMA,
                    "minItems": len(batch),
                    "maxItems": len(batch),
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        }
        batch_prompt = (
            "Classify each IT ticket below. Return exactly one label for every "
            "ticket, in the same order, in the labels array. Do not skip, merge, "
            "or add tickets.\n\n"
            + json.dumps(
                [
                    {"ticket_number": offset + index + 1, "text": text}
                    for index, text in enumerate(batch)
                ],
                ensure_ascii=False,
            )
        )
        response = call_ollama_json(
            batch_prompt,
            label_schema,
            model=model,
            host=host,
        )
        labels = response.get("labels") if isinstance(response, dict) else None
        if not isinstance(labels, list) or len(labels) != len(batch):
            raise OllamaError(
                f"Ollama returned the wrong number of labels for tickets "
                f"{offset + 1}-{offset + len(batch)}."
            )
        for index, (text, label) in enumerate(zip(batch, labels), offset + 1):
            try:
                validate_ticket(label)
            except (TypeError, ValueError) as error:
                raise OllamaError(
                    f"Ollama returned an invalid label for ticket {index}: {error}"
                ) from error
            prompt = build_prompt(text)
            examples.append(
                {
                    "prompt": prompt,
                    "completion": json.dumps(label, ensure_ascii=False),
                }
            )
        print(
            f"Curated {offset + len(batch)}/{len(tickets)}",
            file=sys.stderr,
        )
    return examples


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Tickets to label per Ollama request (default: 10).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    if args.generate < 0:
        parser.error("--generate must be zero or greater.")
    if args.batch_size < 1 or args.batch_size > 20:
        parser.error("--batch-size must be between 1 and 20.")

    try:
        tickets = read_tickets(args.input)
        if args.generate:
            generated = generate_tickets(args.generate, args.model, args.host)
            tickets.extend(generated)
            write_jsonl(args.input, [{"text": text} for text in tickets])
        if not tickets:
            raise ValueError(f"No tickets found in {args.input}.")

        examples = curate_tickets(
            tickets,
            model=args.model,
            host=args.host,
            batch_size=args.batch_size,
        )
        write_jsonl(args.output, examples)
    except (OllamaError, OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
