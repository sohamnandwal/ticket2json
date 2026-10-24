import argparse
import json
import sys
from pathlib import Path

from ticket_schema import (
    DEFAULT_MODEL,
    OllamaError,
    build_prompt,
    call_ollama,
    validate_ticket,
)

ROOT = Path(__file__).resolve().parent.parent


def read_cases(path):
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if isinstance(item, str):
            text, expected = item, None
        elif isinstance(item, dict):
            text = item.get("text")
            expected = item.get("expected")
        else:
            text, expected = None, None
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_number}: expected non-empty text.")
        if expected is not None:
            try:
                validate_ticket(expected)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid expected ticket: {error}"
                ) from error
        cases.append((text.strip(), expected))
    return cases


def main():
    parser = argparse.ArgumentParser(description="Evaluate JSON validity and optional labels.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "holdout.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")

    try:
        cases = read_cases(args.input)[: args.limit]
        if not cases:
            raise ValueError(f"No evaluation cases found in {args.input}.")
        valid = 0
        exact = 0
        scored = sum(isinstance(expected, dict) for _, expected in cases)
        field_correct = {
            field: 0
            for field in (
                "issue_category",
                "urgency",
                "sentiment",
                "extracted_entities",
            )
        }
        for index, (text, expected) in enumerate(cases, 1):
            try:
                result = call_ollama(
                    build_prompt(text), model=args.model, host=args.host
                )
            except OllamaError as error:
                print(f"[{index}/{len(cases)}] invalid: {error}")
                continue
            valid += 1
            if isinstance(expected, dict):
                exact += result == expected
                for field in field_correct:
                    if field == "extracted_entities":
                        predicted_entities = {
                            entity.casefold() for entity in result[field]
                        }
                        expected_entities = {
                            entity.casefold()
                            for entity in expected.get(field, [])
                            if isinstance(entity, str)
                        }
                        field_correct[field] += predicted_entities == expected_entities
                    else:
                        field_correct[field] += result[field] == expected.get(field)
            print(f"[{index}/{len(cases)}] valid schema JSON")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")

    print(f"Schema-valid outputs: {valid}/{len(cases)} ({valid / len(cases):.1%})")
    if scored:
        print(f"Exact label matches: {exact}/{scored} ({exact / scored:.1%})")
        for field, correct in field_correct.items():
            print(f"{field} matches: {correct}/{scored} ({correct / scored:.1%})")
    if valid != len(cases):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
