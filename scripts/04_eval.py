import argparse
import json
import sys
from pathlib import Path

from ticket_schema import (
    DEFAULT_MODEL,
    OLLAMA_HOST,
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
    parser.add_argument("--backend", choices=("ollama", "finetuned"), default="ollama")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "holdout.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=OLLAMA_HOST)
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional Hugging Face base model override; defaults to the adapter's base.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "outputs" / "qwen2.5-3b-ticket2json-lora",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument(
        "--show-predictions",
        action="store_true",
        help="Print each prediction and its expected label when available.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")

    try:
        cases = read_cases(args.input)[: args.limit]
        if not cases:
            raise ValueError(f"No evaluation cases found in {args.input}.")
        if args.backend == "ollama":
            extractor = lambda text: call_ollama(
                build_prompt(text), model=args.model, host=args.host
            )
        else:
            from finetuned_backend import load_finetuned_extractor

            extractor = load_finetuned_extractor(
                model_name=args.base_model,
                adapter_path=args.adapter,
            )
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
        entity_true_positives = 0
        entity_false_positives = 0
        entity_false_negatives = 0
        for index, (text, expected) in enumerate(cases, 1):
            try:
                result = extractor(text)
            except (OllamaError, RuntimeError) as error:
                print(f"[{index}/{len(cases)}] invalid: {error}")
                continue
            valid += 1
            if isinstance(expected, dict):
                exact += result == expected
                predicted_entities = {
                    entity.casefold() for entity in result["extracted_entities"]
                }
                expected_entities = {
                    entity.casefold()
                    for entity in expected.get("extracted_entities", [])
                    if isinstance(entity, str)
                }
                entity_true_positives += len(
                    predicted_entities & expected_entities
                )
                entity_false_positives += len(
                    predicted_entities - expected_entities
                )
                entity_false_negatives += len(
                    expected_entities - predicted_entities
                )
                for field in field_correct:
                    if field == "extracted_entities":
                        field_correct[field] += predicted_entities == expected_entities
                    else:
                        field_correct[field] += result[field] == expected.get(field)
            print(f"[{index}/{len(cases)}] valid schema JSON")
            if args.show_predictions:
                print(f"  predicted: {json.dumps(result, ensure_ascii=False)}")
                if isinstance(expected, dict):
                    print(
                        f"  expected:  {json.dumps(expected, ensure_ascii=False)}"
                    )
    except (OSError, ValueError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")

    print(f"Schema-valid outputs: {valid}/{len(cases)} ({valid / len(cases):.1%})")
    if scored:
        print(f"Exact label matches: {exact}/{scored} ({exact / scored:.1%})")
        for field, correct in field_correct.items():
            print(f"{field} matches: {correct}/{scored} ({correct / scored:.1%})")
        entity_precision = (
            entity_true_positives
            / (entity_true_positives + entity_false_positives)
            if entity_true_positives + entity_false_positives
            else 0
        )
        entity_recall = (
            entity_true_positives
            / (entity_true_positives + entity_false_negatives)
            if entity_true_positives + entity_false_negatives
            else 0
        )
        entity_f1 = (
            2 * entity_precision * entity_recall / (entity_precision + entity_recall)
            if entity_precision + entity_recall
            else 0
        )
        print(
            "Entity micro precision/recall/F1: "
            f"{entity_precision:.1%}/{entity_recall:.1%}/{entity_f1:.1%}"
        )
    if valid != len(cases):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
