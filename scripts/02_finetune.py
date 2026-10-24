import argparse
import json
import math
import os
from pathlib import Path

from ticket_schema import validate_ticket

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OUTPUT = ROOT / "outputs" / "qwen2.5-3b-ticket2json-lora"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def read_examples(path):
    examples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object.")
        prompt = item.get("prompt")
        completion = item.get("completion")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}:{line_number}: prompt must be non-empty text.")
        if not isinstance(completion, str):
            raise ValueError(f"{path}:{line_number}: completion must be JSON text.")
        try:
            validate_ticket(json.loads(completion))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:{line_number}: completion is not a valid ticket: {error}"
            ) from error
        examples.append({"prompt": prompt.strip(), "completion": completion.strip()})
    if not examples:
        raise ValueError(f"No training examples found in {path}.")
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a Qwen instruction model with 4-bit QLoRA."
    )
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data" / "curated_dataset.jsonl"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-seq-length", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.max_seq_length < 128:
        parser.error("--max-seq-length must be at least 128.")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("Batch size and gradient accumulation must be at least 1.")

    try:
        examples = read_examples(args.input)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Invalid training data: {error}\n")

    try:
        from unsloth import FastLanguageModel
        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        parser.exit(
            1,
            "Training dependencies are missing. In WSL, run "
            "`bash scripts/setup_training.sh` first. "
            f"Import failed: {error}\n",
        )

    if not torch.cuda.is_available():
        parser.exit(1, "CUDA GPU not detected; run this script in GPU-enabled WSL.\n")
    print(
        f"GPU: {torch.cuda.get_device_name(0)} "
        f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB)"
    )
    print(
        f"Training {len(examples)} examples with {args.model} "
        f"for {args.epochs:g} epochs."
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=TARGET_MODULES,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    training_data = Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": example["prompt"]}],
                "completion": [
                    {"role": "assistant", "content": example["completion"]}
                ],
            }
            for example in examples
        ]
    )
    updates_per_epoch = math.ceil(
        len(examples) / (args.batch_size * args.gradient_accumulation_steps)
    )
    warmup_steps = max(1, math.ceil(updates_per_epoch * args.epochs * 0.05))
    training_args = SFTConfig(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_length=args.max_seq_length,
        completion_only_loss=True,
        packing=False,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        processing_class=tokenizer,
    )
    trainer.train()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter and tokenizer saved to {args.output}")


if __name__ == "__main__":
    main()
