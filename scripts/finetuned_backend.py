import json
import os
import threading
from pathlib import Path

from ticket_schema import TICKET_SCHEMA, build_prompt, validate_ticket

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))


class TicketModelError(RuntimeError):
    pass


def load_finetuned_extractor(
    model_name=None,
    adapter_path=Path("outputs/qwen2.5-3b-ticket2json-lora"),
):
    try:
        import torch
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )
        from peft import PeftConfig, PeftModel
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
    except ImportError as error:
        raise TicketModelError(
            "Fine-tuned serving dependencies are missing. Run "
            "`bash scripts/setup_training.sh` in WSL."
        ) from error

    adapter_path = Path(adapter_path)
    if not adapter_path.is_dir():
        raise TicketModelError(f"LoRA adapter directory not found: {adapter_path}")
    if not torch.cuda.is_available():
        raise TicketModelError("CUDA GPU not detected; run the fine-tuned server in WSL.")

    adapter_config = PeftConfig.from_pretrained(adapter_path)
    model_name = model_name or adapter_config.base_model_name_or_path
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    base_config = AutoConfig.from_pretrained(model_name)
    model_kwargs = {"config": base_config, "device_map": "auto", "dtype": compute_dtype}
    if getattr(base_config, "quantization_config", None) is None:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    allowed_tokens = build_transformers_prefix_allowed_tokens_fn(
        tokenizer, JsonSchemaParser(TICKET_SCHEMA)
    )
    generation_lock = threading.Lock()

    def extract_ticket(text):
        messages = [{"role": "user", "content": build_prompt(text)}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
        try:
            with generation_lock, torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    prefix_allowed_tokens_fn=allowed_tokens,
                )
        except (RuntimeError, ValueError) as error:
            raise TicketModelError(f"Fine-tuned model generation failed: {error}") from error

        response = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        try:
            return validate_ticket(json.loads(response))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise TicketModelError(
                f"Fine-tuned model did not produce a valid ticket: {error}"
            ) from error

    return extract_ticket
