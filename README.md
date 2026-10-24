# Ticket2JSON

Extracts a fixed set of fields from unstructured IT support tickets using a
locally installed Ollama model. Ollama's JSON-schema constrained generation is
used for inference, and the application checks the returned object before
serving it.

## Local setup

The default Ollama-based app and curation/evaluation scripts use only the
Python standard library. Start Ollama, confirm that the `llama3.1:latest`
model is installed, and run from the project directory:

```powershell
.\venv\Scripts\python.exe scripts\03_serve.py
```

Open <http://127.0.0.1:8000>. The app binds to loopback and proxies requests to
Ollama at `http://127.0.0.1:11434`; the browser does not call Ollama directly.
To use another installed model, update `DEFAULT_MODEL` in
[`scripts/ticket_schema.py`](scripts/ticket_schema.py).

The extraction object has exactly these fields:

```json
{
  "issue_category": "hardware|software|network",
  "urgency": "low|medium|high",
  "sentiment": "angry|neutral|panicked",
  "extracted_entities": ["string"]
}
```

## Curate and evaluate

`data/raw_prompts.jsonl` contains one unstructured ticket per JSONL row. Curate
the examples into the `{"prompt": "...", "completion": "..."}` format:

```powershell
.\venv\Scripts\python.exe scripts\01_curate.py
```

To have Ollama add synthetic tickets before labeling them:

```powershell
.\venv\Scripts\python.exe scripts\01_curate.py --generate 300
```

Generated examples are weak labels from the same model and are not a substitute
for human-reviewed training or evaluation data. Keep a separately labeled
holdout set in `data/holdout.jsonl`, then run:

```powershell
.\venv\Scripts\python.exe scripts\04_eval.py
```

The evaluator reports schema-valid JSON separately from exact label matches.
Schema-constrained decoding aims to prevent malformed JSON; it does not
guarantee correct classifications. The included holdout examples are a smoke
test, not a statistically meaningful benchmark.

## Fine-tuning and GPU notes

The installed `llama3.1:latest` is an Ollama-managed 4-bit quantized model and
is suitable for local inference, but Ollama's quantized model file cannot be
fine-tuned in place. The training script therefore uses the public Qwen
2.5-3B-Instruct model through Unsloth's 4-bit checkpoint and saves a QLoRA adapter;
the installed Qwen Coder 7B checkpoint is code-focused, while this 3B
instruction model is a better fit for ticket classification and uses less VRAM.

The RTX 5060 Laptop GPU has 8 GB VRAM and is visible in this machine's Kali
WSL environment. Set up the separate Linux training environment there:

```bash
cd /mnt/d/MyWork/SyntaxError26/ticket2json
bash scripts/setup_training.sh
```

The training environment is installed under `~/.venvs/ticket2json` inside the
Linux filesystem (not on the mounted Windows drive).
Hugging Face model files are cached in the project `.cache` directory on D:\
to avoid filling the Windows system drive.

Generate a larger weakly labeled dataset from Windows PowerShell, where
Ollama's local API is available:

```powershell
.\venv\Scripts\python.exe scripts\01_curate.py --generate 300
```

After curation completes, run this in Windows PowerShell to release the
teacher model's VRAM:

```powershell
ollama stop llama3.1:latest
```

Then start the QLoRA fine-tuning in WSL:

```bash
source "$HOME/.venvs/ticket2json/bin/activate"
python scripts/02_finetune.py
```

The adapter is saved under
`outputs/qwen2.5-3b-ticket2json-lora`. QLoRA uses a 4-bit base model,
gradient checkpointing, rank-16 LoRA, and small micro-batches for the 8 GB
GPU. The 300 generated labels are weak supervision from Llama 3.1; review the
data before training and use a separately maintained gold holdout set for
evaluation. The included holdout is only a smoke test. Training does not
guarantee better labels, and schema-constrained decoding remains a serving-time
feature rather than a property of the adapter itself.

Run the fine-tuned model with the same UI/API and schema-constrained JSON
decoding, or evaluate it against the holdout:

```bash
source "$HOME/.venvs/ticket2json/bin/activate"
python scripts/03_serve.py --backend finetuned
python scripts/04_eval.py --backend finetuned
```

The fine-tuned backend loads the Qwen base in 4-bit and applies the saved LoRA
adapter; the Ollama backend remains available with the default server command.
