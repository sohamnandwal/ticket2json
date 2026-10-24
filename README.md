# Ticket2JSON

Extracts a fixed set of fields from unstructured IT support tickets using a
locally installed Ollama model. Ollama's JSON-schema constrained generation is
used for inference, and the application checks the returned object before
serving it.

## Local setup

This project uses only the Python standard library. Start Ollama, confirm that
the `llama3.1:latest` model is installed, and run from the project directory:

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

## Training and GPU notes

The installed `llama3.1:latest` is an Ollama-managed 4-bit quantized model and
is used directly for local inference. The detected RTX 5060 Laptop GPU has
8 GB VRAM; Ollama can use it for inference when its GPU runtime is available.
This application does not fine-tune the Ollama model. The suggested Unsloth,
TRL, and SGLang training/serving flow is a distinct Linux environment with
separate dependencies and checkpoint compatibility requirements; it is not
needed for this local extraction workflow.
