import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ticket_schema import (
    DEFAULT_MODEL,
    OLLAMA_HOST,
    OllamaError,
    build_prompt,
    call_ollama,
)

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui"
MAX_REQUEST_BYTES = 16_384
EXTRACT_TICKET = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class TicketHandler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type):
        encoded = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/":
            self._send(200, (UI / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/main.js":
            self._send(
                200, (UI / "main.js").read_bytes(), "text/javascript; charset=utf-8"
            )
        else:
            self._send(404, '{"error":"Not found"}', "application/json; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/extract":
            self._send(404, '{"error":"Not found"}', "application/json; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_REQUEST_BYTES:
                raise ValueError("Request body must be between 1 and 16384 bytes.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = payload.get("text") if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Provide a non-empty ticket in the text field.")
            if EXTRACT_TICKET is None:
                raise RuntimeError("Ticket extractor was not initialized.")
            result = EXTRACT_TICKET(text.strip())
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._send(
                400,
                json.dumps({"error": str(error)}),
                "application/json; charset=utf-8",
            )
            return
        except (OllamaError, RuntimeError) as error:
            logging.error("Ticket extraction failed: %s", error)
            self._send(
                502,
                json.dumps({"error": str(error)}),
                "application/json; charset=utf-8",
            )
            return
        self._send(
            200,
            json.dumps(result, ensure_ascii=False),
            "application/json; charset=utf-8",
        )

    def log_message(self, format_string, *args):
        logging.info("%s - %s", self.address_string(), format_string % args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the Ticket2JSON web UI and API.")
    parser.add_argument("--backend", choices=("ollama", "finetuned"), default="ollama")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--host", default=OLLAMA_HOST, help="Ollama API base URL.")
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
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.backend == "ollama":
        EXTRACT_TICKET = lambda text: call_ollama(
            build_prompt(text), model=args.model, host=args.host
        )
        backend_description = f"Ollama model {args.model} at {args.host}"
    else:
        try:
            from finetuned_backend import load_finetuned_extractor

            EXTRACT_TICKET = load_finetuned_extractor(
                model_name=args.base_model,
                adapter_path=args.adapter,
            )
        except RuntimeError as error:
            parser.exit(1, f"Cannot load fine-tuned model: {error}\n")
        backend_description = f"fine-tuned adapter {args.adapter}"

    address = ("127.0.0.1", args.port)
    server = ThreadingHTTPServer(address, TicketHandler)
    logging.info("Ticket2JSON available at http://%s:%s", *address)
    logging.info("Using %s", backend_description)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down.")
        server.server_close()
