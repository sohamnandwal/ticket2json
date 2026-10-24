import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ticket_schema import (
    DEFAULT_MODEL,
    OllamaError,
    build_prompt,
    call_ollama,
)

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui"
MAX_REQUEST_BYTES = 16_384

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
            result = call_ollama(build_prompt(text.strip()), model=DEFAULT_MODEL)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self._send(
                400,
                json.dumps({"error": str(error)}),
                "application/json; charset=utf-8",
            )
            return
        except OllamaError as error:
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
    address = ("127.0.0.1", 8000)
    server = ThreadingHTTPServer(address, TicketHandler)
    logging.info("Ticket2JSON available at http://%s:%s", *address)
    logging.info("Using Ollama model %s at http://127.0.0.1:11434", DEFAULT_MODEL)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down.")
        server.server_close()
