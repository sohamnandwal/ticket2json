import importlib
import json
import sys
import threading
import unittest
from urllib.error import HTTPError
from pathlib import Path
from urllib.request import Request, urlopen

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
server_module = importlib.import_module("03_serve")


class TicketServerTests(unittest.TestCase):
    def setUp(self):
        self.original_extractor = server_module.EXTRACT_TICKET
        server_module.EXTRACT_TICKET = lambda text: {
            "issue_category": "network",
            "urgency": "medium",
            "sentiment": "neutral",
            "extracted_entities": [text],
        }
        self.server = server_module.ThreadingHTTPServer(
            ("127.0.0.1", 0), server_module.TicketHandler
        )
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        server_module.EXTRACT_TICKET = self.original_extractor

    def test_extract_endpoint_returns_schema_result(self):
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/api/extract",
            data=json.dumps({"text": "Wi-Fi is down."}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            result = json.loads(response.read())
        self.assertEqual(result["issue_category"], "network")
        self.assertEqual(result["extracted_entities"], ["Wi-Fi is down."])

    def test_extract_endpoint_rejects_empty_ticket(self):
        request = Request(
            f"http://127.0.0.1:{self.server.server_port}/api/extract",
            data=json.dumps({"text": " "}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as context:
            urlopen(request)
        self.assertEqual(context.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
