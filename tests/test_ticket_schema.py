import unittest

from scripts.ticket_schema import build_prompt, validate_ticket


class TicketSchemaTests(unittest.TestCase):
    def test_accepts_exact_schema(self):
        ticket = {
            "issue_category": "hardware",
            "urgency": "high",
            "sentiment": "panicked",
            "extracted_entities": ["laptop", "screen"],
        }
        self.assertEqual(validate_ticket(ticket), ticket)

    def test_rejects_extra_fields(self):
        with self.assertRaisesRegex(ValueError, "exactly the four required fields"):
            validate_ticket(
                {
                    "issue_category": "hardware",
                    "urgency": "low",
                    "sentiment": "neutral",
                    "extracted_entities": [],
                    "commentary": "extra",
                }
            )

    def test_rejects_invalid_enum(self):
        with self.assertRaisesRegex(ValueError, "issue_category"):
            validate_ticket(
                {
                    "issue_category": "other",
                    "urgency": "low",
                    "sentiment": "neutral",
                    "extracted_entities": [],
                }
            )

    def test_rejects_non_string_enum(self):
        with self.assertRaisesRegex(ValueError, "issue_category"):
            validate_ticket(
                {
                    "issue_category": [],
                    "urgency": "low",
                    "sentiment": "neutral",
                    "extracted_entities": [],
                }
            )

    def test_prompt_includes_ticket(self):
        prompt = build_prompt("Wi-Fi is down.")
        self.assertIn("Wi-Fi is down.", prompt)
        self.assertIn("Return only", prompt)


if __name__ == "__main__":
    unittest.main()
