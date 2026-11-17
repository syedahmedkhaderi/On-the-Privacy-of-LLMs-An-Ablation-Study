import json
from pathlib import Path
import tempfile
import unittest

from attacks.AIA.audit_saved_files import audit


class SavedFileAuditTests(unittest.TestCase):
    def test_schema_only_output_does_not_copy_raw_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset.jsonl"
            prediction = root / "prediction.jsonl"
            dataset.write_text(json.dumps({"feature": "age", "response": "synthetic raw text"}) + "\n")
            prediction.write_text(json.dumps({
                "username": "synthetic-user", "comments": [{"text": "private comment"}],
                "predictions": {"model": {"age": "42"}},
            }) + "\n")
            report = audit(dataset, [prediction])
            serialized = json.dumps(report)
            self.assertNotIn("synthetic raw text", serialized)
            self.assertNotIn("synthetic-user", serialized)
            self.assertNotIn("private comment", serialized)
            self.assertFalse(report["prediction_to_dataset_row_order_verified"])
            self.assertEqual(report["dataset"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
