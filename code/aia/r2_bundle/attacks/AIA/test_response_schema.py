import unittest

from attacks.AIA.response_schema import Outcome, classify_response


class AIAResponseTests(unittest.TestCase):
    def setUp(self):
        self.allowed = {"female", "male", "non-binary"}

    def test_valid_and_explicit_alias(self):
        self.assertEqual(classify_response(" Female ", self.allowed)[0], Outcome.VALID)
        self.assertEqual(
            classify_response("woman", self.allowed, {"woman": "female"}),
            (Outcome.VALID, "female"),
        )
        self.assertEqual(
            classify_response("woman", self.allowed, {"woman": "FEMALE"}),
            (Outcome.VALID, "female"),
        )

    def test_invalid_is_not_forced_to_non_binary(self):
        outcome, label = classify_response("probably a man", self.allowed)
        self.assertEqual(outcome, Outcome.OUT_OF_TAXONOMY)
        self.assertIsNone(label)

    def test_refusal_unknown_empty_malformed_are_distinct(self):
        self.assertEqual(classify_response("I don't know", self.allowed)[0], Outcome.UNKNOWN)
        self.assertEqual(classify_response("I can't help infer that", self.allowed)[0], Outcome.REFUSAL)
        self.assertEqual(classify_response("I can’t assist with that", self.allowed)[0], Outcome.REFUSAL)
        self.assertEqual(classify_response("Unknown", self.allowed)[0], Outcome.UNKNOWN)
        self.assertEqual(classify_response("", self.allowed)[0], Outcome.EMPTY)
        self.assertEqual(classify_response([], self.allowed)[0], Outcome.EMPTY)
        self.assertEqual(
            classify_response(None, self.allowed, field_present=False),
            (Outcome.MISSING, None),
        )
        self.assertEqual(classify_response(["male"], self.allowed)[0], Outcome.MALFORMED)


if __name__ == "__main__":
    unittest.main()
