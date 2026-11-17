import unittest

from attacks.AIA.scoring import summarize_aia_rows


def row(profile, condition, truth, prediction, outcome="valid"):
    return {
        "profile_id_hash": f"{profile:064x}",
        "condition": condition,
        "ground_truth_label": truth,
        "prediction_label": prediction,
        "outcome": outcome,
    }


class AIAScoringTests(unittest.TestCase):
    def test_aggregate_outcomes_baselines_and_paired_difference(self):
        rows = [
            row(1, "full", "female", "female"),
            row(1, "no_text_prior", "female", "male"),
            row(2, "full", "male", None, "refusal"),
            row(2, "no_text_prior", "male", None, "unknown"),
            row(3, "full", "male", "male"),
            row(3, "no_text_prior", "male", "male"),
        ]
        summary = summarize_aia_rows(rows, {"female", "male"}, repetitions=100, seed=3)
        self.assertEqual(summary["conditions"]["full"]["n_profiles"], 3)
        self.assertEqual(summary["conditions"]["full"]["outcome_counts"]["refusal"], 1)
        self.assertAlmostEqual(summary["conditions"]["full"]["overall_correct_rate"], 2 / 3)
        self.assertAlmostEqual(summary["conditions"]["full"]["majority_baseline_rate"], 2 / 3)
        self.assertEqual(summary["paired_overall_correct_rate_delta_full_minus_no_text_prior"]["n_paired_profiles"], 3)
        self.assertFalse(summary["raw_rows_written"])

    def test_unpaired_conditions_do_not_get_paired_estimate(self):
        summary = summarize_aia_rows(
            [row(1, "full", "female", "female"), row(2, "no_text_prior", "male", "male")],
            {"female", "male"}, repetitions=100,
        )
        self.assertIsNone(summary["paired_overall_correct_rate_delta_full_minus_no_text_prior"])

    def test_duplicate_profile_condition_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize_aia_rows(
                [row(1, "full", "female", "female"), row(1, "full", "female", "male")],
                {"female", "male"}, repetitions=100,
            )

    def test_nonvalid_outcome_cannot_carry_prediction(self):
        with self.assertRaises(ValueError):
            summarize_aia_rows([row(1, "full", "female", "male", "unknown")],
                               {"female", "male"}, repetitions=100)

    def test_paired_conditions_require_same_truth(self):
        with self.assertRaises(ValueError):
            summarize_aia_rows(
                [row(1, "full", "female", "female"),
                 row(1, "no_text_prior", "male", "male")],
                {"female", "male"}, repetitions=100,
            )

    def test_missing_guesses_use_full_population_bounds(self):
        rows = [
            row(1, "full", "female", "female"),
            row(2, "full", "male", None, "missing"),
            row(3, "full", "male", None, "empty"),
            row(4, "full", "male", None, "refusal"),
        ]
        summary = summarize_aia_rows(rows, {"female", "male"}, repetitions=100, seed=4)
        result = summary["conditions"]["full"]
        self.assertEqual(result["n_profiles"], 4)
        self.assertEqual(result["n_profiles_scored"], 3)
        self.assertEqual(result["n_profiles_unresolved"], 1)
        self.assertEqual(result["outcome_counts"]["missing"], 1)
        self.assertIsNone(result["overall_correct_rate"])
        self.assertEqual(result["overall_correct_rate_bounds"], (0.25, 0.5))
        self.assertEqual(result["valid_coverage"], 0.25)

    def test_paired_delta_retains_profiles_with_unresolved_parse(self):
        rows = [
            row(1, "full", "female", "female"),
            row(1, "no_text_prior", "female", "male"),
            row(2, "full", "male", None, "missing"),
            row(2, "no_text_prior", "male", "male"),
        ]
        summary = summarize_aia_rows(rows, {"female", "male"}, repetitions=100, seed=5)
        paired = summary["paired_overall_correct_rate_delta_full_minus_no_text_prior"]
        self.assertEqual(paired["n_paired_profiles"], 2)
        self.assertEqual(paired["n_unresolved_pairs"], 1)
        self.assertIsNone(paired["estimate"])
        self.assertEqual(paired["bounds"], (0.0, 0.5))


if __name__ == "__main__":
    unittest.main()
