"""Combine completed, hash-pinned Cycle A aggregates into analysis tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from attacks.AIA.experiment import ATTRIBUTES
from attacks.AIA.scoring import validate_ranked_summary
from attacks.MIA.score_tokenized import reject_duplicate_json_keys
from scripts.run_cycle_a import DEA_SOURCE, DIGEST, RUN_ID
from scripts.run_safety import private_dir, private_file, publish_json, read_bound_bytes


def load_results(sources, expected_kinds):
    if not isinstance(sources, list) or not sources:
        raise ValueError("specify the exact completed summaries")
    run_ids = set()
    code_hashes = set()
    results = []
    for source in sources:
        summary = json.loads(read_bound_bytes(source), object_pairs_hook=reject_duplicate_json_keys)
        if (summary["status"] != "complete" or summary["kind"] not in expected_kinds
                or not RUN_ID.fullmatch(summary["run_id"]) or summary["run_id"] in run_ids
                or not DIGEST.fullmatch(summary["code_sha256"])):
            raise ValueError("summary is incomplete, duplicated or outside the selected analysis")
        run_ids.add(summary["run_id"])
        code_hashes.add(summary["code_sha256"])
        results.append(summary["result"])
    if len(code_hashes) != 1:
        raise ValueError("summaries use different analysis code bundles")
    return results


def collect_dea(results):
    sys.path.insert(0, str(DEA_SOURCE))
    from dea_a3.aggregate import mean_ci

    compare = {(row["model"], row["condition"], row["comparison_sha256"]) for row in results}
    if len(compare) != 1 or sorted(row["seed"] for row in results) != [42, 43, 44]:
        raise ValueError("a DEA publication group requires all three comparable completed seeds")
    if not DIGEST.fullmatch(next(iter(compare))[2]):
        raise ValueError("invalid DEA comparison identity")
    modes = {row.get("evidence_mode", "targeted_rerun") for row in results}
    if len(modes) != 1 or not modes <= {"targeted_rerun", "historical_replay"}:
        raise ValueError("DEA publication groups require one qualified evidence mode")
    budgets = [row["generations"] for row in results]
    if any(type(n) is not int or n <= 0 for n in budgets) or len(set(budgets)) != 1:
        raise ValueError("DEA seeds must share one positive query budget")
    for row in results:
        for kind in ("email", "phone"):
            extracted = row["distinct"][f"distinct_extracted_{kind}"]
            verified = row["distinct"][f"distinct_verified_{kind}"]
            extracted_total, verified_total = row[f"extracted_{kind}"], row[f"verified_{kind}"]
            if (any(type(value) is not int for value in (extracted, verified, extracted_total, verified_total))
                    or not 0 <= verified <= min(extracted, verified_total)
                    or not max(extracted, verified_total) <= extracted_total):
                raise ValueError("DEA distinct counts do not reconcile with occurrence counts")
    measures = ("distinct_extracted_email", "distinct_verified_email",
                "distinct_extracted_phone", "distinct_verified_phone")
    values = {}
    for measure in measures:
        counts = [row["distinct"][measure] for row in results]
        if any(type(count) is not int or count < 0 for count in counts):
            raise ValueError("invalid DEA count")
        summary = mean_ci(counts)
        values[measure] = {"mean": summary["mean"], "n_seeds": summary["n"],
                           "ci_95_half_width": summary["half_width"]}
    model, condition, identity = compare.pop()
    return {"model": model, "condition": condition, "comparison_sha256": identity,
            "queries_per_seed": budgets[0], "evidence_mode": modes.pop(), "metrics": values}


def collect_aia(results, expected_attributes):
    if (not isinstance(expected_attributes, list) or not expected_attributes
            or len(set(expected_attributes)) != len(expected_attributes)
            or not set(expected_attributes) <= ATTRIBUTES):
        raise ValueError("AIA analysis requires a unique prespecified attribute list")
    attributes = {row["attribute"]: row for row in results}
    if len(attributes) != len(results) or set(attributes) != set(expected_attributes):
        raise ValueError("results do not cover the exact prespecified attributes")
    identity = {(row["model_id"], row.get("model_revision"), row["dataset_identity"],
                 row.get("paired_control", False), row["comparison_sha256"]) for row in results}
    if len(identity) != 1:
        raise ValueError("AIA reports must use the same dataset, model and inference design")
    model, revision, dataset, paired, comparison = identity.pop()
    if type(paired) is not bool or not DIGEST.fullmatch(comparison):
        raise ValueError("invalid AIA comparison identity")
    for row in results:
        validate_ranked_summary(row["metrics"], paired=paired)
    metrics = {}
    for rank in ("top_1", "top_3"):
        metrics[rank] = {}
        for condition in (("full", "evidence_removed") if paired else ("full",)):
            group = [attributes[attr]["metrics"][rank]["conditions"][condition]
                     for attr in expected_attributes]
            n = sum(row["n_profiles"] for row in group)
            correct = sum(row["n_observed_correct"] for row in group)
            unresolved = sum(row["n_profiles_unresolved"] for row in group)
            macro = sum(row["n_observed_correct"] / row["n_profiles"] for row in group) / len(group)
            macro_upper = sum((row["n_observed_correct"] + row["n_profiles_unresolved"])
                              / row["n_profiles"] for row in group) / len(group)
            metrics[rank][condition] = {
                "n_eligible_attribute_rows": n,
                "n_unresolved": unresolved,
                "micro_accuracy": correct / n if not unresolved else None,
                "micro_bounds": [correct / n, (correct + unresolved) / n],
                "macro_accuracy": macro if not unresolved else None,
                "macro_bounds": [macro, macro_upper],
            }
        if paired:
            name = "paired_overall_correct_rate_delta_full_minus_evidence_removed"
            effects = [attributes[attr]["metrics"][rank][name] for attr in expected_attributes]
            n_pairs = sum(effect["n_paired_profiles"] for effect in effects)
            unresolved = sum(effect["n_unresolved_pairs"] for effect in effects)
            micro = [
                sum(effect["bounds"][i] * effect["n_paired_profiles"] for effect in effects) / n_pairs
                for i in (0, 1)
            ]
            macro = [sum(effect["bounds"][i] for effect in effects) / len(effects) for i in (0, 1)]
            metrics[rank]["paired_effect"] = {
                "n_pairs": n_pairs, "n_unresolved_pairs": unresolved,
                "micro_delta": None if unresolved else micro[0], "micro_delta_bounds": micro,
                "macro_delta": None if unresolved else macro[0], "macro_delta_bounds": macro,
            }
    return {"model_id": model, "model_revision": revision, "dataset_identity": dataset,
            "comparison_sha256": comparison,
            "attributes": expected_attributes, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        private_file(args.spec)
        spec = json.loads(args.spec.read_bytes(), object_pairs_hook=reject_duplicate_json_keys)
        kind = spec["kind"]
        if kind == "dea_cell" and spec.keys() == {"kind", "sources"}:
            result = collect_dea(load_results(spec["sources"], {"dea_cell", "dea_replay"}))
        elif kind == "aia" and spec.keys() == {"kind", "sources", "expected_attributes"}:
            result = collect_aia(load_results(spec["sources"], {"aia_inference", "aia_rescore"}),
                                 spec["expected_attributes"])
        else:
            raise ValueError("unsupported collection spec")
        private_dir(args.output.parent)
        publish_json(args.output, {"kind": kind, "source_sha256s": [s["sha256"] for s in spec["sources"]],
                                   "result": result})
    except Exception as error:
        parser.exit(1, f"Cycle A analysis blocked or failed ({type(error).__name__}).\n")


if __name__ == "__main__":
    main()
