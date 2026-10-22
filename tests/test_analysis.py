"""Synthetic unit-test fixtures only: no claimed human or market observations."""
import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from auspicious.analysis import (
    CONTEXT_FIELDS, HUMAN_FIELDS, analyze, contextual_series, import_humans,
)
from auspicious.core import write_jsonl


def payload(gold=30, buy=False):
    return {"gold_pct": gold, "equity_pct": 40, "cash_pct": 60 - gold,
            "buy_now": buy, "timing_days": 0 if buy else 7,
            "willingness_to_pay_inr": 5000, "expected_return_pct": 4,
            "rationale": "Synthetic test answer."}


def pair(cluster="s1", difference=10, language="en", replicate=0, version="v1",
         subject=None, role="investor", calendar=None):
    pair_id = f"{cluster}-{language}-{replicate}-{role}-{calendar}"
    records = []
    for condition in ("event", "neutral"):
        prompt = f"Synthetic fixture {pair_id} {condition}"
        trial = {"trial_id": f"{pair_id}-{condition}", "pair_id": pair_id,
                 "cluster_id": cluster, "scenario_id": cluster, "split": "test",
                 "language": language, "festival": "dhanteras", "condition": condition,
                 "event_valence": "auspicious", "cue": "festival_named", "role": role,
                 "persona_id": "default", "paraphrase_id": "p0", "replicate": replicate,
                 "calendar_pair_id": calendar, "prompt": prompt,
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                 "review_status": "draft", "review": {}}
        answer = payload(30 + difference if condition == "event" else 30)
        result = {"status": "ok", "raw_text": json.dumps(answer), "payload": answer,
                  "provider": "fake", "requested_model": "fixture", "model_version": version,
                  "generation": {"temperature": 0}, "usage": {}, "attempts": 1, "error": None}
        if subject is not None:
            result.update(provider="human", requested_model="human", model_version="self_report",
                          generation={}, subject_id=subject)
        records.append({"trial": trial, "result": result})
    return records


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_analysis(self, records, humans=None, bootstrap=100, seed=42):
        records_path = self.root / "records.jsonl"
        write_jsonl(records_path, records)
        kwargs = {}
        if humans is not None:
            human_path = self.root / "humans.jsonl"
            write_jsonl(human_path, humans)
            kwargs["human_path"] = human_path
        return analyze(records_path, self.root / "analysis", bootstrap=bootstrap, seed=seed, **kwargs)

    def gold(self, report, **filters):
        return next(row for row in report["effects"] if row["metric"] == "gold_pct" and
                    all(row.get(key) == value for key, value in filters.items()))

    def csv_file(self, name, fields, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_equal_scenario_weight_not_repetition_weight(self):
        records = pair("s1", 10) + pair("s1", 10, replicate=1) + pair("s1", 10, replicate=2) + pair("s2", 0)
        row = self.gold(self.run_analysis(records))
        self.assertEqual(row["estimate"], 5)
        self.assertEqual(row["n_pairs"], 4)
        self.assertEqual(row["n_clusters"], 2)
        self.assertEqual(row["unit"], "percentage_points")
        self.assertLessEqual(row["ci_low"], row["estimate"])
        self.assertGreaterEqual(row["ci_high"], row["estimate"])

    def test_seed_and_output_are_deterministic(self):
        records = pair("s1", 5) + pair("s2", 20) + pair("s3", -5)
        first = self.run_analysis(records)
        second = self.run_analysis(records)
        self.assertEqual(first, second)
        written = json.loads((self.root / "analysis" / "analysis.json").read_text())
        self.assertEqual(written, second)
        self.assertTrue((self.root / "analysis" / "effects.csv").read_text().startswith("provider,"))

    def test_single_cluster_has_no_inferential_interval(self):
        row = self.gold(self.run_analysis(pair()))
        self.assertEqual(row["estimate"], 10)
        self.assertIsNone(row["ci_low"])
        self.assertEqual(row["ci_unavailable_reason"], "fewer_than_two_clusters")

    def test_disabled_bootstrap(self):
        row = self.gold(self.run_analysis(pair("s1") + pair("s2"), bootstrap=0))
        self.assertIsNone(row["ci_low"])
        self.assertEqual(row["ci_unavailable_reason"], "bootstrap_disabled")

    def test_invalid_blocked_error_and_missing_pairs_are_counted(self):
        records = pair("valid")
        for status in ("invalid", "blocked", "error"):
            bad = pair(status)
            bad[0]["result"].update(status=status, payload=None, raw_text="not an answer")
            records.extend(bad)
        records.extend(pair("missing")[:1])
        report = self.run_analysis(records)
        self.assertEqual(self.gold(report)["n_pairs"], 1)
        coverage = report["pair_coverage"][0]
        self.assertEqual(coverage["observed_pairs"], 5)
        self.assertEqual(coverage["complete_pair_coverage"], .2)
        counts = {row["status"]: row for row in report["missingness"]}
        self.assertEqual(counts["blocked"]["count"], 1)
        self.assertEqual(counts["invalid"]["denominator"], 9)
        self.assertTrue(report["warnings"])
        self.assertEqual(len(report["exclusions"]), 4)

    def test_ok_label_does_not_bypass_payload_validation(self):
        records = pair()
        records[0]["result"]["payload"]["gold_pct"] = 101
        report = self.run_analysis(records)
        self.assertIsNone(self.gold(report)["estimate"])
        self.assertEqual(next(r for r in report["missingness"] if r["status"] == "invalid")["count"], 1)

    def test_malformed_third_record_invalidates_pair_and_counts_globally(self):
        records = pair()
        malformed = copy.deepcopy(records[0])
        del malformed["trial"]["scenario_id"]
        records.extend([malformed, {"unexpected": "shape"}])
        report = self.run_analysis(records)
        self.assertEqual(report["pair_coverage"][0]["complete_pairs"], 0)
        total = next(row for row in report["record_status_totals"] if row["status"] == "malformed")
        self.assertEqual(total["count"], 2)
        self.assertEqual(total["denominator"], 4)

    def test_binary_outcome_uses_probability_difference_not_percentage(self):
        records = pair()
        records[0]["result"]["payload"].update(buy_now=True, timing_days=0)
        report = self.run_analysis(records)
        row = next(row for row in report["effects"] if row["metric"] == "buy_now")
        self.assertEqual(row["estimate"], 1)
        self.assertEqual(row["unit"], "probability_difference")

    def test_duplicate_trial_excludes_whole_pair(self):
        records = pair()
        records.append(copy.deepcopy(records[0]))
        report = self.run_analysis(records)
        self.assertEqual(report["pair_coverage"][0]["complete_pairs"], 0)
        self.assertTrue(any("duplicate_trial_observation" in row["reason"] for row in report["exclusions"]))

    def test_relabelled_duplicate_task_is_not_pseudoreplicated(self):
        records = pair()
        clone = copy.deepcopy(records)
        for record in clone:
            record["trial"]["pair_id"] += "-copy"
            record["trial"]["trial_id"] += "-copy"
        report = self.run_analysis(records + clone)
        self.assertEqual(report["pair_coverage"][0]["complete_pairs"], 0)
        self.assertTrue(any("duplicate_semantic_task" in r["reason"] for r in report["exclusions"]))

    def test_pair_metadata_mismatch_is_excluded(self):
        records = pair()
        records[1]["trial"]["scenario_id"] = "different"
        report = self.run_analysis(records)
        self.assertEqual(report["pair_coverage"][0]["complete_pairs"], 0)
        self.assertIn("inconsistent_pair_metadata", [r["reason"] for r in report["exclusions"]])

    def test_versions_and_settings_are_never_mixed(self):
        records = pair("s1", 10, version="v1") + pair("s1", -10, version="v2")
        report = self.run_analysis(records)
        self.assertEqual(self.gold(report, model_version="v1")["estimate"], 10)
        self.assertEqual(self.gold(report, model_version="v2")["estimate"], -10)
        records = pair()
        records[1]["result"]["model_version"] = "v2"
        report = self.run_analysis(records)
        self.assertTrue(all(r["complete_pairs"] == 0 for r in report["pair_coverage"]))
        records = pair()
        records[1]["result"]["generation"] = {"temperature": 1}
        report = self.run_analysis(records)
        self.assertTrue(all(r["complete_pairs"] == 0 for r in report["pair_coverage"]))

    def test_language_interaction_uses_identical_tasks_and_paired_clusters(self):
        records = (pair("s1", 5) + pair("s1", 15, language="hi") +
                   pair("s2", -5) + pair("s2", 5, language="hi") +
                   pair("s3", 25, language="hi"))
        report = self.run_analysis(records)
        row = next(r for r in report["language_interactions"] if r["metric"] == "gold_pct")
        self.assertEqual(row["estimate"], 10)
        self.assertEqual(row["ci_low"], 10)
        self.assertEqual(row["ci_high"], 10)
        self.assertEqual(row["matched_pairs"], 2)
        self.assertEqual(row["unmatched_language_tasks"], 1)

    def test_language_interaction_does_not_match_different_calendar_pairs(self):
        records = pair(calendar="a") + pair(language="hi", calendar="b")
        report = self.run_analysis(records)
        self.assertTrue(all(row["matched_pairs"] == 0 for row in report["language_interactions"]))

    def test_language_roles_not_conflated(self):
        report = self.run_analysis(pair(role="investor") + pair(language="hi", role="advisor"))
        self.assertEqual(report["language_interactions"], [])

    def test_human_comparison_matches_tasks_and_bootstraps_subjects_scenarios(self):
        models = pair("s1", 15) + pair("s2", 15)
        humans = []
        for subject in ("a", "b"):
            humans.extend(pair("s1", 5, subject=subject) + pair("s2", 5, subject=subject))
        report = self.run_analysis(models, humans)
        row = next(r for r in report["human_comparisons"] if r["metric"] == "gold_pct")
        self.assertEqual(row["estimate"], 10)
        self.assertEqual(row["n_subjects"], 2)
        self.assertEqual(row["n_clusters"], 2)
        self.assertEqual(row["matched_subject_pairs"], 4)
        self.assertEqual(row["ci_low"], 10)
        self.assertEqual(row["bootstrap_valid_draws"], 100)

    def test_human_comparison_rejects_same_ids_with_different_role(self):
        models, humans = pair(), pair(subject="a")
        for record in humans:
            record["trial"]["role"] = "advisor"
        report = self.run_analysis(models, humans)
        self.assertTrue(all(r["matched_subject_pairs"] == 0 for r in report["human_comparisons"]))
        self.assertIn("human_task_metadata_mismatch", [r["reason"] for r in report["exclusions"]])

    def test_human_subject_must_supply_both_conditions(self):
        humans = [pair(subject="a")[0], pair(subject="b")[1]]
        report = self.run_analysis(pair(), humans)
        self.assertTrue(all(r["matched_subject_pairs"] == 0 for r in report["human_comparisons"]))

    def test_duplicate_human_observations_are_rejected(self):
        humans = pair(subject="a")
        humans.append(copy.deepcopy(humans[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate human"):
            self.run_analysis(pair(), humans)

    def test_empty_input_produces_header_only_tables(self):
        report = self.run_analysis([])
        self.assertEqual(report["effects"], [])
        for name in ("effects", "human_comparisons", "missingness", "exclusions"):
            self.assertEqual(len((self.root / "analysis" / f"{name}.csv").read_text().splitlines()), 1)

    def test_import_humans_retains_invalid_rows_without_imputation(self):
        trials = [r["trial"] for r in pair()]
        trial_path = self.root / "trials.jsonl"
        write_jsonl(trial_path, trials)
        rows = [{"trial_id": trial["trial_id"], "subject_id": "anon-1", **payload()} for trial in trials]
        rows[1]["gold_pct"] = "refused"
        csv_path = self.csv_file("human.csv", HUMAN_FIELDS, rows)
        output = self.root / "imported.jsonl"
        summary = import_humans(trial_path, csv_path, output)
        self.assertEqual(summary["status_counts"]["ok"], 1)
        self.assertEqual(summary["status_counts"]["invalid"], 1)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertIsNone(records[1]["result"]["payload"])
        self.assertEqual(records[1]["result"]["source_cells"]["gold_pct"], "refused")

    def test_import_humans_header_only_and_duplicate_and_unknown_ids(self):
        trials = [r["trial"] for r in pair()]
        trial_path = self.root / "trials.jsonl"
        write_jsonl(trial_path, trials)
        csv_path = self.csv_file("human.csv", HUMAN_FIELDS, [])
        output = self.root / "imported.jsonl"
        self.assertEqual(import_humans(trial_path, csv_path, output)["records"], 0)
        row = {"trial_id": trials[0]["trial_id"], "subject_id": "anon-1", **payload()}
        self.csv_file("human.csv", HUMAN_FIELDS, [row, row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            import_humans(trial_path, csv_path, output)
        self.csv_file("human.csv", HUMAN_FIELDS, [{**row, "trial_id": "unknown"}])
        with self.assertRaisesRegex(ValueError, "unknown trial_id"):
            import_humans(trial_path, csv_path, output)

    def test_import_human_schema_violations_become_invalid(self):
        trials = [r["trial"] for r in pair()]
        trial_path = self.root / "trials.jsonl"
        write_jsonl(trial_path, trials)
        base = {"trial_id": trials[0]["trial_id"], "subject_id": "a", **payload()}
        rows = [{**base, "subject_id": "a", "gold_pct": "NaN"},
                {**base, "subject_id": "b", "buy_now": "yes"},
                {**base, "subject_id": "c", "timing_days": "1.5"},
                {**base, "subject_id": "d", "cash_pct": "90"}]
        path = self.csv_file("human.csv", HUMAN_FIELDS, rows)
        result = import_humans(trial_path, path, self.root / "imported.jsonl")
        self.assertEqual(result["status_counts"]["invalid"], 4)

    def test_context_preserves_units_sources_and_annotations(self):
        rows = [{"date": "2020-01-01", "series": "fixture-gold", "value": 100,
                 "unit": "INR/gram", "source": "test-source-a", "event": "hypothetical"},
                {"date": "2020-01-02", "series": "fixture-gold", "value": 110,
                 "unit": "INR/gram", "source": "test-source-a", "event": ""},
                {"date": "2020-01-01", "series": "fixture-gold", "value": 5,
                 "unit": "tonnes", "source": "test-source-a", "event": ""},
                {"date": "2020-01-01", "series": "fixture-gold", "value": 99,
                 "unit": "INR/gram", "source": "test-source-b", "event": ""}]
        path = self.csv_file("context.csv", CONTEXT_FIELDS, rows)
        report = contextual_series(path, self.root / "context")
        self.assertEqual(len(report["series"]), 3)
        selected = next(r for r in report["series"] if r["n_observations"] == 2)
        self.assertEqual(selected["absolute_change"], 10)
        self.assertEqual(selected["unit"], "INR/gram")
        self.assertIn("not individual", report["interpretation"])
        self.assertIn("hypothetical", (self.root / "context" / "contextual_data.csv").read_text())

    def test_context_header_only_is_valid(self):
        path = self.csv_file("context.csv", CONTEXT_FIELDS, [])
        report = contextual_series(path, self.root / "context")
        self.assertEqual(report["observations"], 0)
        self.assertEqual(report["series"], [])
        self.assertEqual(len((self.root / "context" / "contextual_data.csv").read_text().splitlines()), 1)

    def test_context_rejects_nonfinite_duplicate_and_non_iso_date(self):
        row = {"date": "2020-01-01", "series": "fixture", "value": 5,
               "unit": "INR", "source": "test", "event": ""}
        for rows in ([{**row, "value": "NaN"}], [row, row], [{**row, "date": "20200101"}]):
            with self.subTest(rows=rows):
                path = self.csv_file("context.csv", CONTEXT_FIELDS, rows)
                with self.assertRaises(ValueError):
                    contextual_series(path, self.root / "context")

    def test_csv_missing_headers_is_error(self):
        path = self.csv_file("context.csv", ("date", "value"), [])
        with self.assertRaisesRegex(ValueError, "missing CSV columns"):
            contextual_series(path, self.root / "context")


if __name__ == "__main__":
    unittest.main()
