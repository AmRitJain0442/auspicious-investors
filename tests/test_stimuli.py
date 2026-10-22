"""Unit fixtures are synthetic, not calendar verification or human observations."""
import copy
import hashlib
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from auspicious.stimuli import build_trials, export_review, validate_review, validate_trial_review

ROOT = Path(__file__).resolve().parents[1]


class StimuliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = json.loads((ROOT / "examples/pilot.json").read_text(encoding="utf-8"))
        self.config["data_path"] = str(ROOT / "data/stimuli.json")
        self.config_path = self.directory / "config.json"
        self.output_path = self.directory / "trials.jsonl"

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def build(self):
        self.write_config()
        summary = build_trials(self.config_path, self.output_path)
        rows = [json.loads(line) for line in self.output_path.read_text(encoding="utf-8").splitlines()]
        return summary, rows

    def approve(self):
        self.write_config()
        review_path = self.directory / "review.json"
        export_review(self.config_path, review_path)
        manifest = json.loads(review_path.read_text(encoding="utf-8"))
        for subject in manifest["subjects"]:
            subject.update(decision="approved", reviewer="UNIT TEST ATTESTATION, NOT A HUMAN REVIEW", approved_at="2026-01-01T00:00:00+00:00", notes="Synthetic test approval only")
        review_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        self.config["review_path"] = str(review_path)
        return review_path

    def test_pilot_deterministic_complete_and_matched(self):
        summary, rows = self.build()
        first_bytes = self.output_path.read_bytes()
        _, second_rows = self.build()
        self.assertEqual(first_bytes, self.output_path.read_bytes())
        self.assertEqual(rows, second_rows)
        self.assertEqual({r["language"] for r in rows}, {"en", "hi", "ta", "gu", "bn"})
        self.assertEqual({r["festival"] for r in rows}, {"dhanteras", "akshaya_tritiya", "muhurat", "pitru_paksha"})
        self.assertEqual({r["role"] for r in rows}, {"investor", "advisor"})
        groups = defaultdict(list)
        for trial in rows:
            groups[trial["pair_id"]].append(trial)
            self.assertEqual(trial["review_status"], "draft")
            self.assertEqual(trial["prompt_sha256"], hashlib.sha256(trial["prompt"].encode()).hexdigest())
            self.assertEqual(trial["event_valence"], "avoidance" if trial["festival"] == "pitru_paksha" else "auspicious")
            self.assertEqual(trial["design_trial_count"], len(rows))
        self.assertEqual(len({r["trial_id"] for r in rows}), len(rows))
        digest = hashlib.sha256(json.dumps(sorted(r["trial_id"] for r in rows), separators=(",", ":")).encode()).hexdigest()
        for trial in rows:
            self.assertEqual(trial["design_trial_ids_sha256"], digest)
        for pair in groups.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual({r["condition"] for r in pair}, {"event", "neutral"})
            self.assertEqual(pair[0]["economics"], pair[1]["economics"])

    def test_multiple_scenarios_personas_paraphrases_and_cluster_split(self):
        self.config.update(languages=["en"], festivals=["dhanteras"], roles=["investor"], personas=["observant", "nonobservant"], scenarios=["emergency_buffer", "education_payment", "home_deposit"], paraphrases=["direct", "planning"], cues=["festival_named", "explicit_belief"], replicates=2)
        _, rows = self.build()
        self.assertEqual(len(rows), 96)
        splits = defaultdict(set)
        for trial in rows:
            splits[trial["cluster_id"]].add(trial["split"])
        self.assertEqual(splits, {"emergency_buffer": {"train"}, "education_payment": {"dev"}, "home_deposit": {"test"}})

    def test_config_and_text_change_ids(self):
        _, rows = self.build()
        original = {r["trial_id"] for r in rows}
        self.config["seed"] += 1
        _, changed = self.build()
        self.assertFalse(original & {r["trial_id"] for r in changed})
        data = json.loads((ROOT / "data/stimuli.json").read_text(encoding="utf-8"))
        data["languages"]["en"]["introduction"] += " Additional reviewed context."
        path = self.directory / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.config["data_path"] = str(path)
        _, edited = self.build()
        self.assertFalse({r["trial_id"] for r in changed} & {r["trial_id"] for r in edited})

    def test_confirmatory_requires_hash_bound_approval(self):
        self.config["mode"] = "confirmatory"
        with self.assertRaisesRegex(ValueError, "approvals"):
            self.build()
        self.approve()
        summary, rows = self.build()
        self.assertEqual(summary["review_status"], "approved")
        validate_review(self.config_path)
        for row in rows:
            validate_trial_review(row)
        tampered = copy.deepcopy(rows[0])
        tampered["prompt"] += " altered"
        tampered["prompt_sha256"] = hashlib.sha256(tampered["prompt"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "prompt"):
            validate_trial_review(tampered)
        tampered = copy.deepcopy(rows[0])
        tampered["design_trial_count"] -= 2
        with self.assertRaisesRegex(ValueError, "design size"):
            validate_trial_review(tampered)
        tampered = copy.deepcopy(rows[0])
        tampered["review"]["materials"]["language:" + tampered["language"]]["introduction"] += " altered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_trial_review(tampered)
        self.config["seed"] += 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.build()

    def test_pending_review_cannot_be_boolean_bypassed(self):
        self.config["mode"] = "confirmatory"
        self.write_config()
        path = self.directory / "review.json"
        export_review(self.config_path, path)
        self.config["review_path"] = str(path)
        with self.assertRaisesRegex(ValueError, "approval required"):
            self.build()
        manifest = json.loads(path.read_text())
        for subject in manifest["subjects"]:
            subject["decision"] = "approved"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "reviewer"):
            self.build()

    def test_date_only_requires_supplied_calendar(self):
        self.config["cues"] = ["date_only"]
        with self.assertRaisesRegex(ValueError, "calendar_path"):
            self.build()
        self.config["calendar_path"] = str(ROOT / "data/calendar-empty.json")
        with self.assertRaisesRegex(ValueError, "No verified"):
            self.build()

    def calendar_fixture(self):
        # Deliberately arbitrary dates and example.invalid URLs test the metadata
        # protocol only. They MUST NOT be used as festival or neutral-date evidence.
        source = {"url": "https://example.invalid/UNIT-TEST-NOT-EVIDENCE", "title": "SYNTHETIC UNIT TEST, NOT CALENDAR EVIDENCE", "accessed_on": "2026-01-01", "evidence": "Fixture only; no actual date verification"}
        return {"schema_version": 1, "pairs": [{"id": "synthetic-test-only", "festival": "dhanteras", "event_date": "2026-01-01", "neutral_date": "2026-01-02", "locality": "Synthetic test locality", "timezone": "Asia/Kolkata", "verified": True, "verified_by": "UNIT TEST FIXTURE; NOT A RESEARCHER", "verified_at": "2026-01-01T00:00:00+00:00", "verification_method": "Simulated metadata only, no calendar claim", "event_sources": [source], "neutral_sources": [source], "matching": {key: "Synthetic test metadata, not observational evidence" for key in ("selection_rule", "weekday_difference", "seasonality", "market_open_status", "other_calendar_events", "residual_confounders")}}]}

    def test_calendar_metadata_validation_and_date_only_contrast(self):
        calendar = self.calendar_fixture()
        calendar_path = self.directory / "calendar.json"
        calendar_path.write_text(json.dumps(calendar))
        self.config.update(languages=["en"], festivals=["dhanteras"], roles=["investor"], cues=["date_only"], calendar_path=str(calendar_path))
        _, rows = self.build()
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["calendar_pair"], calendar["pairs"][0])
            self.assertNotIn("Dhanteras", row["prompt"])
            self.assertNotIn("Synthetic test metadata", row["prompt"])
        differing_lines = [(a, b) for a, b in zip(rows[0]["prompt"].splitlines(), rows[1]["prompt"].splitlines()) if a != b]
        self.assertEqual(len(differing_lines), 1)
        self.approve()
        _, approved = self.build()
        for row in approved:
            validate_trial_review(row)
        self.config.pop("review_path")
        for key, bad_value, message in (("verified", False, "verified=true"), ("event_date", "2026-02-30", "ISO"), ("timezone", "not-a-timezone", "IANA"), ("event_sources", [], "event_sources"), ("locality", "", "locality"), ("matching", {}, "matching")):
            invalid = copy.deepcopy(calendar)
            invalid["pairs"][0][key] = bad_value
            calendar_path.write_text(json.dumps(invalid))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                self.build()

    def test_duplicate_and_unknown_dimensions_fail(self):
        self.config["languages"] = ["en", "en"]
        with self.assertRaisesRegex(ValueError, "duplicates"):
            self.build()
        self.config["languages"] = ["xx"]
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self.build()


if __name__ == "__main__":
    unittest.main()
