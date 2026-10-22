"""Unit fixtures are synthetic, not calendar verification or human observations."""
import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from auspicious.core import validate_trials
from auspicious.stimuli import _hash, build_trials, export_review, load_review_context, sidecar_path, validate_review, validate_trial_review

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ["no_calendar", "occasion_control", "purchase_control"]
FESTIVAL_NAMES = ("Dhanteras", "Akshaya Tritiya", "Muhurat", "Pitru Paksha")


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

    def small_controls(self, **overrides):
        self.config.update(languages=["en"], festivals=["dhanteras", "pitru_paksha"], roles=["investor"], personas=["unspecified"], scenarios=["emergency_buffer", "annual_bonus"], paraphrases=["direct"], cues=["festival_named", "explicit_belief"], controls=list(CONTROLS), templates=["t1", "t2"])
        self.config.update(overrides)

    def test_pilot_deterministic_complete_and_matched(self):
        summary, rows = self.build()
        first_bytes = self.output_path.read_bytes()
        first_sidecar = sidecar_path(self.output_path).read_bytes()
        _, second_rows = self.build()
        self.assertEqual(first_bytes, self.output_path.read_bytes())
        self.assertEqual(first_sidecar, sidecar_path(self.output_path).read_bytes())
        self.assertEqual(rows, second_rows)
        self.assertEqual({r["language"] for r in rows}, {"en", "hi", "ta", "gu", "bn"})
        self.assertEqual({r["festival"] for r in rows}, {"dhanteras", "akshaya_tritiya", "muhurat", "pitru_paksha"})
        self.assertEqual({r["role"] for r in rows}, {"investor", "advisor"})
        self.assertEqual({r["template_id"] for r in rows}, {"t1"})
        groups = defaultdict(list)
        for trial in rows:
            groups[trial["pair_id"]].append(trial)
            self.assertEqual(trial["review_status"], "draft")
            self.assertEqual(set(trial["review"]), {"context_sha256", "translation_status", "calendar_status"})
            self.assertEqual(trial["review"]["translation_status"], "unreviewed_draft")
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
        # Distinct-prompt reporting: 5 languages x 2 roles neutral texts repeat across 4 festivals.
        self.assertEqual({key: summary[key] for key in ("trials", "pairs", "distinct_prompts", "distinct_neutral_prompts", "scenarios", "templates", "controls", "conditions")},
                         {"trials": 80, "pairs": 40, "distinct_prompts": 50, "distinct_neutral_prompts": 10, "scenarios": 1, "templates": 1, "controls": 0, "conditions": 2})
        self.assertEqual(summary["distinct_prompts_by_condition"], {"event": 40, "neutral": 10})

    def test_review_sidecar_replaces_embedded_materials(self):
        summary, rows = self.build()
        context = load_review_context(self.output_path)
        self.assertEqual(context["schema_version"], 2)
        self.assertIsNone(context["manifest"])
        self.assertEqual(context["context_sha256"], _hash({"materials": context["materials"], "manifest": context["manifest"]}))
        self.assertEqual(summary["review_context_sha256"], context["context_sha256"])
        self.assertEqual(summary["review_context_path"], str(sidecar_path(self.output_path)))
        self.assertEqual({row["review"]["context_sha256"] for row in rows}, {context["context_sha256"]})
        self.assertLess(max(len(json.dumps(row, ensure_ascii=False).encode()) for row in rows), 15000)
        self.assertIn("design", context["materials"])
        for language in ("en", "hi", "ta", "gu", "bn"):
            self.assertIn(f"language:{language}:t1", context["materials"])
        self.assertEqual(sidecar_path("a/b/trials.jsonl"), Path("a/b/trials.jsonl.review.json"))
        # Sidecars are always written and are self-verifying.
        self.assertIsNone(load_review_context(self.directory / "absent.jsonl"))
        text = sidecar_path(self.output_path).read_text(encoding="utf-8")
        sidecar_path(self.output_path).write_text(text.replace("synthetic financial decision exercise", "synthetic financial decision EXERCISE"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_review_context(self.output_path)

    def test_control_conditions_complete_pairs_and_wording(self):
        self.small_controls()
        summary, rows = self.build()
        self.assertEqual(len(rows), 2 * 2 * 2 * 2 * 5)  # festivals x scenarios x cues x templates pairs, 5 conditions
        self.assertEqual(summary["pairs"], 16)
        self.assertEqual(summary["controls"], 3)
        self.assertEqual(summary["conditions"], 5)
        self.assertEqual(summary["condition_ids"], ["event", "neutral", *CONTROLS])
        self.assertEqual(summary["templates"], 2)
        groups = defaultdict(dict)
        for row in rows:
            groups[row["pair_id"]][row["condition"]] = row
        for pair in groups.values():
            self.assertEqual(set(pair), {"event", "neutral", *CONTROLS})
            self.assertEqual(len({r["prompt_sha256"] for r in pair.values()}), 5)
            for field in ("economics", "festival", "event_valence", "template_id", "scenario_id", "cue", "persona_id", "split"):
                self.assertEqual(len({json.dumps(r[field]) for r in pair.values()}), 1)
            for control in CONTROLS:
                prompt = pair[control]["prompt"]
                for name in FESTIVAL_NAMES:
                    self.assertNotIn(name, prompt)
                self.assertNotIn("uspicious", prompt)  # also covers "inauspicious"
            self.assertNotIn("calendar", pair["no_calendar"]["prompt"].lower())
            self.assertNotIn("Cultural", pair["no_calendar"]["prompt"])
            # The economic disclaimer is kept in every arm, including the unlabeled control.
            for condition, row in pair.items():
                self.assertIn("exactly as", row["prompt"], condition)
                self.assertIn("trading hours", row["prompt"], condition)
            self.assertIn("Republic Day", pair["occasion_control"]["prompt"])
            if pair["event"]["event_valence"] == "avoidance":
                self.assertIn("deferring large purchases" if pair["event"]["template_id"] == "t1" else "put off large purchases", pair["purchase_control"]["prompt"])
            else:
                self.assertIn("buying gold", pair["purchase_control"]["prompt"])
            for condition in ("event", "neutral", "occasion_control", "purchase_control"):
                self.assertEqual("Cultural" in pair[condition]["prompt"], pair["event"]["cue"] == "explicit_belief", condition)
        # Unlabeled/occasion controls repeat across festivals; purchase text depends on valence only.
        self.assertEqual(summary["distinct_prompts_by_condition"], {"event": 16, "neutral": 8, "no_calendar": 4, "occasion_control": 8, "purchase_control": 16})
        self.assertEqual(summary["distinct_neutral_prompts"], 8)
        self.assertEqual(summary["distinct_prompts"], 16 + 8 + 4 + 8 + 16)

    def test_condition_set_and_controls_are_part_of_design_identity(self):
        self.small_controls(controls=["no_calendar"])
        _, one = self.build()
        self.small_controls(controls=["no_calendar", "occasion_control"])
        _, two = self.build()
        self.assertEqual({r["config_sha256"] for r in one} & {r["config_sha256"] for r in two}, set())
        self.assertNotEqual({r["design_trial_ids_sha256"] for r in one}, {r["design_trial_ids_sha256"] for r in two})
        self.config.pop("controls")
        self.config.pop("templates")
        _, defaults = self.build()
        explicit = dict(self.config, templates=["t1"], controls=[])
        self.config = explicit
        _, spelled = self.build()
        self.assertEqual([r["trial_id"] for r in defaults], [r["trial_id"] for r in spelled])

    def test_templates_are_distinct_variants_with_draft_provenance(self):
        self.small_controls(controls=[], cues=["festival_named"], festivals=["dhanteras"], scenarios=["emergency_buffer"], languages=["en", "hi", "ta", "gu", "bn"])
        _, rows = self.build()
        by_key = defaultdict(dict)
        for row in rows:
            by_key[(row["language"], row["condition"])][row["template_id"]] = row
        self.assertEqual(len(by_key), 10)
        for (language, condition), pair in by_key.items():
            self.assertEqual(set(pair), {"t1", "t2"})
            self.assertNotEqual(pair["t1"]["prompt"], pair["t2"]["prompt"], (language, condition))
            self.assertNotEqual(pair["t1"]["pair_id"], pair["t2"]["pair_id"])
            self.assertEqual(pair["t1"]["economics"], pair["t2"]["economics"])
            # Same numeric economics appear in both templates.
            for key, value in pair["t1"]["economics"].items():
                self.assertIn(f": {value}", pair["t1"]["prompt"])
                self.assertIn(f": {value}", pair["t2"]["prompt"])
        context = load_review_context(self.output_path)
        for language in ("en", "hi", "ta", "gu", "bn"):
            t1, t2 = context["materials"][f"language:{language}:t1"], context["materials"][f"language:{language}:t2"]
            self.assertIn("NOT independent human authorship", t2["provenance"]["authorship"])
            self.assertIn("requires independent bilingual review", t2["provenance"]["authorship"])
            self.assertEqual(t2["provenance"]["status"], "unreviewed_draft")
            self.assertEqual(t2["provenance"]["variant_of"], "t1")
            self.assertNotEqual(t1["layout"], t2["layout"])
            self.assertEqual(sorted(t1["layout"]), sorted(t2["layout"]))

    def test_prompt_clarity_fixes_in_all_templates(self):
        self.small_controls(controls=[], cues=["festival_named"], festivals=["dhanteras"], scenarios=["emergency_buffer"])
        _, rows = self.build()
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertIn("held entirely in cash", row["prompt"])
            self.assertIn("expected annual gold return you are assuming in your decision", row["prompt"])
            self.assertNotIn("your expected annual gold return", row["prompt"])
            self.assertIn("ONE gold unit", row["prompt"])
        data = json.loads((ROOT / "data/stimuli.json").read_text(encoding="utf-8"))
        for language, templates in data["languages"].items():
            for template_id, text in templates.items():
                self.assertIn("expected_return_pct", text["decision"], (language, template_id))
                self.assertIn("willingness_to_pay_inr", text["decision"], (language, template_id))
                if language == "en":
                    self.assertNotIn("your expected annual gold return", text["decision"])

    def test_scenario_diversity_and_splits(self):
        data = json.loads((ROOT / "data/stimuli.json").read_text(encoding="utf-8"))
        scenarios = data["scenarios"]
        self.assertEqual(len(scenarios), 12)
        splits = defaultdict(list)
        for scenario_id, scenario in scenarios.items():
            splits[scenario["split"]].append(scenario_id)
            economics = scenario["economics"]
            self.assertEqual(len(economics), 13)
            for language, templates in data["languages"].items():
                for template_id, text in templates.items():
                    self.assertTrue(text["scenarios"][scenario_id].strip(), (language, template_id, scenario_id))
        self.assertEqual({key: len(value) for key, value in splits.items()}, {"train": 4, "dev": 4, "test": 4})
        for original, expected in (("emergency_buffer", 200000), ("retirement_growth", 1000000), ("education_payment", 500000), ("income_reserve", 350000), ("home_deposit", 800000), ("diversification", 600000)):
            self.assertEqual(scenarios[original]["economics"]["capital_inr"], expected)
        gold_above_equity = [s for s, v in scenarios.items() if v["economics"]["gold_expected_annual_pct"] > v["economics"]["equity_expected_annual_pct"]]
        large_liquidity = [s for s, v in scenarios.items() if v["economics"]["liquidity_need_inr"] >= 0.5 * v["economics"]["capital_inr"]]
        long_horizon = [s for s, v in scenarios.items() if v["economics"]["horizon_years"] >= 10]
        self.assertGreaterEqual(len(gold_above_equity), 2)
        self.assertGreaterEqual(len(large_liquidity), 2)
        self.assertGreaterEqual(len(long_horizon), 2)
        for scenario in scenarios.values():
            economics = scenario["economics"]
            self.assertLessEqual(economics["liquidity_need_inr"], economics["capital_inr"])
        self.assertGreaterEqual(len({v["economics"]["gold_unit_price_inr"] for v in scenarios.values()}), 3)

    def test_shipped_configs_design_sizes(self):
        for name, cue in (("full-config.json", "festival_named"), ("belief-config.json", "explicit_belief")):
            config = json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))
            self.assertEqual(config["cues"], [cue])
            self.assertEqual(config["controls"], CONTROLS)
            self.assertEqual(config["templates"], ["t1", "t2"])
            self.assertEqual(config["paraphrases"], ["direct"])
            self.assertEqual(config["personas"], ["unspecified", "observant", "nonobservant"])
            self.assertEqual(len(config["languages"]), 5)
            self.assertEqual(len(config["festivals"]), 4)
            self.assertEqual(len(config["roles"]), 2)
            self.assertEqual(len(config["scenarios"]), 12)
            self.assertEqual(config["replicates"], 1)
            pairs = 5 * 4 * 2 * 3 * 12 * 1 * 1 * 2
            self.assertEqual(pairs, 2880)
            self.assertEqual(pairs * (2 + len(config["controls"])), 14400)
        # Build the English-only slice of the shipped primary configuration end to end.
        config = json.loads((ROOT / "data/full-config.json").read_text(encoding="utf-8"))
        config.update(data_path=str(ROOT / "data/stimuli.json"), languages=["en"])
        self.config = config
        summary, rows = self.build()
        self.assertEqual((summary["trials"], summary["pairs"], summary["scenarios"], summary["templates"], summary["controls"], summary["conditions"]), (2880, 576, 12, 2, 3, 5))
        self.assertEqual(summary["distinct_prompts_by_condition"], {"event": 576, "neutral": 144, "no_calendar": 144, "occasion_control": 144, "purchase_control": 288})
        self.assertEqual(summary["distinct_prompts"], 1296)
        self.assertEqual(summary["distinct_neutral_prompts"], 144)
        self.assertEqual(len(rows), 2880)
        splits = {row["cluster_id"]: row["split"] for row in rows}
        self.assertEqual(sorted(Counter(splits.values()).items()), [("dev", 4), ("test", 4), ("train", 4)])

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
        data["languages"]["en"]["t1"]["introduction"] += " Additional reviewed context."
        path = self.directory / "data.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.config["data_path"] = str(path)
        _, edited = self.build()
        self.assertFalse({r["trial_id"] for r in changed} & {r["trial_id"] for r in edited})

    def test_confirmatory_requires_hash_bound_approval(self):
        self.small_controls()
        self.config["mode"] = "confirmatory"
        with self.assertRaisesRegex(ValueError, "approvals"):
            self.build()
        review_path = self.approve()
        manifest = json.loads(review_path.read_text(encoding="utf-8"))
        subjects = {subject["subject_id"] for subject in manifest["subjects"]}
        self.assertEqual(subjects, {"design", "language:en:t1", "language:en:t2"})  # templates are approval scopes
        summary, rows = self.build()
        self.assertEqual(summary["review_status"], "approved")
        validate_review(self.config_path)
        context = load_review_context(self.output_path)
        self.assertEqual(context["manifest"]["config_sha256"], rows[0]["config_sha256"])
        for row in rows:
            validate_trial_review(row, context)
            self.assertEqual(row["review"]["translation_status"], "reviewer_attested")
        validate_trials(rows, confirmatory=True, review_context=context)
        validate_review(rows[0], context)
        # Missing sidecar / context is a hard failure.
        with self.assertRaisesRegex(ValueError, "sidecar"):
            validate_trial_review(rows[0], None)
        with self.assertRaisesRegex(ValueError, "sidecar"):
            validate_trials(rows, confirmatory=True)
        tampered = copy.deepcopy(rows[0])
        tampered["prompt"] += " altered"
        tampered["prompt_sha256"] = hashlib.sha256(tampered["prompt"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "prompt"):
            validate_trial_review(tampered, context)
        tampered = copy.deepcopy(rows[0])
        tampered["design_trial_count"] -= 5
        with self.assertRaisesRegex(ValueError, "design size"):
            validate_trial_review(tampered, context)
        tampered = copy.deepcopy(rows[0])
        tampered["review"]["context_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "context hash"):
            validate_trial_review(tampered, context)
        tampered = copy.deepcopy(rows[0])
        tampered["review"]["translation_status"] = "unreviewed_draft"
        with self.assertRaisesRegex(ValueError, "not approved"):
            validate_trial_review(tampered, context)
        tampered = copy.deepcopy(rows[0])
        tampered["condition"] = "occasion_control" if rows[0]["condition"] != "occasion_control" else "neutral"
        with self.assertRaises(ValueError):
            validate_trial_review(tampered, context)
        # Edited materials: context hash fails first; with a recomputed context hash, the manifest binding fails.
        edited = copy.deepcopy(context)
        edited["materials"]["language:en:t2"]["introduction"] += " altered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_trial_review(rows[0], edited)
        edited["context_sha256"] = _hash({"materials": edited["materials"], "manifest": edited["manifest"]})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_trial_review(rows[0], edited)
        # Draft sidecar (no manifest) never validates.
        draft = copy.deepcopy(context)
        draft["manifest"] = None
        draft["context_sha256"] = _hash({"materials": draft["materials"], "manifest": None})
        with self.assertRaisesRegex(ValueError, "manifest"):
            validate_trial_review(rows[0], draft)
        # A subset of the reviewed design (whole pairs removed, or an incomplete condition set) is rejected.
        victim = rows[0]["pair_id"]
        with self.assertRaisesRegex(ValueError, "full reviewed design"):
            validate_trials([r for r in rows if r["pair_id"] != victim], confirmatory=True, review_context=context)
        with self.assertRaises(ValueError):
            validate_trials([r for r in rows if not (r["pair_id"] == victim and r["condition"] == "purchase_control")], confirmatory=True, review_context=context)
        # Config drift after approval, including the condition set and templates, is rejected at build time.
        self.config["seed"] += 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.build()
        self.config["seed"] -= 1
        self.config["controls"] = ["no_calendar"]
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.build()
        self.config["controls"] = list(CONTROLS)
        self.config["templates"] = ["t1"]
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.build()

    def test_unapproved_template_scope_blocks_build(self):
        self.small_controls()
        self.config["mode"] = "confirmatory"
        path = self.approve()
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for subject in manifest["subjects"]:
            if subject["subject_id"] == "language:en:t2":
                subject["decision"] = "pending"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "approval required: language:en:t2"):
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

    def test_controls_other_than_no_calendar_rejected_with_date_only(self):
        for controls in (["occasion_control"], ["purchase_control"], ["no_calendar", "occasion_control"]):
            self.config.update(cues=["date_only"], controls=controls)
            with self.subTest(controls=controls), self.assertRaisesRegex(ValueError, "not defined for cue date_only"):
                self.build()
        self.config.update(cues=["festival_named", "date_only"], controls=["purchase_control"])
        with self.assertRaisesRegex(ValueError, "not defined for cue date_only"):
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
        # no_calendar is the only control defined for date_only: it omits the date but stays paired.
        self.config["controls"] = ["no_calendar"]
        _, with_control = self.build()
        self.assertEqual(len(with_control), 3)
        blank = next(r for r in with_control if r["condition"] == "no_calendar")
        self.assertNotIn("2026-01", blank["prompt"])
        self.assertNotIn("Synthetic test locality", blank["prompt"])
        self.assertEqual(blank["calendar_pair"], calendar["pairs"][0])
        self.config["controls"] = []
        self.approve()
        _, approved = self.build()
        context = load_review_context(self.output_path)
        for row in approved:
            validate_trial_review(row, context)
            self.assertEqual(row["review"]["calendar_status"], "researcher_attested_not_independently_verified")
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
        self.config["languages"] = ["en"]
        for key, value, message in (("templates", ["t9"], "Unsupported templates"), ("templates", [], "templates"), ("templates", ["t1", "t1"], "duplicates"), ("controls", ["nope"], "Unsupported controls"), ("controls", ["no_calendar", "no_calendar"], "duplicates"), ("controls", "no_calendar", "controls")):
            config = dict(self.config, **{key: value})
            self.config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, message):
                build_trials(self.config_path, self.output_path)

    def test_template_data_validation(self):
        data = json.loads((ROOT / "data/stimuli.json").read_text(encoding="utf-8"))
        path = self.directory / "data.json"
        self.config.update(languages=["en"], data_path=str(path), templates=["t1", "t2"], scenarios=["emergency_buffer", "parent_care"])
        for mutate, message in (
            (lambda d: d["languages"]["en"]["t2"].pop("provenance"), "provenance"),
            (lambda d: d["languages"]["en"]["t2"]["provenance"].pop("authorship"), "authorship"),
            (lambda d: d["languages"]["en"]["t2"].update(layout=["introduction"]), "layout"),
            (lambda d: d["languages"]["en"]["t2"]["purchase_labels"].pop("avoidance"), "purchase_labels"),
            (lambda d: d["languages"]["en"]["t2"].update(named_frame="{wrong}"), "placeholders"),
            (lambda d: d["languages"]["en"]["t2"]["scenarios"].pop("parent_care"), "parent_care"),
            (lambda d: d["languages"]["en"].pop("t2"), "template"),
        ):
            candidate = copy.deepcopy(data)
            mutate(candidate)
            path.write_text(json.dumps(candidate), encoding="utf-8")
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.build()


if __name__ == "__main__":
    unittest.main()
