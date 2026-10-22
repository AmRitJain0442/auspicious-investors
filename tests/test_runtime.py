import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from auspicious.core import normalize_result, read_jsonl, run_trials, validate_trials, write_jsonl


def payload():
    return {"gold_pct": 20, "equity_pct": 50, "cash_pct": 30, "buy_now": True,
            "timing_days": 0, "willingness_to_pay_inr": 0, "expected_return_pct": 5,
            "rationale": "Keep emergency liquidity and diversify."}


def trials(conditions=("event", "neutral"), pair_ids=("pair",)):
    rows = []
    for pair_id in pair_ids:
        for condition in conditions:
            prompt = f"Financial decision {pair_id} {condition}"
            rows.append({"trial_id": hashlib.sha256(prompt.encode()).hexdigest(),
                         "pair_id": pair_id, "cluster_id": "scenario", "split": "test",
                         "language": "en", "festival": "dhanteras", "condition": condition,
                         "event_valence": "auspicious", "cue": "festival_named", "role": "investor",
                         "persona_id": "unspecified", "scenario_id": "scenario", "paraphrase_id": "p1",
                         "template_id": "t1", "replicate": 0, "prompt": prompt,
                         "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                         "review_status": "draft", "review": {}})
    return rows


class CountingProvider:
    def __init__(self, status="ok"):
        self.identity = {"provider": "test-only", "requested_model": "fixture", "generation": {"temperature": 0}}
        self.calls = 0
        self.status = status

    def generate(self, prompt):
        self.calls += 1
        return {"status": self.status, "raw_text": json.dumps(payload()) if self.status == "ok" else "not JSON",
                "error": "Transport failed" if self.status == "error" else None}


class ResponseValidationTests(unittest.TestCase):
    def test_invalid_allocations_are_not_repaired(self):
        for values in ((20, 50, 40), (-1, 51, 50), (True, 49, 50)):
            data = payload()
            data.update(zip(("gold_pct", "equity_pct", "cash_pct"), values))
            result = normalize_result({"status": "ok", "raw_text": json.dumps(data)})
            self.assertEqual(result["status"], "invalid")
            self.assertIsNone(result["payload"])

    def test_timing_contradiction_and_nonfinite_rejected(self):
        for change in ({"timing_days": 2}, {"expected_return_pct": float("nan")}, {"timing_days": True}):
            data = payload() | change
            result = normalize_result({"status": "ok", "raw_text": json.dumps(data)})
            self.assertEqual(result["status"], "invalid")

    def test_scenario_timing_limit_is_enforced_without_repair(self):
        trial = {"economics": {"max_delay_days": 30}}
        at_limit = payload() | {"buy_now": False, "timing_days": 30}
        beyond = at_limit | {"timing_days": 31}
        self.assertEqual(normalize_result({"raw_text": json.dumps(at_limit)}, trial=trial)["status"], "ok")
        result = normalize_result({"raw_text": json.dumps(beyond)}, trial=trial)
        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["payload"])
        self.assertEqual(json.loads(result["raw_text"])["timing_days"], 31)

    def test_blocked_json_is_not_counted_as_a_decision(self):
        result = normalize_result({"status": "blocked", "raw_text": json.dumps(payload())})
        self.assertEqual(result["status"], "blocked")
        self.assertIsNone(result["payload"])

    def test_duplicate_keys_rejected(self):
        raw = json.dumps(payload()).replace('"gold_pct": 20', '"gold_pct": 20, "gold_pct": 20')
        self.assertEqual(normalize_result({"status": "ok", "raw_text": raw})["status"], "invalid")


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "trials.jsonl"
        write_jsonl(self.path, trials())

    def tearDown(self):
        self.temp.cleanup()

    def test_resume_does_not_resample_completed_trials(self):
        provider = CountingProvider()
        run_trials(self.path, self.root / "run", provider, concurrency=1)
        summary = run_trials(self.path, self.root / "run", provider, concurrency=1)
        self.assertEqual(provider.calls, 2)
        self.assertEqual(summary["executed"], 0)
        records = read_jsonl(self.root / "run" / "records.jsonl")
        self.assertEqual({r["trial"]["condition"] for r in records}, {"event", "neutral"})
        provider.identity["generation"]["temperature"] = 1
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            run_trials(self.path, self.root / "run", provider)

    def test_retry_errors_never_retries_invalid_responses(self):
        provider = CountingProvider(status="invalid")
        run_trials(self.path, self.root / "run", provider, concurrency=1)
        summary = run_trials(self.path, self.root / "run", provider, concurrency=1, retry_errors=True)
        self.assertEqual(summary["executed"], 0)
        self.assertEqual(summary["counts"], {"invalid": 2})

    def test_transport_errors_require_explicit_retry(self):
        provider = CountingProvider(status="error")
        run_trials(self.path, self.root / "run", provider, concurrency=1)
        provider.status = "ok"
        self.assertEqual(run_trials(self.path, self.root / "run", provider)["executed"], 0)
        summary = run_trials(self.path, self.root / "run", provider, retry_errors=True)
        self.assertEqual(summary["counts"], {"ok": 2})

    def test_incomplete_or_mismatched_pairs_rejected(self):
        rows = trials()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_trials(rows[:1])
        rows[1]["language"] = "hi"
        with self.assertRaisesRegex(ValueError, "mismatches"):
            validate_trials(rows)

    def test_cluster_split_leakage_rejected(self):
        rows = trials()
        rows[1]["split"] = "train"
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_trials(rows)

    def test_control_arms_are_part_of_the_pair(self):
        rows = trials(("event", "neutral", "no_calendar", "purchase_control"))
        validate_trials(rows)
        rows[3]["template_id"] = "t2"
        with self.assertRaisesRegex(ValueError, "mismatches template_id in purchase_control"):
            validate_trials(rows)

    def test_every_pair_must_carry_the_same_controls(self):
        rows = trials(("event", "neutral", "no_calendar"), ("a",)) + trials(("event", "neutral"), ("b",))
        with self.assertRaisesRegex(ValueError, "inconsistent condition sets"):
            validate_trials(rows)

    def test_limit_keeps_whole_pairs_with_all_conditions(self):
        rows = trials(("event", "neutral", "no_calendar"), ("a", "b", "c"))
        path = self.root / "controls.jsonl"
        write_jsonl(path, rows)
        summary = run_trials(path, self.root / "limited", CountingProvider(), concurrency=1, limit=7)
        self.assertEqual(summary["selected"], 6)
        stored = read_jsonl(self.root / "limited" / "records.jsonl")
        by_pair = {}
        for record in stored:
            by_pair.setdefault(record["trial"]["pair_id"], set()).add(record["trial"]["condition"])
        self.assertEqual(len(by_pair), 2)
        self.assertTrue(all(c == {"event", "neutral", "no_calendar"} for c in by_pair.values()))
        with self.assertRaisesRegex(ValueError, "limit must be >= 3"):
            run_trials(path, self.root / "tiny", CountingProvider(), limit=2)


if __name__ == "__main__":
    unittest.main()
