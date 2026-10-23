"""Offline cost-accounting and cloud-resume regressions, without paid calls."""
from collections import Counter
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from auspicious.budget import BudgetExhausted, BudgetLedger
from auspicious.cloud import estimate_cost, run_cloud
from auspicious.core import read_jsonl
from auspicious.gemini import GeminiProvider
from test_cloud import Blob, Bucket, storage
from test_runtime import payload


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "budget.json"
        self.blob = Blob()
        self.spec = json.loads(Path("cloud/run-template.json").read_text())
        self.spec.update(project="research-project")
        self.trials = [{"trial_id": "one", "prompt": "synthetic"}, {"trial_id": "two", "prompt": "synthetic"}]

    def ledger(self, limit=1):
        return BudgetLedger(self.blob, self.path, "synthetic-spec", limit)

    def record(self, trial, usage=None, log=None, attempts=1):
        return {"trial": trial, "result": {"attempts": attempts,
            "usage": usage or {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 3},
            "attempt_log": log if log is not None else [{"http_status": 200}]}}

    def test_durable_reservation_and_verified_refund(self):
        ledger = self.ledger()
        ledger.reserve(self.trials, .05)
        self.assertEqual(json.loads(self.blob.data)["held_trial_count"], 2)
        state = ledger.settle([self.record(t) for t in self.trials], self.spec, estimate_cost)
        self.assertEqual(state["accounted_microusd"], 324)
        self.assertEqual(state["held_microusd"], 0)

    def test_refuses_reservation_beyond_allowance_and_wrong_settlement(self):
        ledger = self.ledger(.01)
        with self.assertRaises(BudgetExhausted):
            ledger.reserve(self.trials, .05)
        self.assertEqual(ledger.state["held_microusd"], 0)
        ledger.reserve(self.trials, .001)
        with self.assertRaisesRegex(ValueError, "every reserved trial"):
            ledger.settle([self.record(self.trials[0])], self.spec, estimate_cost)

    def test_unknown_outcomes_keep_full_conservative_charge(self):
        ledger = self.ledger()
        bound = estimate_cost(self.trials, self.spec)["model_cost_upper_bound_usd"]
        ledger.reserve(self.trials, bound)
        records = [self.record(t, log=[{"transport_error": "TimeoutError"}]) for t in self.trials]
        state = ledger.settle(records, self.spec, estimate_cost)
        self.assertEqual(state["last_chunk_uncertain_trials"], 2)
        self.assertGreater(state["accounted_microusd"], 10000)
        self.assertLess(state["accounted_microusd"], state["limit_microusd"])

    def test_documented_non_200_http_attempts_are_not_refunded_as_unknown(self):
        ledger = self.ledger()
        ledger.reserve(self.trials, .05)
        records = [self.record(t, log=[{"http_status": 429}, {"http_status": 500}], attempts=2) for t in self.trials]
        state = ledger.settle(records, self.spec, estimate_cost)
        self.assertEqual(state["accounted_microusd"], 0)

    def test_crash_recovery_charges_outstanding_reservation_without_releasing_it(self):
        ledger = self.ledger()
        ledger.reserve(self.trials, .05)
        reserved = ledger.state["held_microusd"]
        recovered = self.ledger()
        self.assertEqual(recovered.state["accounted_microusd"], reserved)
        self.assertEqual(recovered.state["held_microusd"], 0)
        self.assertEqual(recovered.state["crash_reservations_charged"], 1)
        with self.assertRaisesRegex(ValueError, "identity/limit changed"):
            self.ledger(2)

    def test_failed_remote_write_cannot_return_a_reservation(self):
        ledger = self.ledger()
        with patch.object(self.blob, "upload_from_string", side_effect=OSError("synthetic persistence failure")):
            with self.assertRaises(OSError):
                ledger.reserve(self.trials, .05)


@unittest.skipIf(storage is None, "Google Cloud optional extra unavailable")
class GuardedWorkerTests(unittest.TestCase):
    def specification(self, budget=1):
        spec = json.loads(Path("cloud/run-template.json").read_text())
        spec.update(project="research-project", design_config="examples/pilot-controls.json",
                    budget_mode="guarded_chunks", budget_chunk_trials=5, model_budget_usd=budget,
                    exploratory_split_tables=True,
                    collection_plan={"status": "synthetic prospective descriptive fixture"})
        return spec

    def reply(self, trial):
        return {"status": "ok", "raw_text": json.dumps(payload()), "attempts": 1,
                "attempt_log": [{"http_status": 200}],
                "usage": {"promptTokenCount": 100, "candidatesTokenCount": 20, "thoughtsTokenCount": 3}}

    def test_whole_pair_chunks_complete_bind_plan_and_do_not_repeat_on_resume(self):
        bucket = Bucket()
        spec = self.specification()
        bucket.blob("runs/fixture/run.json").upload_from_string(json.dumps(spec))
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(storage, "Client", return_value=SimpleNamespace(bucket=lambda _: bucket)), \
             patch.object(GeminiProvider, "prepare"), \
             patch.object(GeminiProvider, "generate_trial", side_effect=self.reply) as calls:
            result = run_cloud("bucket", "fixture", folder)
            self.assertEqual(result["state"], "completed")
            self.assertEqual(result["summary"]["stored"], 10)
            self.assertEqual(result["summary"]["executed"], 10)
            self.assertEqual(result["budget_ledger"]["settled_chunks"], 2)
            records = read_jsonl(Path(folder) / "model/records.jsonl")
            self.assertTrue(all(r["collection"]["protocol_sha256"] for r in records))
            self.assertEqual(sorted(Counter(r["trial"]["pair_id"] for r in records).values()), [5, 5])
            run_cloud("bucket", "fixture", folder)
            self.assertEqual(calls.call_count, 10)

    def test_guard_stops_before_paid_calls_and_archives_explicit_incomplete_status(self):
        bucket = Bucket()
        bucket.blob("runs/fixture/run.json").upload_from_string(json.dumps(self.specification(.000001)))
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(storage, "Client", return_value=SimpleNamespace(bucket=lambda _: bucket)), \
             patch.object(GeminiProvider, "prepare"), \
             patch.object(GeminiProvider, "generate_trial") as calls:
            result = run_cloud("bucket", "fixture", folder)
            self.assertEqual(result["state"], "budget_exhausted")
            self.assertFalse(result["summary"]["complete"])
            self.assertEqual(result["summary"]["stored"], 0)
            self.assertTrue(bucket.blob("runs/fixture/results.zip").exists())
            calls.assert_not_called()
