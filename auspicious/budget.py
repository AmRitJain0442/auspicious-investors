"""Durable whole-batch cost reservations; no request is sent before reservation."""
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path

from .core import atomic_text, canonical, strict_json


class BudgetExhausted(ValueError):
    pass


def micros(usd):
    return int((Decimal(str(usd)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


class BudgetLedger:
    """A GCS lease serializes workers; generation checks protect every ledger update.

    Reserve all possible HTTP attempts for the next whole-pair chunk before calls.
    Release only verifiable savings. A crash's outstanding reservation is charged
    in full on resume, including responses lost after the last raw-data checkpoint.
    This accounts for model tokens at declared prices, not infrastructure or tax.
    """
    def __init__(self, blob, path, spec_sha256, limit_usd):
        self.blob, self.path = blob, Path(path)
        self.generation = 0
        self.state = {"schema_version": 1, "spec_sha256": spec_sha256,
                      "limit_microusd": micros(limit_usd), "accounted_microusd": 0,
                      "held_microusd": 0, "settled_chunks": 0, "crash_reservations_charged": 0}
        if blob.exists():
            blob.reload()
            self.generation = int(blob.generation)
            prior = strict_json(blob.download_as_text())
            if any(prior.get(k) != self.state[k] for k in ("schema_version", "spec_sha256", "limit_microusd")):
                raise ValueError("Budget ledger identity/limit changed; use a new run identifier")
            if any(type(prior.get(k)) is not int or prior[k] < 0 for k in
                   ("accounted_microusd", "held_microusd", "settled_chunks", "crash_reservations_charged")):
                raise ValueError("Invalid budget ledger amounts")
            self.state = prior
            if prior["held_microusd"]:
                self.state["accounted_microusd"] += self.state["held_microusd"]
                self.state["held_microusd"] = 0
                self.state["crash_reservations_charged"] += 1
        self._write()

    def _write(self):
        raw = canonical(self.state)
        # Persist remotely FIRST. A failed reservation cannot authorize API calls.
        self.blob.upload_from_string(raw, content_type="application/json", if_generation_match=self.generation)
        self.generation = int(self.blob.generation)
        atomic_text(self.path, raw + "\n")

    def reserve(self, trials, upper_bound_usd):
        if self.state["held_microusd"]:
            raise ValueError("A budget chunk is already reserved")
        amount = micros(upper_bound_usd) + 2 * len(trials) + 1  # cover per-trial outward rounding
        if amount + self.state["accounted_microusd"] > self.state["limit_microusd"]:
            raise BudgetExhausted("Next whole-pair chunk would exceed the model allowance")
        ids = sorted(t["trial_id"] for t in trials)
        self.state.update(held_microusd=amount, held_trial_count=len(ids),
                          held_trial_ids_sha256=hashlib.sha256(canonical(ids).encode()).hexdigest())
        self._write()

    def settle(self, records, spec, estimate):
        ids = sorted(r["trial"]["trial_id"] for r in records)
        if (len(ids) != self.state.get("held_trial_count") or
                hashlib.sha256(canonical(ids).encode()).hexdigest() != self.state.get("held_trial_ids_sha256")):
            raise ValueError("Budget settlement does not match every reserved trial")
        price = spec["prices_usd_per_million"]
        accounted, uncertain = 0, 0
        for record in records:
            result = record["result"]
            usage, log = result.get("usage", {}), result.get("attempt_log", [])
            complete = (isinstance(usage, dict) and bool(log)
                        and len(log) == result.get("attempts")
                        and all(type(a.get("http_status")) is int for a in log))
            successes = sum(a.get("http_status") == 200 for a in log)
            complete = complete and successes <= 1
            if successes:
                complete = complete and all(type(usage.get(k)) is int and usage[k] >= 0
                                           for k in ("promptTokenCount", "candidatesTokenCount"))
                complete = complete and type(usage.get("thoughtsTokenCount", 0)) is int and usage.get("thoughtsTokenCount", 0) >= 0
            if not complete:
                # Ambiguous transport/parse failure may have consumed tokens: no refund.
                accounted += micros(estimate([record["trial"]], spec)["model_cost_upper_bound_usd"]) + 1
                uncertain += 1
            elif successes:
                amount = (Decimal(usage["promptTokenCount"]) * Decimal(str(price["input"]))
                          + Decimal(usage["candidatesTokenCount"] + usage.get("thoughtsTokenCount", 0))
                          * Decimal(str(price["output"])))
                accounted += int(amount.to_integral_value(rounding=ROUND_CEILING))
            # Documented non-200 HTTP attempts are not charged by Vertex AI.
        reserved = self.state["held_microusd"]
        if accounted > reserved:
            # Do not authorize further chunks if actual metadata contradicts the bound.
            self.state["accounted_microusd"] += accounted
            self.state["held_microusd"] = 0
            self._write()
            raise ValueError("Observed token charges exceeded the declared conservative reservation")
        self.state["accounted_microusd"] += accounted
        self.state.update(held_microusd=0, settled_chunks=self.state["settled_chunks"] + 1,
                          last_chunk_accounted_microusd=accounted, last_chunk_uncertain_trials=uncertain)
        self._write()
        return dict(self.state)
