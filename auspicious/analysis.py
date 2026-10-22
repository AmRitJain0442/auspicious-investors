"""Complete-case, scenario-clustered analyses; contextual series are descriptive only.

All contrasts are event minus neutral. Repeated prompts are averaged within
scenario, not counted as independent observations. No outcomes are imputed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import fmean

from .core import load_records, normalize_result, read_jsonl, validate_trials, write_jsonl

METRICS = (
    "gold_pct", "equity_pct", "cash_pct", "buy_now", "timing_days",
    "willingness_to_pay_inr", "expected_return_pct",
)
UNITS = {
    "gold_pct": "percentage_points", "equity_pct": "percentage_points",
    "cash_pct": "percentage_points", "buy_now": "probability_difference",
    "timing_days": "days", "willingness_to_pay_inr": "INR",
    "expected_return_pct": "percentage_points_annual_expected_return",
}
PAYLOAD_FIELDS = (*METRICS, "rationale")
HUMAN_FIELDS = ("trial_id", "subject_id", *PAYLOAD_FIELDS)
CONTEXT_FIELDS = ("date", "series", "value", "unit", "source", "event")
GROUP_FIELDS = (
    "provider", "requested_model", "model_version", "generation_settings",
    "language", "festival", "cue", "role", "persona_id",
)
TASK_FIELDS = (
    "cluster_id", "scenario_id", "split", "festival", "cue", "role",
    "persona_id", "paraphrase_id", "replicate", "event_valence",
)
PAIR_FIELDS = (*TASK_FIELDS, "language")
OPTIONAL_TASK_FIELDS = ("calendar_pair_id",)
STATUSES = ("ok", "invalid", "blocked", "error")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _csv_write(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv_read(path, fields):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: missing or duplicate CSV headers")
        missing = set(fields) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row or any(row[field] is None for field in fields):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            rows.append((line, row))
        return rows


def _normalize_payload(result, trial):
    """Validate stored payloads too, rather than trusting an 'ok' label."""
    if result.get("status") not in STATUSES:
        return {**result, "status": "invalid", "payload": None,
                "error": "Unknown or missing result status"}
    if result["status"] != "ok":
        return result
    candidate = dict(result)
    try:
        if isinstance(candidate.get("payload"), dict):
            candidate["raw_text"] = _canonical(candidate["payload"])
        return normalize_result(candidate, trial=trial)
    except (ValueError, TypeError, OverflowError) as exc:
        return {**result, "status": "invalid", "payload": None, "error": str(exc)}


def _group(trial, result):
    generation = result.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object")
    for name in ("provider", "requested_model"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f"missing {name}")
    version = result.get("model_version")
    if version is not None and (not isinstance(version, str) or not version):
        raise ValueError("model_version must be a nonempty string or null")
    for field in GROUP_FIELDS[4:]:
        if not isinstance(trial.get(field), str) or not trial[field]:
            raise ValueError(f"missing or invalid trial {field}")
    return (result["provider"], result["requested_model"], version,
            _canonical(generation), *(trial[field] for field in GROUP_FIELDS[4:]))


def _group_row(key):
    return dict(zip(GROUP_FIELDS, key))


def _percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _rng(seed, key):
    digest = hashlib.sha256(_canonical([seed, key]).encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _interval(draws):
    return (_percentile(draws, .025), _percentile(draws, .975)) if draws else (None, None)


def _cluster_values(pairs, metric):
    # Equal subject weighting within each scenario, then equal scenario weighting.
    cells = defaultdict(list)
    for pair in pairs:
        cells[(pair["cluster"], pair["subject"])].append(pair["diff"][metric])
    clusters = defaultdict(list)
    for (cluster, _subject), values in cells.items():
        clusters[cluster].append(fmean(values))
    return {cluster: fmean(values) for cluster, values in clusters.items()}


def _estimate(values, bootstrap, rng):
    numbers = [values[key] for key in sorted(values)]
    if not numbers:
        return None, None, None
    draws = []
    if len(numbers) >= 2 and bootstrap:
        draws = [fmean(rng.choices(numbers, k=len(numbers))) for _ in range(bootstrap)]
    low, high = _interval(draws)
    return fmean(numbers), low, high


def _prepare(records, label):
    buckets = defaultdict(list)
    counts = defaultdict(Counter)
    exclusions = []
    warnings = set()
    duplicates = defaultdict(list)
    candidates = []
    rejected_pairs = set()
    seen_pairs = defaultdict(set)
    for index, record in enumerate(records, 1):
        trial = record.get("trial") if isinstance(record, dict) else None
        result = record.get("result") if isinstance(record, dict) else None
        group = None
        pair_key = None
        try:
            if not isinstance(trial, dict) or not isinstance(result, dict):
                raise ValueError("record requires trial and result objects")
            group = _group(trial, result)
            subject = result.get("subject_id")
            if group[0] == "human" and (not isinstance(subject, str) or not subject.strip()):
                raise ValueError("human record requires subject_id")
            if subject is not None and (not isinstance(subject, str) or not subject.strip()):
                raise ValueError("subject_id must be a nonempty string")
            if group[0] != "human" and subject is not None:
                raise ValueError("subject_id is reserved for human observations")
            if not isinstance(trial.get("pair_id"), str) or not trial["pair_id"]:
                raise ValueError("missing or invalid trial pair_id")
            pair_key = (group, subject, trial["pair_id"])
            seen_pairs[group].add((subject, trial["pair_id"]))
            for field in (*PAIR_FIELDS, "pair_id", "trial_id"):
                value = trial.get(field)
                if field == "replicate":
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError("replicate must be a nonnegative integer")
                elif not isinstance(value, str) or not value:
                    raise ValueError(f"missing or invalid trial {field}")
            for field in OPTIONAL_TASK_FIELDS:
                value = trial.get(field)
                if value is not None and (not isinstance(value, str) or not value):
                    raise ValueError(f"invalid trial {field}")
            if trial.get("condition") not in ("event", "neutral"):
                raise ValueError("condition must be event or neutral")
            normalized = _normalize_payload(result, trial)
            counts[group][normalized["status"]] += 1
            if group[2] is None:
                warnings.add("Unresolved model_version is kept in its own stratum; cross-run checkpoint identity cannot be established for that stratum.")
            item = {"trial": trial, "result": normalized, "group": group,
                    "subject": subject, "index": index, "pair_key": pair_key}
            candidates.append(item)
            duplicates[(group[:4], subject, trial["trial_id"])].append(item)
        except (ValueError, TypeError, KeyError) as exc:
            exclusions.append({"source": label, "record_index": index,
                               "pair_id": trial.get("pair_id", "") if isinstance(trial, dict) else "",
                               "reason": f"malformed_record: {exc}"})
            if pair_key is not None:
                rejected_pairs.add(pair_key)
            if group is not None:
                counts[group]["malformed"] += 1
    for items in duplicates.values():
        if len(items) > 1:
            for item in items:
                rejected_pairs.add(item["pair_key"])
            exclusions.append({"source": label, "record_index": ",".join(str(item["index"]) for item in items),
                               "pair_id": items[0]["trial"]["pair_id"],
                               "reason": "duplicate_trial_observation: all affected pairs excluded"})
    # A nominal pair may not cross model versions, generations, or design strata.
    nominal = defaultdict(set)
    for item in candidates:
        nominal[(item["group"][:2], item["subject"], item["trial"]["pair_id"])].add(item["group"])
        buckets[item["pair_key"]].append(item)
    for groups in nominal.values():
        if len(groups) > 1:
            # Complete pairs in separate versions are valid separate experiments;
            # incomplete cross-version fragments remain incomplete, never combined.
            warnings.add("A pair_id occurs in multiple model/design strata; strata are never pooled or paired across versions/settings.")
    complete = defaultdict(list)
    coverage = []
    for pair_key, items in buckets.items():
        group, subject, pair_id = pair_key
        reason = None
        conditions = Counter(item["trial"]["condition"] for item in items)
        if pair_key in rejected_pairs:
            reason = "duplicate_or_malformed_record_in_pair"
        elif conditions != {"event": 1, "neutral": 1}:
            reason = "incomplete_or_duplicate_condition"
        elif any(item["result"]["status"] != "ok" for item in items):
            reason = "non_ok_pair"
        elif any(items[0]["trial"].get(field) != items[1]["trial"].get(field)
                 for field in (*PAIR_FIELDS, *OPTIONAL_TASK_FIELDS)):
            reason = "inconsistent_pair_metadata"
        if reason:
            exclusions.append({"source": label, "record_index": ",".join(str(item["index"]) for item in items),
                               "pair_id": pair_id, "reason": reason})
            continue
        by_condition = {item["trial"]["condition"]: item for item in items}
        event, neutral = by_condition["event"], by_condition["neutral"]
        diff = {metric: float(event["result"]["payload"][metric]) -
                float(neutral["result"]["payload"][metric]) for metric in METRICS}
        if not all(math.isfinite(value) for value in diff.values()):
            exclusions.append({"source": label, "record_index": ",".join(str(item["index"]) for item in items),
                               "pair_id": pair_id, "reason": "nonfinite_pair_difference"})
            continue
        complete[group].append({"pair_id": pair_id, "subject": subject,
                                "cluster": event["trial"]["cluster_id"], "diff": diff,
                                "event": event["trial"], "neutral": neutral["trial"]})
    # Re-labelling a task/pair must not make a duplicate observation independent.
    for group, pairs in complete.items():
        semantic = defaultdict(list)
        for pair in pairs:
            semantic[_task_key(pair)].append(pair)
        ambiguous = {key for key, values in semantic.items() if len(values) > 1}
        if ambiguous:
            complete[group] = [pair for pair in pairs if _task_key(pair) not in ambiguous]
            for key in sorted(ambiguous, key=_canonical):
                exclusions.append({"source": label, "record_index": "",
                                   "pair_id": ",".join(pair["pair_id"] for pair in semantic[key]),
                                   "reason": "duplicate_semantic_task: all affected pairs excluded"})
    for group in sorted(seen_pairs, key=_canonical):
        n = len(complete[group])
        total = len(seen_pairs[group])
        coverage.append({**_group_row(group), "source": label,
                         "observed_pairs": total, "complete_pairs": n,
                         "excluded_pairs": total - n,
                         "complete_pair_coverage": n / total if total else None,
                         "n_clusters": len({pair["cluster"] for pair in complete[group]})})
    missingness = []
    for group in sorted(counts, key=_canonical):
        total = sum(counts[group].values())
        for status in (*STATUSES, "malformed"):
            n = counts[group][status]
            missingness.append({**_group_row(group), "source": label, "status": status,
                                "count": n, "denominator": total,
                                "fraction": n / total if total else None})
    if exclusions:
        warnings.add(f"{label}: {len(exclusions)} explicit exclusion entries; inspect exclusions.csv. No missing or invalid outcome was replaced with zero.")
    return complete, coverage, missingness, exclusions, warnings


def _effects(groups, bootstrap, seed, source):
    rows = []
    for group in sorted(groups, key=_canonical):
        pairs = groups[group]
        for metric in METRICS:
            values = _cluster_values(pairs, metric)
            effect, low, high = _estimate(values, bootstrap, _rng(seed, [source, group, metric]))
            rows.append({**_group_row(group), "source": source, "metric": metric,
                         "unit": UNITS[metric], "contrast": "event_minus_neutral",
                         "estimate": effect, "ci_low": low, "ci_high": high,
                         "n_pairs": len(pairs), "n_clusters": len(values),
                         "n_subjects": len({p["subject"] for p in pairs if p["subject"] is not None}),
                         "ci_method": "scenario_cluster_percentile" if low is not None else "unavailable",
                         "ci_unavailable_reason": "" if low is not None else ("bootstrap_disabled" if not bootstrap else "fewer_than_two_clusters")})
    return rows


def _task_key(pair):
    return (pair["subject"], *(pair["event"].get(field)
                              for field in (*TASK_FIELDS, *OPTIONAL_TASK_FIELDS)))


def _interactions(groups, bootstrap, seed, source, exclusions):
    rows = []
    for group in sorted(groups, key=_canonical):
        if group[4] == "en":
            continue
        english = (*group[:4], "en", *group[5:])
        if english not in groups:
            continue
        left, right = defaultdict(list), defaultdict(list)
        for pair in groups[group]:
            left[_task_key(pair)].append(pair)
        for pair in groups[english]:
            right[_task_key(pair)].append(pair)
        matched = []
        ambiguous = 0
        for key in sorted(left.keys() & right.keys(), key=_canonical):
            if len(left[key]) != 1 or len(right[key]) != 1:
                ambiguous += 1
                exclusions.append({"source": source, "record_index": "",
                                   "pair_id": ",".join(p["pair_id"] for p in left[key] + right[key]),
                                   "reason": "ambiguous_language_task_match"})
                continue
            target, baseline = left[key][0], right[key][0]
            matched.append({**target, "diff": {m: target["diff"][m] - baseline["diff"][m] for m in METRICS}})
        for metric in METRICS:
            values = _cluster_values(matched, metric)
            effect, low, high = _estimate(values, bootstrap, _rng(seed, ["interaction", source, group, metric]))
            rows.append({**_group_row(group), "source": source, "reference_language": "en",
                         "metric": metric, "unit": UNITS[metric],
                         "contrast": "language_event_effect_minus_english_event_effect",
                         "estimate": effect, "ci_low": low, "ci_high": high,
                         "matched_pairs": len(matched), "n_clusters": len(values),
                         "unmatched_language_tasks": len(left.keys() - right.keys()),
                         "unmatched_english_tasks": len(right.keys() - left.keys()),
                         "ambiguous_task_matches": ambiguous,
                         "ci_method": "paired_scenario_cluster_percentile" if low is not None else "unavailable"})
    return rows


def _same_task(left, right):
    return all(left.get(field) == right.get(field) for field in
               (*PAIR_FIELDS, *OPTIONAL_TASK_FIELDS, "pair_id", "trial_id", "condition", "prompt_sha256", "prompt"))


def _human_comparisons(models, humans, bootstrap, seed, exclusions):
    rows = []
    human_index = defaultdict(list)
    for pairs in humans.values():
        for pair in pairs:
            human_index[(pair["event"]["trial_id"], pair["neutral"]["trial_id"])].append(pair)
    for group in sorted(models, key=_canonical):
        matches = []
        matched_model_pairs = set()
        for model in models[group]:
            ids = (model["event"]["trial_id"], model["neutral"]["trial_id"])
            for human in human_index.get(ids, []):
                if not (_same_task(model["event"], human["event"]) and
                        _same_task(model["neutral"], human["neutral"])):
                    exclusions.append({"source": "comparison", "record_index": "",
                                       "pair_id": model["pair_id"], "reason": "human_task_metadata_mismatch"})
                    continue
                matched_model_pairs.add(ids)
                matches.append({**human, "diff": {m: model["diff"][m] - human["diff"][m] for m in METRICS}})
        for metric in METRICS:
            cells = defaultdict(list)
            for pair in matches:
                cells[(pair["cluster"], pair["subject"])].append(pair["diff"][metric])
            cells = {key: fmean(values) for key, values in cells.items()}
            clusters = sorted({key[0] for key in cells})
            subjects = sorted({key[1] for key in cells})
            estimate_values = _cluster_values(matches, metric)
            estimate = fmean(estimate_values.values()) if estimate_values else None
            draws = []
            if len(clusters) >= 2 and len(subjects) >= 2 and bootstrap:
                rng = _rng(seed, ["human", group, metric])
                for _ in range(bootstrap):
                    cluster_counts = Counter(rng.choices(clusters, k=len(clusters)))
                    subject_counts = Counter(rng.choices(subjects, k=len(subjects)))
                    sampled = []
                    for cluster, cluster_weight in cluster_counts.items():
                        observed = [(value, subject_counts[subject]) for (c, subject), value in cells.items()
                                    if c == cluster and subject_counts[subject]]
                        denominator = sum(weight for _, weight in observed)
                        if denominator:
                            mean = sum(value * weight for value, weight in observed) / denominator
                            sampled.extend([mean] * cluster_weight)
                    if sampled:
                        draws.append(fmean(sampled))
            low, high = _interval(draws)
            rows.append({**_group_row(group), "metric": metric, "unit": UNITS[metric],
                         "contrast": "model_event_effect_minus_human_event_effect",
                         "estimate": estimate, "ci_low": low, "ci_high": high,
                         "matched_model_pairs": len(matched_model_pairs),
                         "matched_subject_pairs": len(matches), "n_clusters": len(clusters),
                         "n_subjects": len(subjects), "bootstrap_valid_draws": len(draws),
                         "ci_method": "crossed_subject_scenario_percentile" if low is not None else "unavailable",
                         "ci_unavailable_reason": "" if low is not None else ("bootstrap_disabled" if not bootstrap else "requires_two_subjects_and_two_clusters"),
                         "unmatched_model_pairs": len(models[group]) - len(matched_model_pairs)})
    return rows


def analyze(records_path, output_dir, *, bootstrap=2000, seed=42, human_path=None):
    """Write complete-case effects, matched interactions, and explicit exclusions.

    CIs are unadjusted 95% percentile intervals. Coverage denominators describe
    observed records/pairs, not trials absent from the supplied result files.
    """
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    records = load_records(records_path)
    if any(isinstance(r, dict) and isinstance(r.get("result"), dict) and
           r["result"].get("provider") == "human" for r in records):
        raise ValueError("Model records must not contain human providers; use human_path")
    groups, coverage, missingness, exclusions, warnings = _prepare(records, "model")
    effects = _effects(groups, bootstrap, seed, "model")
    interactions = _interactions(groups, bootstrap, seed, "model", exclusions)
    comparisons = []
    record_totals = {"model": len(records)}
    if human_path is not None:
        human_records = load_records(human_path)
        record_totals["human"] = len(human_records)
        if any(not isinstance(r, dict) or not isinstance(r.get("result"), dict) or
               r["result"].get("provider") != "human" for r in human_records):
            raise ValueError("human_path must contain imported human result records only")
        humans, hc, hm, he, hw = _prepare(human_records, "human")
        # A subject/trial observation may appear only once even if a caller relabels
        # its human provider model/version/generation metadata.
        identifiers = [(r["result"].get("subject_id"), r["trial"].get("trial_id"))
                       for r in human_records if isinstance(r.get("trial"), dict)
                       and isinstance(r["result"].get("subject_id"), str)
                       and isinstance(r["trial"].get("trial_id"), str)]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate human (subject_id, trial_id) observations; comparisons would pseudoreplicate")
        coverage.extend(hc)
        missingness.extend(hm)
        exclusions.extend(he)
        warnings.update(hw)
        effects.extend(_effects(humans, bootstrap, seed, "human"))
        interactions.extend(_interactions(humans, bootstrap, seed, "human", exclusions))
        comparisons = _human_comparisons(groups, humans, bootstrap, seed, exclusions)
    if exclusions:
        warnings.add("Exclusions are explicit; consult exclusions and pair_coverage before interpreting complete-case effects.")
    status_totals = []
    for source, denominator in record_totals.items():
        counts = Counter()
        for row in missingness:
            if row["source"] == source:
                counts[row["status"]] += row["count"]
        counts["malformed"] += denominator - sum(counts.values())
        for status in (*STATUSES, "malformed"):
            status_totals.append({"source": source, "status": status,
                                  "count": counts[status], "denominator": denominator,
                                  "fraction": counts[status] / denominator if denominator else None})
    report = {
        "schema_version": 1, "bootstrap": bootstrap, "seed": seed,
        "confidence_level": .95, "effects": effects,
        "language_interactions": interactions, "human_comparisons": comparisons,
        "missingness": missingness, "pair_coverage": coverage, "exclusions": exclusions,
        "record_status_totals": status_totals,
        "warnings": sorted(warnings),
        "methods": {
            "estimand": "Mean scenario-level event-minus-neutral complete-pair difference; average repetitions within subject/scenario, then subjects within scenario, then scenarios equally.",
            "missingness": "No imputation; pair coverage uses observed pair IDs only. Grouped missingness covers attributable records; record_status_totals includes unstratifiable malformed records. Trials entirely absent from result files are not identifiable.",
            "effects_ci": "Resample scenarios with replacement; no independent-generation or paraphrase pseudoreplication. Human-only effects CIs condition on the observed subjects.",
            "language_interactions": "Subtract English paired effects only on identical matched task dimensions and subjects, then resample shared scenarios.",
            "human_comparison": "Exact trial IDs and task/role metadata required. Matched complete-case model-minus-human paired effects; independently resample subjects and scenarios (crossed bootstrap). Sparse crossed designs can yield empty draws, which are counted and omitted. Subject/scenario selection and shared-task dependence limit population generalization.",
            "model_identity": "Provider, requested model, returned model version and generation settings are separate strata; unresolved versions cannot prove cross-run checkpoint identity.",
            "inference": "Unadjusted exploratory 95% percentile CIs; no multiple-testing correction or causal identification from observational human/market data.",
            "units": UNITS,
        },
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "effects": (effects, (*GROUP_FIELDS, "source", "metric", "unit", "contrast", "estimate", "ci_low", "ci_high", "n_pairs", "n_clusters", "n_subjects", "ci_method", "ci_unavailable_reason")),
        "language_interactions": (interactions, (*GROUP_FIELDS, "source", "reference_language", "metric", "unit", "contrast", "estimate", "ci_low", "ci_high", "matched_pairs", "n_clusters", "unmatched_language_tasks", "unmatched_english_tasks", "ambiguous_task_matches", "ci_method")),
        "human_comparisons": (comparisons, (*GROUP_FIELDS, "metric", "unit", "contrast", "estimate", "ci_low", "ci_high", "matched_model_pairs", "matched_subject_pairs", "n_clusters", "n_subjects", "bootstrap_valid_draws", "ci_method", "ci_unavailable_reason", "unmatched_model_pairs")),
        "missingness": (missingness, (*GROUP_FIELDS, "source", "status", "count", "denominator", "fraction")),
        "record_status_totals": (status_totals, ("source", "status", "count", "denominator", "fraction")),
        "pair_coverage": (coverage, (*GROUP_FIELDS, "source", "observed_pairs", "complete_pairs", "excluded_pairs", "complete_pair_coverage", "n_clusters")),
        "exclusions": (exclusions, ("source", "record_index", "pair_id", "reason")),
    }
    report["outputs"] = {name: str(directory / f"{name}.csv") for name in tables}
    report["outputs"]["analysis"] = str(directory / "analysis.json")
    for name, (rows, fields) in tables.items():
        _csv_write(directory / f"{name}.csv", rows, fields)
    _json_write(directory / "analysis.json", report)
    return report


def import_humans(trials_path, csv_path, output_path):
    """Import consented, deidentified observations without inventing missing data.

    CSV booleans are true/false (case insensitive) or 1/0. Invalid answer cells
    become invalid result records; unknown trial IDs and duplicate observations
    are structural errors and abort the import before any output is written.
    """
    trials = read_jsonl(trials_path)
    validate_trials(trials)
    by_id = {trial["trial_id"]: trial for trial in trials}
    rows = _csv_read(csv_path, HUMAN_FIELDS)
    seen = set()
    records = []
    status_counts = Counter()
    for line, row in rows:
        trial_id, subject_id = row["trial_id"].strip(), row["subject_id"].strip()
        if trial_id not in by_id:
            raise ValueError(f"{csv_path}:{line}: unknown trial_id {trial_id!r}")
        if not subject_id:
            raise ValueError(f"{csv_path}:{line}: subject_id must not be empty")
        if (subject_id, trial_id) in seen:
            raise ValueError(f"{csv_path}:{line}: duplicate (subject_id, trial_id)")
        seen.add((subject_id, trial_id))
        raw_cells = {field: row[field] for field in PAYLOAD_FIELDS}
        result = {"provider": "human", "requested_model": "human", "model_version": "self_report",
                  "generation": {}, "usage": {}, "attempts": 1, "subject_id": subject_id,
                  "status": "ok", "payload": None, "raw_text": _canonical(raw_cells), "error": None}
        try:
            answer = {field: float(row[field]) for field in METRICS if field not in ("buy_now", "timing_days")}
            flag = row["buy_now"].strip().lower()
            if flag not in ("true", "false", "1", "0"):
                raise ValueError("buy_now must be true/false or 1/0")
            answer["buy_now"] = flag in ("true", "1")
            answer["timing_days"] = int(row["timing_days"].strip())
            answer["rationale"] = row["rationale"]
            result["raw_text"] = _canonical(answer)
            result = normalize_result(result, trial=by_id[trial_id])
        except (ValueError, TypeError, OverflowError) as exc:
            result.update(status="invalid", payload=None, error=str(exc))
        result["source_row"] = line
        result["source_cells"] = raw_cells
        status_counts[result["status"]] += 1
        records.append({"trial": by_id[trial_id], "result": result})
    write_jsonl(output_path, records)
    return {"records": len(records), "subjects": len({subject for subject, _ in seen}),
            "status_counts": {status: status_counts[status] for status in STATUSES},
            "output": str(output_path), "warnings": ["Invalid answers retained as invalid, never imputed."] if status_counts["invalid"] else []}


def contextual_series(csv_path, output_dir):
    """Import sourced observational series without equating them to decisions.

    Each (series, unit, source) remains separate. Changes are absolute changes in
    the supplied unit, not automatically returns, allocations, or event effects.
    """
    rows = []
    seen = set()
    for line, row in _csv_read(csv_path, CONTEXT_FIELDS):
        item = {field: row[field].strip() for field in CONTEXT_FIELDS}
        try:
            parsed = date.fromisoformat(item["date"])
            if parsed.isoformat() != item["date"]:
                raise ValueError("date must use YYYY-MM-DD")
            value = float(item["value"])
            if not math.isfinite(value):
                raise ValueError("value must be finite")
            for field in ("series", "unit", "source"):
                if not item[field]:
                    raise ValueError(f"{field} must not be empty")
            key = tuple(item[field] for field in ("date", "series", "unit", "source"))
            if key in seen:
                raise ValueError("duplicate (date, series, unit, source) observation")
            seen.add(key)
        except ValueError as exc:
            raise ValueError(f"{csv_path}:{line}: {exc}") from exc
        item["value"] = value
        rows.append(item)
    rows.sort(key=lambda row: tuple(row[field] for field in ("series", "unit", "source", "date")))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["series"], row["unit"], row["source"])].append(row)
    summaries = []
    for (series, unit, source), observations in sorted(groups.items()):
        values = [row["value"] for row in observations]
        summaries.append({"series": series, "unit": unit, "source": source,
                          "n_observations": len(values), "first_date": observations[0]["date"],
                          "last_date": observations[-1]["date"], "first_value": values[0],
                          "last_value": values[-1], "minimum": min(values), "maximum": max(values),
                          "mean": fmean(values), "absolute_change": values[-1] - values[0] if len(values) >= 2 else None})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "observations": len(rows), "series": summaries,
              "interpretation": "Descriptive contextual observations only. Event labels are supplied annotations, not verified dates or causal treatment assignments. Market prices, quantities, imports and returns are not individual allocations, preferences or decisions. No pooling across units/sources, market-human comparison, event study, currency conversion, interpolation or implied returns is performed.",
              "outputs": {"data": str(directory / "contextual_data.csv"),
                          "summary": str(directory / "contextual_summary.csv"),
                          "json": str(directory / "contextual_summary.json")}}
    _csv_write(directory / "contextual_data.csv", rows, CONTEXT_FIELDS)
    _csv_write(directory / "contextual_summary.csv", summaries,
               ("series", "unit", "source", "n_observations", "first_date", "last_date", "first_value", "last_value", "minimum", "maximum", "mean", "absolute_change"))
    _json_write(directory / "contextual_summary.json", report)
    return report
