"""Deterministic, economically paired multilingual stimuli and attributable review gates.

Review manifests are human attestations, not cryptographic proof of reviewer identity.
Calendar sources and translations must be checked by researchers; this module only
checks metadata and content binding, never claims independent calendar verification.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlparse

LANGUAGES = ("en", "hi", "ta", "gu", "bn")
FESTIVALS = ("dhanteras", "akshaya_tritiya", "muhurat", "pitru_paksha")
CUES = ("date_only", "festival_named", "explicit_belief")
ROLES = ("investor", "advisor")
ECONOMIC_KEYS = ("capital_inr", "gold_unit_price_inr", "transaction_fee_inr", "horizon_years", "liquidity_need_inr", "liquidity_due_days", "gold_expected_annual_pct", "equity_expected_annual_pct", "cash_expected_annual_pct", "gold_annual_volatility_pct", "equity_annual_volatility_pct", "cash_annual_volatility_pct", "max_delay_days")
RESPONSE_CONTRACT = ('{"gold_pct": number, "equity_pct": number, "cash_pct": number, '
                     '"buy_now": boolean, "timing_days": integer, '
                     '"willingness_to_pay_inr": number, "expected_return_pct": number, "rationale": string}')


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load(path):
    def bad_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=bad_constant)


def _save(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _relative(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def _selection(config, key, allowed):
    values = config.get(key, list(allowed))
    if not isinstance(values, list) or not values or len(set(values)) != len(values):
        raise ValueError(f"{key} must be a nonempty list without duplicates")
    if not set(values).issubset(allowed):
        raise ValueError(f"Unsupported {key}: {set(values) - set(allowed)}")
    return values


def _nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")


def _timestamp(value, label):
    _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO 8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include timezone")


def _validate_calendar(calendar):
    if not isinstance(calendar, dict) or calendar.get("schema_version") != 1 or not isinstance(calendar.get("pairs"), list):
        raise ValueError("Calendar requires schema_version=1 and pairs list")
    seen = set()
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    for pair in calendar["pairs"]:
        for key in ("id", "locality", "timezone", "verification_method", "verified_by", "verified_at"):
            _nonempty(pair.get(key), f"calendar {key}")
        if pair["id"] in seen:
            raise ValueError("Duplicate calendar pair id")
        seen.add(pair["id"])
        if pair.get("festival") not in FESTIVALS or pair.get("verified") is not True:
            raise ValueError("Calendar pair requires known festival and explicit verified=true attestation")
        _timestamp(pair["verified_at"], "calendar verified_at")
        try:
            ZoneInfo(pair["timezone"])
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("Calendar timezone must be a valid IANA timezone") from exc
        parsed_dates = []
        for key in ("event_date", "neutral_date"):
            try:
                value = date.fromisoformat(pair[key])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Calendar {key} requires ISO YYYY-MM-DD") from exc
            if value.isoformat() != pair[key]:
                raise ValueError(f"Calendar {key} requires ISO YYYY-MM-DD")
            parsed_dates.append(value)
        if parsed_dates[0] == parsed_dates[1]:
            raise ValueError("Calendar event and neutral dates must differ")
        for key in ("event_sources", "neutral_sources"):
            sources = pair.get(key)
            if not isinstance(sources, list) or not sources:
                raise ValueError(f"Calendar requires nonempty {key}")
            for source in sources:
                for field in ("url", "title", "accessed_on", "evidence"):
                    _nonempty(source.get(field), f"calendar source {field}")
                parsed = urlparse(source["url"])
                if parsed.scheme not in ("https", "http") or not parsed.netloc:
                    raise ValueError("Calendar source URL must be HTTP(S)")
                try:
                    date.fromisoformat(source["accessed_on"])
                except ValueError as exc:
                    raise ValueError("Calendar source accessed_on must be ISO date") from exc
        matching = pair.get("matching")
        if not isinstance(matching, dict):
            raise ValueError("Calendar pair requires matching metadata")
        for key in ("selection_rule", "weekday_difference", "seasonality", "market_open_status", "other_calendar_events", "residual_confounders"):
            _nonempty(matching.get(key), f"calendar matching.{key}")


def _prepare(config_path, *, load_review=True):
    config_path = Path(config_path)
    config = _load(config_path)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Config requires schema_version=1")
    allowed = {"schema_version", "data_path", "calendar_path", "review_path", "languages", "festivals", "roles", "personas", "scenarios", "paraphrases", "cues", "replicates", "seed", "mode", "calendar_pair_ids"}
    if set(config) - allowed:
        raise ValueError(f"Unknown config keys: {sorted(set(config) - allowed)}")
    if config.get("mode", "exploratory") not in ("exploratory", "confirmatory"):
        raise ValueError("mode must be exploratory or confirmatory")
    for key, default in (("replicates", 1), ("seed", 42)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or (key == "replicates" and value < 1):
            raise ValueError(f"{key} must be an integer" + (" >=1" if key == "replicates" else ""))
    data = _load(_relative(config_path.parent, config["data_path"]))
    if data.get("schema_version") != 1:
        raise ValueError("Stimuli data requires schema_version=1")
    selections = {
        "languages": _selection(config, "languages", LANGUAGES),
        "festivals": _selection(config, "festivals", FESTIVALS),
        "roles": _selection(config, "roles", ROLES),
        "cues": _selection(config, "cues", CUES),
        "personas": _selection(config, "personas", ("unspecified", "observant", "nonobservant")),
        "scenarios": _selection(config, "scenarios", data["scenarios"]),
        "paraphrases": _selection(config, "paraphrases", ("direct", "planning")),
    }
    for scenario_id, scenario in data["scenarios"].items():
        if scenario.get("split") not in ("train", "dev", "test"):
            raise ValueError(f"Scenario {scenario_id} requires cluster-level split")
        economics = scenario.get("economics", {})
        if set(economics) != set(ECONOMIC_KEYS):
            raise ValueError(f"Scenario {scenario_id}: economic fields must equal {ECONOMIC_KEYS}")
        for key, value in economics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Scenario {scenario_id}.{key}: finite numeric value required")
            if key not in ("gold_expected_annual_pct", "equity_expected_annual_pct", "cash_expected_annual_pct") and value < 0:
                raise ValueError(f"Scenario {scenario_id}.{key}: nonnegative value required")
        if economics["capital_inr"] <= 0 or economics["gold_unit_price_inr"] <= 0 or economics["max_delay_days"] > 365:
            raise ValueError("Scenario capital/price must be positive; delay at most 365")
    for language in selections["languages"]:
        text = data["languages"][language]
        for key in ("introduction", "hypothetical", "neutral", "date_frame", "named_frame", "belief_frame", "economics_intro", "decision", "constraints", "rationale_instruction"):
            _nonempty(text.get(key), f"{language}.{key}")
        for group, ids in (("roles", selections["roles"]), ("personas", selections["personas"]), ("paraphrases", selections["paraphrases"]), ("scenarios", selections["scenarios"]), ("festival_names", selections["festivals"]), ("beliefs", selections["festivals"]), ("economic_labels", ECONOMIC_KEYS)):
            for item in ids:
                _nonempty(text[group].get(item), f"{language}.{group}.{item}")
    calendar = None
    if "date_only" in selections["cues"]:
        if not config.get("calendar_path"):
            raise ValueError("date_only requires a researcher-supplied verified calendar_path; no dates are bundled")
        calendar = _load(_relative(config_path.parent, config["calendar_path"]))
        _validate_calendar(calendar)
        selected_ids = config.get("calendar_pair_ids")
        if selected_ids is not None:
            _selection(config, "calendar_pair_ids", [pair["id"] for pair in calendar["pairs"]])
        for festival in selections["festivals"]:
            if not any(p["festival"] == festival and (selected_ids is None or p["id"] in selected_ids) for p in calendar["pairs"]):
                raise ValueError(f"No verified selected calendar pair for {festival}")
    normalized = {key: value for key, value in config.items() if key != "review_path"}
    materials = {"design": {"config": normalized, "data_sha256": _hash(data), "response_contract": RESPONSE_CONTRACT, "scenarios": data["scenarios"], "festival_valence": {f: "avoidance" if f == "pitru_paksha" else "auspicious" for f in FESTIVALS}}}
    for language in selections["languages"]:
        materials[f"language:{language}"] = data["languages"][language]
    if calendar is not None:
        materials["calendar"] = calendar
    manifest = _load(_relative(config_path.parent, config["review_path"])) if load_review and config.get("review_path") else None
    return config, selections, materials, manifest, _hash(data)


def export_review(config_path, output_path):
    """Export pending scope approvals; reviewers edit decision/reviewer/approved_at/notes.

    Point config.review_path at that edited file. No other config/content may change.
    Set mode=confirmatory BEFORE exporting when planning confirmatory collection.
    """
    config, _, materials, _, _ = _prepare(config_path, load_review=False)
    manifest = {"schema_version": 1, "config_sha256": _hash(materials["design"]["config"]), "subjects": []}
    for subject_id, material in materials.items():
        manifest["subjects"].append({"subject_id": subject_id, "sha256": _hash(material), "decision": "pending", "reviewer": None, "approved_at": None, "notes": "", "material": material})
    _save(output_path, manifest)
    return {"output_path": str(output_path), "config_sha256": manifest["config_sha256"], "subjects": len(manifest["subjects"]), "mode": config.get("mode", "exploratory")}


def _check_manifest(manifest, materials):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Missing hash-bound review manifest")
    if manifest.get("config_sha256") != _hash(materials["design"]["config"]):
        raise ValueError("Review config hash mismatch")
    subjects = manifest.get("subjects")
    if not isinstance(subjects, list):
        raise ValueError("Review requires subjects list")
    lookup = {}
    for subject in subjects:
        if not isinstance(subject, dict) or not isinstance(subject.get("subject_id"), str):
            raise ValueError("Review subject requires subject_id")
        if subject["subject_id"] in lookup:
            raise ValueError("Duplicate review subject")
        lookup[subject["subject_id"]] = subject
    for subject_id, material in materials.items():
        subject = lookup.get(subject_id, {})
        if subject.get("sha256") != _hash(material) or _hash(subject.get("material")) != _hash(material):
            raise ValueError(f"Review content hash mismatch: {subject_id}")
        if subject.get("decision") != "approved":
            raise ValueError(f"Review approval required: {subject_id}")
        _nonempty(subject.get("reviewer"), f"{subject_id} reviewer")
        _timestamp(subject.get("approved_at"), f"{subject_id} approved_at")


def validate_review(config_or_data):
    """Validate all selected approval scopes in a config path, or one trial dict."""
    if isinstance(config_or_data, dict):
        validate_trial_review(config_or_data)
    else:
        _, _, materials, manifest, _ = _prepare(config_or_data)
        _check_manifest(manifest, materials)


def _render(trial, materials):
    text = materials[f"language:{trial['language']}"]
    economics = materials["design"]["scenarios"][trial["scenario_id"]]["economics"]
    parts = [text["introduction"], text["roles"][trial["role"]], text["personas"][trial["persona_id"]], text["scenarios"][trial["scenario_id"]], text["paraphrases"][trial["paraphrase_id"]]]
    if trial["cue"] == "date_only":
        pair = next(p for p in materials["calendar"]["pairs"] if p["id"] == trial["calendar_pair_id"])
        parts.append(text["date_frame"].format(date=pair["event_date"] if trial["condition"] == "event" else pair["neutral_date"], locality=pair["locality"], timezone=pair["timezone"]))
    else:
        parts.append(text["hypothetical"])
        label = text["festival_names"][trial["festival"]] if trial["condition"] == "event" else text["neutral"]
        parts.append(text["named_frame"].format(label=label))
        if trial["cue"] == "explicit_belief":
            belief = text["beliefs"][trial["festival"]] if trial["condition"] == "event" else text["neutral_belief"]
            parts.append(text["belief_frame"].format(belief=belief))
    parts.append(text["economics_intro"])
    parts.extend(f"{text['economic_labels'][key]}: {economics[key]}" for key in ECONOMIC_KEYS)
    parts.extend([text["decision"], text["constraints"], materials["design"]["response_contract"], text["rationale_instruction"]])
    return "\n\n".join(parts)


def _identity(trial):
    keys = ("language", "festival", "cue", "role", "persona_id", "scenario_id", "paraphrase_id", "replicate", "calendar_pair_id", "config_sha256", "data_sha256", "calendar_sha256")
    return {key: trial[key] for key in keys}


def validate_trial_review(trial):
    """Recheck review hashes and re-render approved prompt; raises ValueError on drift."""
    try:
        review = trial["review"]
        materials = review["materials"]
        _check_manifest(review["manifest"], materials)
        if trial.get("review_status") != "approved":
            raise ValueError("Trial is not approved")
        if trial["config_sha256"] != _hash(materials["design"]["config"]):
            raise ValueError("Trial config hash mismatch")
        design = materials["design"]
        config = design["config"]
        if trial["data_sha256"] != design["data_sha256"]:
            raise ValueError("Trial data hash mismatch")
        dimensions = (("language", "languages", LANGUAGES), ("festival", "festivals", FESTIVALS), ("role", "roles", ROLES), ("persona_id", "personas", ("unspecified", "observant", "nonobservant")), ("scenario_id", "scenarios", design["scenarios"]), ("paraphrase_id", "paraphrases", ("direct", "planning")), ("cue", "cues", CUES))
        for field, key, default in dimensions:
            if trial[field] not in config.get(key, default):
                raise ValueError(f"Trial {field} outside reviewed design")
        if type(trial["replicate"]) is not int or not 0 <= trial["replicate"] < config.get("replicates", 1):
            raise ValueError("Trial replicate outside reviewed design")
        if trial["condition"] not in ("event", "neutral") or trial["event_valence"] != design["festival_valence"][trial["festival"]]:
            raise ValueError("Trial condition/valence mismatch")
        count = 2 * config.get("replicates", 1)
        for _, key, default in dimensions:
            if key not in ("cues", "festivals"):
                count *= len(config.get(key, default))
        calendar_pairs = materials.get("calendar", {}).get("pairs", [])
        selected_pairs = [p for p in calendar_pairs if p["festival"] in config.get("festivals", FESTIVALS) and ("calendar_pair_ids" not in config or p["id"] in config["calendar_pair_ids"])]
        cue_count = sum(len(selected_pairs) if cue == "date_only" else len(config.get("festivals", FESTIVALS)) for cue in config.get("cues", CUES))
        if trial.get("design_trial_count") != count * cue_count:
            raise ValueError("Trial design size does not match reviewed grid")
        if trial["prompt_sha256"] != _prompt_hash(trial["prompt"]) or trial["prompt"] != _render(trial, materials):
            raise ValueError("Trial prompt does not match reviewed material")
        scenario = materials["design"]["scenarios"][trial["scenario_id"]]
        if trial["economics"] != scenario["economics"] or trial["split"] != scenario["split"] or trial["cluster_id"] != trial["scenario_id"]:
            raise ValueError("Trial scenario metadata mismatch")
        if trial["pair_id"] != _hash(_identity(trial)) or trial["trial_id"] != _hash({"pair_id": trial["pair_id"], "condition": trial["condition"], "prompt_sha256": trial["prompt_sha256"]}):
            raise ValueError("Trial identity mismatch")
        if trial["cue"] == "date_only":
            _validate_calendar(materials["calendar"])
            pair = next(p for p in materials["calendar"]["pairs"] if p["id"] == trial["calendar_pair_id"])
            if pair != trial["calendar_pair"] or pair["festival"] != trial["festival"] or trial["calendar_sha256"] != _hash(materials["calendar"]):
                raise ValueError("Trial calendar metadata mismatch")
            if "calendar_pair_ids" in config and pair["id"] not in config["calendar_pair_ids"]:
                raise ValueError("Trial calendar pair outside reviewed selection")
        elif trial["calendar_pair_id"] is not None or trial["calendar_pair"] is not None:
            raise ValueError("Named/belief trial must not contain a dated pair")
        expected_calendar_hash = _hash(materials["calendar"]) if "calendar" in materials else None
        if trial["calendar_sha256"] != expected_calendar_hash:
            raise ValueError("Trial calendar provenance hash mismatch")
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError(f"Incomplete trial review provenance: {exc}") from exc


def build_trials(config_path, output_path):
    """Build a deterministic full factorial over selected dimensions; never call a model."""
    config, selections, materials, manifest, data_hash = _prepare(config_path)
    approved = False
    if manifest is not None:
        _check_manifest(manifest, materials)
        approved = True
    if config.get("mode", "exploratory") == "confirmatory" and not approved:
        raise ValueError("Confirmatory stimuli require hash-bound reviewer approvals; export_review first")
    config_hash = _hash(materials["design"]["config"])
    calendar = materials.get("calendar")
    calendar_hash = _hash(calendar) if calendar is not None else None
    trials = []
    dimensions = (selections[key] for key in ("languages", "festivals", "roles", "personas", "scenarios", "paraphrases", "cues"))
    for language, festival, role, persona, scenario_id, paraphrase, cue in itertools.product(*dimensions):
        calendar_pairs = [None]
        if cue == "date_only":
            calendar_pairs = [p for p in calendar["pairs"] if p["festival"] == festival and ("calendar_pair_ids" not in config or p["id"] in config["calendar_pair_ids"])]
        for calendar_pair, replicate in itertools.product(calendar_pairs, range(config.get("replicates", 1))):
            for condition in ("event", "neutral"):
                scenario = materials["design"]["scenarios"][scenario_id]
                trial = {"language": language, "festival": festival, "condition": condition, "event_valence": materials["design"]["festival_valence"][festival], "cue": cue, "role": role, "persona_id": persona, "scenario_id": scenario_id, "cluster_id": scenario_id, "split": scenario["split"], "paraphrase_id": paraphrase, "replicate": replicate, "calendar_pair_id": calendar_pair["id"] if calendar_pair else None, "calendar_pair": calendar_pair, "economics": scenario["economics"], "config_sha256": config_hash, "data_sha256": data_hash, "calendar_sha256": calendar_hash, "review_status": "approved" if approved else "draft", "synthetic": True}
                trial["pair_id"] = _hash(_identity(trial))
                trial["prompt"] = _render(trial, materials)
                trial["prompt_sha256"] = _prompt_hash(trial["prompt"])
                trial["trial_id"] = _hash({"pair_id": trial["pair_id"], "condition": condition, "prompt_sha256": trial["prompt_sha256"]})
                trial_materials = {key: value for key, value in materials.items() if key in ("design", f"language:{language}", "calendar")}
                trial_manifest = None if manifest is None else {**manifest, "subjects": [subject for subject in manifest["subjects"] if subject["subject_id"] in trial_materials]}
                trial["review"] = {"materials": trial_materials, "manifest": trial_manifest, "translation_status": "reviewer_attested" if approved else "unreviewed_draft", "calendar_status": "researcher_attested_not_independently_verified" if calendar_pair else "hypothetical_not_calendar_dated"}
                trials.append(trial)
    trials.sort(key=lambda trial: _hash({"seed": config.get("seed", 42), "trial_id": trial["trial_id"]}))
    design_digest = _hash(sorted(trial["trial_id"] for trial in trials))
    for trial in trials:
        trial["design_trial_ids_sha256"] = design_digest
        trial["design_trial_count"] = len(trials)
    from .core import write_jsonl
    write_jsonl(output_path, trials)
    return {"output_path": str(output_path), "trials": len(trials), "pairs": len(trials) // 2, "clusters": len(selections["scenarios"]), "config_sha256": config_hash, "data_sha256": data_hash, "calendar_sha256": calendar_hash, "review_status": "approved" if approved else "draft", "mode": config.get("mode", "exploratory"), "warning": None if approved else "Draft translations and hypothetical calendar cues: exploratory use only"}
