"""Deterministic, economically paired multilingual stimuli and attributable review gates.

Schema v2: every pair holds an event and a neutral trial plus the configured control
conditions (no_calendar, occasion_control, purchase_control). Review materials and
the approval manifest live in ONE sidecar ``<trials>.review.json`` bound to each trial
by ``review.context_sha256``; trials carry only a small review stub.

Review manifests are human attestations, not cryptographic proof of reviewer identity.
Calendar sources and translations must be checked by researchers; this module only
checks metadata and content binding, never claims independent calendar verification.
All shipped template text (t1, t2) is an AI-drafted, unreviewed draft: t2 is NOT
independent human authorship and requires independent bilingual review.
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
PERSONAS = ("unspecified", "observant", "nonobservant")
PARAPHRASES = ("direct", "planning")
CONTROLS = ("no_calendar", "occasion_control", "purchase_control")
VALENCES = ("auspicious", "avoidance")
SIDECAR_SCHEMA_VERSION = 2
DATA_SCHEMA_VERSION = 2
LAYOUT_BLOCKS = ("introduction", "role", "persona", "scenario", "paraphrase", "calendar", "economics")
TEXT_KEYS = ("introduction", "hypothetical", "hypothetical_no_label", "neutral", "neutral_belief", "occasion_label", "occasion_belief", "date_frame", "named_frame", "belief_frame", "economics_intro", "decision", "constraints", "rationale_instruction")
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


def _selection(config, key, allowed, default=None, allow_empty=False):
    values = config.get(key, list(allowed) if default is None else default)
    if not isinstance(values, list) or (not values and not allow_empty) or len(set(values)) != len(values):
        raise ValueError(f"{key} must be a " + ("" if allow_empty else "nonempty ") + "list without duplicates")
    if not set(values).issubset(allowed):
        raise ValueError(f"Unsupported {key}: {set(values) - set(allowed)}")
    return values


def _check_controls(cues, controls):
    if "date_only" in cues and set(controls) - {"no_calendar"}:
        raise ValueError(f"Controls {sorted(set(controls) - {'no_calendar'})} are not defined for cue date_only; only no_calendar may be combined with it")


def _check_template(text, label, scenarios, festivals):
    if not isinstance(text, dict):
        raise ValueError(f"{label}: template must be an object")
    for key in TEXT_KEYS:
        _nonempty(text.get(key), f"{label}.{key}")
    groups = (("roles", ROLES), ("personas", PERSONAS), ("paraphrases", PARAPHRASES), ("scenarios", scenarios), ("festival_names", festivals), ("beliefs", festivals), ("purchase_labels", VALENCES), ("purchase_beliefs", VALENCES), ("economic_labels", ECONOMIC_KEYS))
    for group, ids in groups:
        if not isinstance(text.get(group), dict):
            raise ValueError(f"{label}.{group} must be an object")
        for item in ids:
            _nonempty(text[group].get(item), f"{label}.{group}.{item}")
    layout = text.get("layout")
    if not isinstance(layout, list) or sorted(layout) != sorted(LAYOUT_BLOCKS):
        raise ValueError(f"{label}.layout must be a permutation of {LAYOUT_BLOCKS}")
    provenance = text.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label}.provenance required")
    for key in ("authorship", "status"):
        _nonempty(provenance.get(key), f"{label}.provenance.{key}")
    try:
        text["date_frame"].format(date="d", locality="l", timezone="z")
        text["named_frame"].format(label="x")
        text["belief_frame"].format(belief="x")
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"{label}: invalid frame placeholders: {exc!r}") from exc


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
    allowed = {"schema_version", "data_path", "calendar_path", "review_path", "languages", "festivals", "roles", "personas", "scenarios", "paraphrases", "cues", "templates", "controls", "replicates", "seed", "mode", "calendar_pair_ids"}
    if set(config) - allowed:
        raise ValueError(f"Unknown config keys: {sorted(set(config) - allowed)}")
    if config.get("mode", "exploratory") not in ("exploratory", "confirmatory"):
        raise ValueError("mode must be exploratory or confirmatory")
    for key, default in (("replicates", 1), ("seed", 42)):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or (key == "replicates" and value < 1):
            raise ValueError(f"{key} must be an integer" + (" >=1" if key == "replicates" else ""))
    data = _load(_relative(config_path.parent, config["data_path"]))
    if data.get("schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError(f"Stimuli data requires schema_version={DATA_SCHEMA_VERSION}")
    selections = {
        "languages": _selection(config, "languages", LANGUAGES),
        "festivals": _selection(config, "festivals", FESTIVALS),
        "roles": _selection(config, "roles", ROLES),
        "cues": _selection(config, "cues", CUES),
        "personas": _selection(config, "personas", PERSONAS),
        "scenarios": _selection(config, "scenarios", data["scenarios"]),
        "paraphrases": _selection(config, "paraphrases", PARAPHRASES),
        "templates": _selection(config, "templates", sorted({tid for lang in data["languages"].values() for tid in lang}), default=["t1"]),
        "controls": _selection(config, "controls", CONTROLS, default=[], allow_empty=True),
    }
    _check_controls(selections["cues"], selections["controls"])
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
        for template_id in selections["templates"]:
            _check_template(data["languages"][language].get(template_id), f"{language}.{template_id}", selections["scenarios"], selections["festivals"])
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
    # Defaults are written into the hashed config so omitted and explicit defaults are one design.
    normalized = {key: value for key, value in config.items() if key != "review_path"}
    normalized["templates"], normalized["controls"] = selections["templates"], selections["controls"]
    materials = {"design": {"config": normalized, "data_sha256": _hash(data), "response_contract": RESPONSE_CONTRACT, "scenarios": data["scenarios"], "festival_valence": {f: "avoidance" if f == "pitru_paksha" else "auspicious" for f in FESTIVALS}}}
    for language in selections["languages"]:
        for template_id in selections["templates"]:
            materials[f"language:{language}:{template_id}"] = data["languages"][language][template_id]
    if calendar is not None:
        materials["calendar"] = calendar
    manifest = _load(_relative(config_path.parent, config["review_path"])) if load_review and config.get("review_path") else None
    return config, selections, materials, manifest, _hash(data)


def export_review(config_path, output_path):
    """Export pending scope approvals; reviewers edit decision/reviewer/approved_at/notes.

    Scopes: one per (language, template) plus the design block and optional calendar.
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


def sidecar_path(trials_path):
    """Review sidecar location for a trials file: ``<trials_path>.review.json``."""
    return Path(str(trials_path) + ".review.json")


def _context(materials, manifest):
    return {"schema_version": SIDECAR_SCHEMA_VERSION, "context_sha256": _hash({"materials": materials, "manifest": manifest}), "materials": materials, "manifest": manifest}


def load_review_context(trials_path):
    """Read ``<trials_path>.review.json``; None if absent. Raises ValueError if its own hash fails."""
    path = sidecar_path(trials_path)
    if not path.exists():
        return None
    context = _load(path)
    if not isinstance(context, dict) or context.get("schema_version") != SIDECAR_SCHEMA_VERSION or not isinstance(context.get("materials"), dict):
        raise ValueError(f"{path}: not a schema v{SIDECAR_SCHEMA_VERSION} review context")
    if context.get("context_sha256") != _hash({"materials": context["materials"], "manifest": context.get("manifest")}):
        raise ValueError(f"{path}: review context hash mismatch")
    return context


def validate_review(config_or_trial, review_context=None):
    """Validate all selected approval scopes in a config path, or one trial dict against a review context."""
    if isinstance(config_or_trial, dict):
        validate_trial_review(config_or_trial, review_context)
    else:
        _, _, materials, manifest, _ = _prepare(config_or_trial)
        _check_manifest(manifest, materials)


def _label_and_belief(text, trial):
    condition = trial["condition"]
    if condition == "event":
        return text["festival_names"][trial["festival"]], text["beliefs"][trial["festival"]]
    if condition == "neutral":
        return text["neutral"], text["neutral_belief"]
    if condition == "occasion_control":
        return text["occasion_label"], text["occasion_belief"]
    if condition == "purchase_control":
        valence = trial["event_valence"]
        return text["purchase_labels"][valence], text["purchase_beliefs"][valence]
    raise ValueError(f"Unknown condition: {condition}")


def _render(trial, materials):
    text = materials[f"language:{trial['language']}:{trial['template_id']}"]
    economics = materials["design"]["scenarios"][trial["scenario_id"]]["economics"]
    condition, cue = trial["condition"], trial["cue"]
    if cue == "date_only" and condition not in ("event", "neutral", "no_calendar"):
        raise ValueError(f"Condition {condition} is not defined for cue date_only")
    if condition == "no_calendar":
        calendar = [text["hypothetical_no_label"]]
    elif cue == "date_only":
        pair = next(p for p in materials["calendar"]["pairs"] if p["id"] == trial["calendar_pair_id"])
        calendar = [text["date_frame"].format(date=pair["event_date"] if condition == "event" else pair["neutral_date"], locality=pair["locality"], timezone=pair["timezone"])]
    else:
        label, belief = _label_and_belief(text, trial)
        calendar = [text["hypothetical"], text["named_frame"].format(label=label)]
        if cue == "explicit_belief":
            calendar.append(text["belief_frame"].format(belief=belief))
    blocks = {
        "introduction": [text["introduction"]],
        "role": [text["roles"][trial["role"]]],
        "persona": [text["personas"][trial["persona_id"]]],
        "scenario": [text["scenarios"][trial["scenario_id"]]],
        "paraphrase": [text["paraphrases"][trial["paraphrase_id"]]],
        "calendar": calendar,
        "economics": [text["economics_intro"], *(f"{text['economic_labels'][key]}: {economics[key]}" for key in ECONOMIC_KEYS)],
    }
    parts = [part for name in text["layout"] for part in blocks[name]]
    parts.extend([text["decision"], text["constraints"], materials["design"]["response_contract"], text["rationale_instruction"]])
    return "\n\n".join(parts)


def _identity(trial):
    keys = ("language", "template_id", "festival", "cue", "role", "persona_id", "scenario_id", "paraphrase_id", "replicate", "calendar_pair_id", "config_sha256", "data_sha256", "calendar_sha256")
    return {key: trial[key] for key in keys}


def _selected_calendar_pairs(materials):
    config = materials["design"]["config"]
    pairs = materials.get("calendar", {}).get("pairs", [])
    return [p for p in pairs if p["festival"] in config.get("festivals", FESTIVALS) and ("calendar_pair_ids" not in config or p["id"] in config["calendar_pair_ids"])]


def _design_count(materials):
    """Trial count implied by the reviewed grid: pairs x (2 + #controls)."""
    design = materials["design"]
    config = design["config"]
    count = (2 + len(config["controls"])) * config.get("replicates", 1)
    for key, default in (("languages", LANGUAGES), ("roles", ROLES), ("personas", PERSONAS), ("scenarios", design["scenarios"]), ("paraphrases", PARAPHRASES), ("templates", ())):
        count *= len(config.get(key, default))
    selected_pairs = _selected_calendar_pairs(materials)
    cue_count = sum(len(selected_pairs) if cue == "date_only" else len(config.get("festivals", FESTIVALS)) for cue in config.get("cues", CUES))
    return count * cue_count


# Cache approval checks only while the current material and manifest bytes still match.
# Object identity alone is insufficient because callers can mutate nested dictionaries.
_CONTEXT_CACHE = {"context": None, "fingerprint": None, "info": None}


def _verified_context(review_context):
    if not isinstance(review_context, dict):
        raise ValueError("Missing review sidecar: confirmatory validation requires the <trials>.review.json context")
    materials, manifest = review_context["materials"], review_context.get("manifest")
    fingerprint = _hash({"materials": materials, "manifest": manifest})
    if review_context.get("schema_version") != SIDECAR_SCHEMA_VERSION or review_context["context_sha256"] != fingerprint:
        raise ValueError("Review context hash mismatch")
    if (_CONTEXT_CACHE["context"] is review_context and _CONTEXT_CACHE["fingerprint"] == fingerprint):
        return _CONTEXT_CACHE["info"]
    _check_manifest(manifest, materials)
    config = materials["design"]["config"]
    _check_controls(config.get("cues", CUES), config.get("controls", []))
    calendar_hash = None
    if "calendar" in materials:
        _validate_calendar(materials["calendar"])
        calendar_hash = _hash(materials["calendar"])
    info = {"context_sha256": review_context["context_sha256"], "materials": materials, "config": config, "config_sha256": _hash(config), "design": materials["design"],
            "count": _design_count(materials), "calendar_hash": calendar_hash, "pairs": {p["id"]: p for p in materials.get("calendar", {}).get("pairs", [])}}
    _CONTEXT_CACHE.update(context=review_context, fingerprint=fingerprint, info=info)
    return info


def validate_trial_review(trial, review_context):
    """Recheck the sidecar context binding, hashes and approvals, and re-render the prompt; raises ValueError on drift.

    review_context is the dict from load_review_context; None is an error.
    """
    try:
        info = _verified_context(review_context)
        materials, design, config = info["materials"], info["design"], info["config"]
        review = trial["review"]
        if not isinstance(review, dict) or review.get("context_sha256") != info["context_sha256"]:
            raise ValueError("Trial review context hash mismatch")
        if trial.get("review_status") != "approved" or review.get("translation_status") != "reviewer_attested":
            raise ValueError("Trial is not approved")
        if trial["config_sha256"] != info["config_sha256"]:
            raise ValueError("Trial config hash mismatch")
        if trial["data_sha256"] != design["data_sha256"]:
            raise ValueError("Trial data hash mismatch")
        dimensions = (("language", "languages", LANGUAGES), ("festival", "festivals", FESTIVALS), ("role", "roles", ROLES), ("persona_id", "personas", PERSONAS), ("scenario_id", "scenarios", design["scenarios"]), ("paraphrase_id", "paraphrases", PARAPHRASES), ("cue", "cues", CUES), ("template_id", "templates", ()))
        for field, key, default in dimensions:
            if trial[field] not in config.get(key, default):
                raise ValueError(f"Trial {field} outside reviewed design")
        if type(trial["replicate"]) is not int or not 0 <= trial["replicate"] < config.get("replicates", 1):
            raise ValueError("Trial replicate outside reviewed design")
        if trial["condition"] not in ("event", "neutral", *config["controls"]) or trial["event_valence"] != design["festival_valence"][trial["festival"]]:
            raise ValueError("Trial condition/valence mismatch")
        if trial.get("design_trial_count") != info["count"]:
            raise ValueError("Trial design size does not match reviewed grid")
        if trial["prompt_sha256"] != _prompt_hash(trial["prompt"]) or trial["prompt"] != _render(trial, materials):
            raise ValueError("Trial prompt does not match reviewed material")
        scenario = design["scenarios"][trial["scenario_id"]]
        if trial["economics"] != scenario["economics"] or trial["split"] != scenario["split"] or trial["cluster_id"] != trial["scenario_id"]:
            raise ValueError("Trial scenario metadata mismatch")
        if trial["pair_id"] != _hash(_identity(trial)) or trial["trial_id"] != _hash({"pair_id": trial["pair_id"], "condition": trial["condition"], "prompt_sha256": trial["prompt_sha256"]}):
            raise ValueError("Trial identity mismatch")
        if trial["cue"] == "date_only":
            pair = info["pairs"][trial["calendar_pair_id"]]
            if pair != trial["calendar_pair"] or pair["festival"] != trial["festival"] or trial["calendar_sha256"] != info["calendar_hash"]:
                raise ValueError("Trial calendar metadata mismatch")
            if "calendar_pair_ids" in config and pair["id"] not in config["calendar_pair_ids"]:
                raise ValueError("Trial calendar pair outside reviewed selection")
            if review.get("calendar_status") != "researcher_attested_not_independently_verified":
                raise ValueError("Trial calendar status mismatch")
        elif trial["calendar_pair_id"] is not None or trial["calendar_pair"] is not None or review.get("calendar_status") != "hypothetical_not_calendar_dated":
            raise ValueError("Named/belief trial must not contain a dated pair")
        if trial["calendar_sha256"] != info["calendar_hash"]:
            raise ValueError("Trial calendar provenance hash mismatch")
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError(f"Incomplete trial review provenance: {exc!r}") from exc


def build_trials(config_path, output_path):
    """Build a deterministic full factorial over selected dimensions; never call a model.

    Writes the trials JSONL and the review sidecar ``<output_path>.review.json``.

    Repetition note: the neutral and control prompts do not mention the festival, so
    identical neutral/control text recurs across festivals (same language, template,
    role, persona, scenario, paraphrase, cue). Those trials have different pair_id /
    trial_id and sampling seeds, but their text is identical: they are effectively
    repeated sampling of one prompt, not new coverage. The summary therefore reports
    distinct prompt counts next to trial counts.
    """
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
    context = _context(materials, manifest)
    conditions = ("event", "neutral", *selections["controls"])
    trials = []
    dimensions = (selections[key] for key in ("languages", "templates", "festivals", "roles", "personas", "scenarios", "paraphrases", "cues"))
    for language, template_id, festival, role, persona, scenario_id, paraphrase, cue in itertools.product(*dimensions):
        calendar_pairs = [None]
        if cue == "date_only":
            calendar_pairs = [p for p in calendar["pairs"] if p["festival"] == festival and ("calendar_pair_ids" not in config or p["id"] in config["calendar_pair_ids"])]
        for calendar_pair, replicate in itertools.product(calendar_pairs, range(config.get("replicates", 1))):
            for condition in conditions:
                scenario = materials["design"]["scenarios"][scenario_id]
                trial = {"language": language, "template_id": template_id, "festival": festival, "condition": condition, "event_valence": materials["design"]["festival_valence"][festival], "cue": cue, "role": role, "persona_id": persona, "scenario_id": scenario_id, "cluster_id": scenario_id, "split": scenario["split"], "paraphrase_id": paraphrase, "replicate": replicate, "calendar_pair_id": calendar_pair["id"] if calendar_pair else None, "calendar_pair": calendar_pair, "economics": scenario["economics"], "config_sha256": config_hash, "data_sha256": data_hash, "calendar_sha256": calendar_hash, "review_status": "approved" if approved else "draft", "synthetic": True}
                trial["pair_id"] = _hash(_identity(trial))
                trial["prompt"] = _render(trial, materials)
                trial["prompt_sha256"] = _prompt_hash(trial["prompt"])
                trial["trial_id"] = _hash({"pair_id": trial["pair_id"], "condition": condition, "prompt_sha256": trial["prompt_sha256"]})
                trial["review"] = {"context_sha256": context["context_sha256"], "translation_status": "reviewer_attested" if approved else "unreviewed_draft", "calendar_status": "researcher_attested_not_independently_verified" if calendar_pair else "hypothetical_not_calendar_dated"}
                trials.append(trial)
    trials.sort(key=lambda trial: _hash({"seed": config.get("seed", 42), "trial_id": trial["trial_id"]}))
    design_digest = _hash(sorted(trial["trial_id"] for trial in trials))
    for trial in trials:
        trial["design_trial_ids_sha256"] = design_digest
        trial["design_trial_count"] = len(trials)
    from .core import atomic_text, write_jsonl
    sidecar = sidecar_path(output_path)
    atomic_text(sidecar, json.dumps(context, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    write_jsonl(output_path, trials)
    distinct_by_condition = {condition: len({t["prompt_sha256"] for t in trials if t["condition"] == condition}) for condition in conditions}
    return {"output_path": str(output_path), "review_context_path": str(sidecar), "review_context_sha256": context["context_sha256"], "trials": len(trials), "pairs": len({t["pair_id"] for t in trials}), "distinct_prompts": len({t["prompt_sha256"] for t in trials}), "distinct_neutral_prompts": distinct_by_condition["neutral"], "distinct_prompts_by_condition": distinct_by_condition, "scenarios": len(selections["scenarios"]), "templates": len(selections["templates"]), "controls": len(selections["controls"]), "conditions": len(conditions), "condition_ids": list(conditions), "clusters": len(selections["scenarios"]), "config_sha256": config_hash, "data_sha256": data_hash, "calendar_sha256": calendar_hash, "review_status": "approved" if approved else "draft", "mode": config.get("mode", "exploratory"), "warning": None if approved else "Draft translations and hypothetical calendar cues: exploratory use only"}
