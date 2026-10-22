"""Command-line entry point. Optional model dependencies load only on demand."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from .core import read_jsonl, run_trials, validate_trials, write_jsonl


def parser():
    root = argparse.ArgumentParser(prog="auspicious", description="Multilingual calendar-belief investment experiments")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build deterministic paired trials from a JSON design")
    build.add_argument("--config", required=True)
    build.add_argument("--out", required=True)
    review = commands.add_parser("review-export", help="Export hash-bound attestations for human review")
    review.add_argument("--config", required=True)
    review.add_argument("--out", required=True)
    validate = commands.add_parser("validate", help="Check pair integrity, hashes, and optional review approvals")
    validate.add_argument("--trials", required=True)
    validate.add_argument("--confirmatory", action="store_true")
    select = commands.add_parser("select", help="Create a whole-pair exploratory/probe subset")
    select.add_argument("--trials", required=True)
    select.add_argument("--out", required=True)
    for field in ("language", "festival", "cue", "role", "persona_id", "split"):
        select.add_argument("--" + field.replace("_", "-"), action="append")
    commands.add_parser("models", help="List Gemini text-generation models accessible to your key")
    run = commands.add_parser("run", help="Run/resume Gemini calls with SQLite checkpoints")
    local = commands.add_parser("run-openweights", help="Run/resume a local Hugging Face decoder-only model")
    for command in (run, local):
        command.add_argument("--trials", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--temperature", type=float, default=0.7 if command is run else 0.0)
        command.add_argument("--max-output-tokens", type=int, default=1024)
        command.add_argument("--seed", type=int, default=None if command is run else 42)
        command.add_argument("--limit", type=int, help="Exploratory only; rounded down to whole pairs")
        command.add_argument("--confirmatory", action="store_true")
        command.add_argument("--retry-errors", action="store_true", help="Retry transport errors only; preserves invalid/blocked outputs")
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--rpm", type=float, default=30, help="Global request-start rate across all supplied keys")
    run.add_argument("--timeout", type=float, default=120)
    run.add_argument("--max-attempts", type=int, default=4)
    local.add_argument("--revision")
    local.add_argument("--device", default="cpu")
    local.add_argument("--steering-path")
    local.add_argument("--strength", type=float, default=0.0)
    local.add_argument("--layer", type=int)
    local.add_argument("--random-direction-seed", type=int)
    analysis = commands.add_parser("analyze", help="Paired clustered effects and matched-language interactions")
    analysis.add_argument("--records", required=True)
    analysis.add_argument("--out", required=True)
    analysis.add_argument("--bootstrap", type=int, default=2000)
    analysis.add_argument("--seed", type=int, default=42)
    analysis.add_argument("--humans", help="Validated human records JSONL from import-humans")
    human = commands.add_parser("import-humans", help="Validate observed same-task participant CSV responses")
    human.add_argument("--trials", required=True)
    human.add_argument("--csv", required=True)
    human.add_argument("--out", required=True)
    survey = commands.add_parser("export-survey", help="Export prompt-only survey material; no responses generated")
    survey.add_argument("--trials", required=True)
    survey.add_argument("--out", required=True)
    context = commands.add_parser("context", help="Summarize sourced external market series descriptively")
    context.add_argument("--csv", required=True)
    context.add_argument("--out", required=True)
    extract = commands.add_parser("extract", help="Extract prompt-only final-token residual activations")
    extract.add_argument("--trials", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--model", required=True)
    extract.add_argument("--revision")
    extract.add_argument("--layer", type=int, default=-1)
    extract.add_argument("--device", default="cpu")
    probe = commands.add_parser("probe", help="Fit a cluster-isolated train/dev/test linear probe")
    probe.add_argument("--activations", required=True)
    probe.add_argument("--out", required=True)
    probe.add_argument("--ridge", type=float, default=1.0)
    return root


def dispatch(args):
    if args.command == "build":
        from .stimuli import build_trials
        return build_trials(args.config, args.out)
    if args.command == "review-export":
        from .stimuli import export_review
        return export_review(args.config, args.out)
    if args.command == "validate":
        trials = read_jsonl(args.trials)
        validate_trials(trials, confirmatory=args.confirmatory)
        return {"valid": True, "trials": len(trials), "confirmatory": args.confirmatory,
                "review_statuses": sorted({t["review_status"] for t in trials})}
    if args.command == "select":
        trials = read_jsonl(args.trials)
        validate_trials(trials)
        fields = ("language", "festival", "cue", "role", "persona_id", "split")
        selected = [trial for trial in trials if all(
            getattr(args, field) is None or trial[field] in getattr(args, field) for field in fields)]
        validate_trials(selected)
        write_jsonl(args.out, selected)
        return {"output": args.out, "trials": len(selected),
                "warning": "Subset is exploratory unless it is the entire reviewed design"}
    if args.command in {"run", "run-openweights"}:
        validate_trials(read_jsonl(args.trials), confirmatory=args.confirmatory)
        if args.confirmatory and args.limit is not None:
            raise ValueError("Confirmatory runs cannot use --limit")
    if args.command == "models":
        from .gemini import list_models
        return list_models()
    if args.command == "run":
        from .gemini import GeminiProvider
        provider = GeminiProvider(args.model, temperature=args.temperature,
                                  max_output_tokens=args.max_output_tokens, seed=args.seed,
                                  rpm=args.rpm, timeout=args.timeout, max_attempts=args.max_attempts)
        return run_trials(args.trials, args.out, provider, confirmatory=args.confirmatory,
                          concurrency=args.concurrency, limit=args.limit, retry_errors=args.retry_errors)
    if args.command == "run-openweights":
        from .openweights import OpenWeightProvider
        provider = OpenWeightProvider(args.model, revision=args.revision, device=args.device,
                                      temperature=args.temperature, max_output_tokens=args.max_output_tokens,
                                      seed=args.seed, steering_path=args.steering_path, strength=args.strength,
                                      layer=args.layer, random_direction_seed=args.random_direction_seed)
        return run_trials(args.trials, args.out, provider, confirmatory=args.confirmatory,
                          concurrency=1, limit=args.limit, retry_errors=args.retry_errors)
    if args.command == "analyze":
        from .analysis import analyze
        return analyze(args.records, args.out, bootstrap=args.bootstrap, seed=args.seed, human_path=args.humans)
    if args.command == "import-humans":
        from .analysis import import_humans
        return import_humans(args.trials, args.csv, args.out)
    if args.command == "context":
        from .analysis import contextual_series
        return contextual_series(args.csv, args.out)
    if args.command == "export-survey":
        trials = read_jsonl(args.trials)
        validate_trials(trials)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["trial_id", "language", "prompt"])
            writer.writeheader()
            for trial in sorted(trials, key=lambda t: t["trial_id"]):
                writer.writerow({k: trial[k] for k in writer.fieldnames})
        return {"output": str(path), "trials": len(trials),
                "warning": "Survey materials only; obtain ethics approval and preregister participant assignment"}
    if args.command == "extract":
        from .openweights import extract_activations
        return extract_activations(args.trials, args.out, model=args.model, revision=args.revision,
                                   layer=args.layer, device=args.device)
    if args.command == "probe":
        from .openweights import fit_probe
        return fit_probe(args.activations, args.out, ridge=args.ridge)
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; committed results are preserved. Resume with the same command.", file=sys.stderr)
        return 130
