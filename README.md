# auspicious-investors

Reproducible multilingual experiment framework for studying whether LLMs acting as
Indian retail investors or financial advisors let calendar-auspiciousness beliefs
(Dhanteras, Akshaya Tritiya, Diwali Muhurat, Pitru Paksha) shift financial
recommendations, and whether the effect amplifies when prompted in Indian languages.

> **All shipped data, translations, and prompts are draft research materials.
> No trial has independent calendar-date verification or human reviewer attestation.
> Exploratory use only until the review workflow is completed.**

---

## Installation

```sh
python3 -m venv .venv
.venv/bin/pip install -e .                    # core + Gemini (stdlib only)
.venv/bin/pip install -e '.[openweights]'     # + torch/transformers for local models
```

Python ≥ 3.11. `zoneinfo` tzdata required for `date_only` calendar validation
(`pip install tzdata` on Windows/some containers).

---

## Quick start: Gemini pilot run

```sh
# 1  Build the 80-trial pilot (5 languages × 4 festivals × 2 roles × 1 scenario, draft)
auspicious build --config examples/pilot.json --out runs/pilot.jsonl

# 2  Inspect (no API call)
auspicious validate --trials runs/pilot.jsonl

# 3  Run — resumes automatically if interrupted
export GEMINI_API_KEYS='["key-one","key-two","key-three"]'
auspicious run \
  --trials runs/pilot.jsonl \
  --out    runs/pilot-gemini-flash \
  --model  gemini-2.5-flash \
  --concurrency 8 \
  --rpm 60

# 4  Analyze
auspicious analyze \
  --records runs/pilot-gemini-flash/records.jsonl \
  --out     runs/pilot-analysis \
  --bootstrap 2000

# Review outputs
cat runs/pilot-analysis/effects.csv
cat runs/pilot-analysis/language_interactions.csv
```

Full design (5,760 trials): replace `examples/pilot.json` with `data/full-config.json`.

---

## Key commands

| Command | Purpose |
|---|---|
| `build` | Deterministic full-factorial trial JSONL from a JSON design |
| `validate` | Check pair integrity, content hashes, optional review approvals |
| `review-export` | Export hash-bound attestation manifest for human sign-off |
| `select` | Create a whole-pair exploratory subset (by festival/language/cue/role/…) |
| `models` | List Gemini generateContent-capable models for your key |
| `run` | Resumable Gemini execution with SQLite checkpoint |
| `run-openweights` | Resumable local HuggingFace decoder execution |
| `analyze` | Paired clustered effects, language interactions, human comparison |
| `import-humans` | Validate consented participant CSV → records JSONL |
| `export-survey` | Prompt-only CSV for human survey delivery |
| `context` | Descriptive import of sourced external market series |
| `extract` | Prompt-only final-token residual activations (no generation) |
| `probe` | Cluster-isolated ridge linear probe on extracted activations |

```sh
auspicious <command> --help
```

---

## Rate limiting and multi-key throughput

Gemini rate limits are **per project**, not per API key. Multiple keys from the same
project share the same quota ceiling. The runner uses a single shared token-bucket
gate across all keys (default `--rpm 30`). Match `--rpm` to the RPM limit shown in
[AI Studio](https://aistudio.google.com/rate-limit) for your tier and model.

For Tier-2/3 accounts with high RPM ceilings, the Batch API is more efficient for
large full-design runs. The runner uses the synchronous API; adapt `gemini.py` to
batch calls if needed.

---

## Experimental design

### Stimuli

Trials are counterfactual pairs: event (festival named or dated) vs. neutral (identical
economics, same role and persona, different calendar label only). Within a pair:

- All economic parameters are identical
- Market access, prices, fees, expected returns, and liquidity constraints are fixed
- The only varying text is the calendar label (and optional cultural context sentence)

**Three cue conditions** (separate experimental arms, not merged):

| Cue | What changes event vs. neutral |
|---|---|
| `festival_named` | Festival name vs. "ordinary comparison day" |
| `explicit_belief` | + one-sentence cultural note (not an instruction to follow it) |
| `date_only` | A verified local calendar date (researcher-supplied; not bundled) |

**Languages:** `en`, `hi`, `ta`, `gu`, `bn` — each an independently authored template,
not a translation of English. Draft status; require native-speaker review.

**Personas:** `unspecified` (baseline), `observant`, `nonobservant` — crossed with all
cues. The `explicit_belief` belief sentence describes third-party practice, never
instructs the persona.

### Outcomes

Seven measured per trial: `gold_pct`, `equity_pct`, `cash_pct` (allocation), `buy_now`
(immediate execution), `timing_days` (days delay), `willingness_to_pay_inr` (premium
willing to pay for one gold unit), `expected_return_pct` (stated gold return belief).

All outcomes are model-reported JSON. Invalid responses (parse failure, constraint
violation) are preserved as `invalid`, never imputed as zero or discarded silently.

### Analysis

Effects are **paired event-minus-neutral within scenario cluster**, averaged:
1. Repetitions within subject/scenario (eliminates pseudoreplication)
2. Subjects within scenario
3. Scenarios equally (not proportional to trial count)

Language interactions subtract English effects from other-language effects on matched
tasks. CIs are 95% percentile bootstrap resampling scenarios (not independent
generations). All unadjusted and exploratory unless preregistered.

---

## Confirmatory workflow

```sh
# 1  Set confirmatory mode in config BEFORE exporting review
#    Edit full-config.json: "mode": "confirmatory"

# 2  Export review manifest for attestation
auspicious review-export --config data/full-config.json --out review/manifest.json

# 3  Researchers edit manifest.json: set decision="approved", reviewer, approved_at
#    (one entry per language content block + design block; not per trial)

# 4  Add review_path to config:
#    "review_path": "review/manifest.json"

# 5  Build confirmatory corpus
auspicious build --config data/full-config.json --out runs/confirmatory.jsonl

# 6  Run — the full corpus check rejects any subset
auspicious run --trials runs/confirmatory.jsonl --out runs/confirmatory-run \
  --model gemini-2.5-flash --confirmatory --concurrency 8 --rpm 60
```

Confirmatory runs reject: draft review status, content hash mismatches, any subset of
the reviewed design, prompt text edits after approval, changed economic parameters, and
incompatible provider settings.

---

## date_only cue

The `date_only` cue requires researcher-supplied verified calendar pairs. No dates are
bundled; `data/calendar-empty.json` shows the schema. Each pair requires:

- Verified `event_date` and `neutral_date` (ISO YYYY-MM-DD)
- `locality`, `timezone` (IANA), `verification_method`, `verified_by`, `verified_at`
- Sources with `url` and `accessed_on` for both dates
- Explicit `"verified": true` attestation

The economic parameters fix market access identically on both dates ("trading is
available with identical hours regardless of actual market closures"), so the calendar
date is the only varying element.

---

## Probe and steering (open-weight models)

```sh
# Extract prompt-only residuals (no generation; no priming)
auspicious select --trials runs/full.jsonl --festival dhanteras --out runs/dhanteras.jsonl
auspicious extract --trials runs/dhanteras.jsonl --out runs/features.npz \
  --model meta-llama/Llama-3.2-1B-Instruct --layer -1

# Fit cluster-isolated ridge probe (train/dev/test never share clusters)
auspicious probe --activations runs/features.npz --out runs/probe.npz

# Steered inference (positive = toward event; negative = toward neutral)
auspicious run-openweights --trials runs/dhanteras.jsonl \
  --out runs/steered-positive --model meta-llama/Llama-3.2-1B-Instruct \
  --steering-path runs/probe.npz --strength 1 --layer -1

# Norm-matched random direction control (same strength, same norm, different direction)
auspicious run-openweights --trials runs/dhanteras.jsonl \
  --out runs/steered-random --model meta-llama/Llama-3.2-1B-Instruct \
  --steering-path runs/probe.npz --strength 1 --layer -1 --random-direction-seed 42
```

**What probing and steering establish and do not establish:**

- The probe detects event-vs-neutral calendar-salience in activations. It does not
  identify an "auspiciousness neuron" or a causally specific circuit.
- The probe label is `event` (any festival, including avoidance Pitru Paksha) vs.
  `neutral`. Mixed-valence datasets require separate probes; the tool enforces this.
- Steering changes downstream behavior in the tested setting. It does not prove causal
  mediation or rule out that the direction encodes other correlated properties (language,
  gold semantics, sentiment, topic).
- The random-direction control checks specificity; it is diagnostic, not a calibrated
  significance test. Activation patching would strengthen causal claims.
- Probe accuracy on the test split and steering specificity on non-festival decisions
  (arithmetic, risk sensitivity, refusals) should both be reported.

---

## Human comparison

```sh
# Export survey (prompts only; no model outputs)
auspicious export-survey --trials runs/dhanteras.jsonl --out surveys/dhanteras-prompts.csv

# After collecting IRB-approved participant responses:
auspicious import-humans \
  --trials runs/dhanteras.jsonl \
  --csv    surveys/responses.csv \
  --out    runs/human-records.jsonl

# Analyze with human comparison
auspicious analyze \
  --records runs/dhanteras-run/records.jsonl \
  --humans  runs/human-records.jsonl \
  --out     runs/comparison-analysis
```

Human response CSV columns: `trial_id,subject_id,gold_pct,equity_pct,cash_pct,buy_now,
timing_days,willingness_to_pay_inr,expected_return_pct,rationale`

`buy_now` accepts `true/false` or `1/0`. Invalid cells become `invalid` records,
never imputed. Duplicate `(subject_id, trial_id)` pairs abort import.

Human comparisons require identical trial IDs, prompts, and task metadata.
The LLM–human comparison estimand is model-effect minus human-effect on the same tasks.
It does not adjust for human sample representativeness.

---

## External contextual data

```sh
auspicious context --csv data/gold-imports.csv --out runs/context
```

CSV columns: `date,series,value,unit,source,event`

Each `(series,unit,source)` is kept as a separate time series. Seasonal patterns
in import volumes and Muhurat-session returns are descriptive context, not causal
estimates of individual auspiciousness beliefs. Do not pool or convert these series
into allocation percentages.

See `examples/context-format.csv` for the column format.

---

## Scientific limits

- **Identification:** Date/festival salience in the prompt is not randomised across
  model versions or real time. The counterfactual pair controls economic content but
  not the full information environment a model may encode.
- **Multilingual:** Draft templates are researcher-authored, not independently reviewed.
  Language effects bundle fluency, script, register, and cultural framing. Translation
  quality must be validated by competent bilingual domain reviewers before
  multilingual interaction effects are interpreted.
- **Representation:** LLM responses to synthetic vignettes are not observations of
  real household decisions. Thousands of generations from one template are not
  thousands of independent investors.
- **Probing and steering:** The probe direction may encode correlated properties.
  The single label-flip control is diagnostic; calibrated permutation significance
  requires more.
- **Human comparison:** The comparison estimates model-effect minus human-effect on the
  same synthetic tasks. It does not estimate the effect of LLM use on real investment
  decisions.
- **Preregistration:** Exploratory results must not be treated as confirmatory without
  prospective preregistration and the full review workflow.

---

## Credentials

```sh
export GEMINI_API_KEY=your-key          # single key
export GEMINI_API_KEYS='["k1","k2"]'   # JSON array for multiple keys
```

Keys are never written to disk, logs, or result files. The runner checks this
explicitly. Do not commit `.env` files; `.gitignore` excludes them.
