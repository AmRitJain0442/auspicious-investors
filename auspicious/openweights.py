"""Optional local inference and correlational probes, not causal mediation estimates.

Only explicitly selected Hugging Face decoder models are loaded. Residuals are
captured at decoder-block output, before the model's final normalization. Steering
adds a direction to every token at that same block, including cached decoding.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading


_FORMAT = 1
_REPRESENTATION = "decoder_block_output_prompt_final_token"


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("Install the openweights extra (numpy, torch, transformers).") from exc
    return np


def _load_runtime(model, revision, device):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ValueError("Install the openweights extra (numpy, torch, transformers).") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        network = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, trust_remote_code=False
        ).to(device)
        network.eval()
    except Exception as exc:
        raise ValueError(f"Cannot load explicitly selected model {model!r}: {exc}") from exc
    return torch, tokenizer, network


def _decoder_layers(network):
    layers = getattr(getattr(network, "model", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Unsupported architecture: expected decoder blocks at model.model.layers.")
    if getattr(network.config, "is_encoder_decoder", False):
        raise ValueError("Unsupported architecture: encoder-decoder models are not supported.")
    return layers


def _layer_index(layer, count):
    if isinstance(layer, bool) or not isinstance(layer, int) or not -count <= layer < count:
        raise ValueError(f"layer must be an integer in [{-count}, {count - 1}].")
    return layer % count


def _width(network):
    width = getattr(network.config, "hidden_size", None)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("Unsupported architecture: config.hidden_size is required.")
    return width


def _hidden(output):
    value = output[0] if isinstance(output, tuple) else output
    if not hasattr(value, "ndim") or value.ndim != 3:
        raise ValueError("Unsupported decoder output: expected [batch, tokens, hidden] tensor.")
    return value


def _inputs(tokenizer, prompt, device):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be nonempty text.")
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                            return_token_type_ids=False, truncation=False)
    else:
        encoded = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, truncation=False)
    return {key: value.to(device) for key, value in encoded.items()}


def _local_model_digest(model):
    root = Path(model)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    try:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                name = path.relative_to(root).as_posix().encode("utf-8")
                digest.update(len(name).to_bytes(8, "big"))
                digest.update(name)
                digest.update(bytes.fromhex(_artifact_digest(path)))
    except OSError as exc:
        raise ValueError(f"Cannot fingerprint local model/tokenizer files: {exc}") from exc
    return digest.hexdigest()


def _provenance(network, model, revision, layer, *, local_model_sha256=None):
    return {
        "format_version": _FORMAT, "model": model, "revision": revision,
        "resolved_revision": getattr(network.config, "_commit_hash", None),
        "local_model_sha256": local_model_sha256 if local_model_sha256 is not None else _local_model_digest(model),
        "layer": layer, "width": _width(network), "representation": _REPRESENTATION,
        "prompt_format": "tokenizer_chat_template_if_available_else_plain",
        "trust_remote_code": False,
    }


def _atomic_npz(path, **arrays):
    np = _numpy()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _json_array(value):
    return _numpy().asarray(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True))


def _artifact_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _ResidualSteering:
    def __init__(self, direction, strength):
        self.direction = direction * strength

    def __call__(self, module, args, output):
        hidden = _hidden(output)
        if hidden.shape[-1] != self.direction.shape[0]:
            raise ValueError("Steering direction width does not match decoder output.")
        shifted = hidden + self.direction.to(device=hidden.device, dtype=hidden.dtype)
        return (shifted,) + output[1:] if isinstance(output, tuple) else shifted


def _read_steering(path, provenance, random_direction_seed=None):
    np = _numpy()
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            direction = np.asarray(data["direction"], dtype=np.float64).copy()
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid steering artifact: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("kind") != "ridge_condition_probe":
        raise ValueError("Steering requires a ridge_condition_probe artifact.")
    for key in ("format_version", "model", "revision", "resolved_revision", "local_model_sha256", "layer", "width", "representation", "prompt_format"):
        if metadata.get(key) != provenance.get(key):
            raise ValueError(f"Steering provenance mismatch for {key}: {metadata.get(key)!r} != {provenance.get(key)!r}.")
    if direction.shape != (provenance["width"],) or not np.isfinite(direction).all():
        raise ValueError("Steering direction has invalid width or nonfinite values.")
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Steering direction must have a finite nonzero norm.")
    if random_direction_seed is not None:
        random = np.random.default_rng(random_direction_seed).normal(size=direction.shape)
        direction = random * (norm / np.linalg.norm(random))
    return direction


class OpenWeightProvider:
    """Lazy, serialized inference; random control preserves the learned L2 norm.

    ``strength`` multiplies the raw-space classifier coefficient, not a unit
    vector. Positive strength points toward event rather than neutral prompts;
    this does not assume a direction of financial behavior or identify mediation.
    """

    def __init__(self, model, revision=None, device="cpu", temperature=0.0,
                 max_output_tokens=512, seed=42, steering_path=None, strength=0.0,
                 layer=None, *, random_direction_seed=None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("An explicit model name or local model directory is required.")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative.")
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer.")
        if not math.isfinite(strength):
            raise ValueError("strength must be finite.")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if random_direction_seed is not None and (isinstance(random_direction_seed, bool) or not isinstance(random_direction_seed, int) or random_direction_seed < 0):
            raise ValueError("random_direction_seed must be a nonnegative integer.")
        if steering_path is None and (strength != 0 or random_direction_seed is not None):
            raise ValueError("Nonzero steering and random controls require steering_path.")
        self.model_name, self.revision, self.device = model, revision, device
        self.temperature, self.max_output_tokens, self.seed = temperature, max_output_tokens, seed
        self.steering_path, self.strength, self.layer = steering_path, strength, layer
        self.random_direction_seed = random_direction_seed
        self._runtime = None
        self._direction = None
        self._lock = threading.Lock()
        self._local_model_sha256 = _local_model_digest(model)
        try:
            digest = _artifact_digest(steering_path) if steering_path else None
        except OSError as exc:
            raise ValueError(f"Cannot read steering artifact: {exc}") from exc
        self.identity = {
            "provider": "openweights", "requested_model": model, "revision": revision,
            "device": device, "steering_sha256": digest,
            "local_model_sha256": self._local_model_sha256,
            "generation": {"temperature": temperature, "max_output_tokens": max_output_tokens,
                           "seed": seed, "strength": strength, "layer": layer,
                           "random_direction_seed": random_direction_seed,
                           "steering_sha256": digest,
                           "steering_positions": "all_tokens"},
        }

    def _ensure_loaded(self):
        if self._runtime is not None:
            return
        runtime = _load_runtime(self.model_name, self.revision, self.device)
        torch, tokenizer, network = runtime
        layers = _decoder_layers(network)
        selected = self.layer
        if self.steering_path:
            if _artifact_digest(self.steering_path) != self.identity["steering_sha256"]:
                raise ValueError("Steering artifact changed after provider initialization.")
            if selected is None:
                np = _numpy()
                try:
                    with np.load(self.steering_path, allow_pickle=False) as data:
                        selected = json.loads(str(data["metadata"].item()))["layer"]
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid steering artifact: {exc}") from exc
        selected = _layer_index(-1 if selected is None else selected, len(layers))
        provenance = _provenance(network, self.model_name, self.revision, selected,
                                 local_model_sha256=self._local_model_sha256)
        if self.steering_path:
            direction = _read_steering(self.steering_path, provenance, self.random_direction_seed)
            self._direction = torch.as_tensor(
                direction, device=self.device, dtype=next(network.parameters()).dtype
            )
        self._selected_layer = selected
        self._runtime = runtime

    def prepare(self):
        """Resolve checkpoint/hook compatibility before journaling or scheduling trials."""
        with self._lock:
            self._ensure_loaded()
            network = self._runtime[2]
            self.identity["resolved_revision"] = getattr(network.config, "_commit_hash", None)
            self.identity["generation"]["resolved_layer"] = self._selected_layer

    def generate(self, prompt):
        return self._generate(prompt, self.seed)

    def generate_trial(self, trial):
        material = f"{self.seed}:{trial['trial_id']}".encode("utf-8")
        sampling_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)
        from .core import normalize_result
        return normalize_result(self._generate(trial["prompt"], sampling_seed), trial=trial)

    def _generate(self, prompt, sampling_seed):
        # A provider may be shared by run_trials worker threads. Temporary hooks
        # and sampling RNG state must not leak between concurrent requests.
        with self._lock:
            self._ensure_loaded()
            torch, tokenizer, network = self._runtime
            inputs = _inputs(tokenizer, prompt, self.device)
            count = int(inputs["input_ids"].shape[-1])
            if count < 1:
                raise ValueError("Tokenizer produced an empty prompt.")
            kwargs = {"max_new_tokens": self.max_output_tokens,
                      "do_sample": self.temperature > 0, "use_cache": True}
            if self.temperature > 0:
                kwargs["temperature"] = self.temperature
            pad = getattr(tokenizer, "pad_token_id", None)
            eos = getattr(tokenizer, "eos_token_id", None)
            if pad is not None or eos is not None:
                kwargs["pad_token_id"] = pad if pad is not None else eos
            hook = None
            try:
                if self._direction is not None and self.strength != 0:
                    hook = _decoder_layers(network)[self._selected_layer].register_forward_hook(
                        _ResidualSteering(self._direction, self.strength)
                    )
                cuda_devices = []
                target_device = torch.device(self.device)
                if target_device.type == "cuda":
                    cuda_devices = [target_device.index if target_device.index is not None else torch.cuda.current_device()]
                with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices):
                    torch.random.default_generator.manual_seed(sampling_seed)
                    if cuda_devices:
                        with torch.cuda.device(cuda_devices[0]):
                            torch.cuda.manual_seed(sampling_seed)
                    generated = network.generate(**inputs, **kwargs)
                continuation = generated[0, count:]
                raw = tokenizer.decode(continuation, skip_special_tokens=True)
            finally:
                if hook is not None:
                    hook.remove()
            result = {
                "status": "ok", "raw_text": raw, "payload": None,
                "provider": "openweights", "requested_model": self.model_name,
                "model_version": getattr(network.config, "_commit_hash", None) or self._local_model_sha256 or self.revision,
                "generation": dict(self.identity["generation"], resolved_layer=self._selected_layer),
                "usage": {"input_tokens": count, "output_tokens": int(continuation.shape[-1])},
                "attempts": 1, "error": None,
                "sampling_seed": sampling_seed,
            }
            from .core import normalize_result
            return normalize_result(result)


def extract_activations(trials_path, output_path, *, model, revision=None, layer=-1, device="cpu"):
    """Capture prompt-only residuals without generating or priming decision runs."""
    from .core import read_jsonl, validate_trials
    np = _numpy()
    trials = read_jsonl(trials_path)
    validate_trials(trials)
    if not trials:
        raise ValueError("Activation extraction requires at least one trial.")
    torch, tokenizer, network = _load_runtime(model, revision, device)
    layers = _decoder_layers(network)
    selected = _layer_index(layer, len(layers))
    provenance = _provenance(network, model, revision, selected)
    features = np.empty((len(trials), provenance["width"]), dtype=np.float32)
    captured = []

    def capture(module, args, output):
        hidden = _hidden(output)
        if hidden.shape[0] != 1 or hidden.shape[-1] != provenance["width"]:
            raise ValueError("Unexpected residual shape during activation extraction.")
        captured.append(hidden[0, -1].detach().float().cpu().numpy().copy())

    hook = layers[selected].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            for index, trial in enumerate(trials):
                captured.clear()
                inputs = _inputs(tokenizer, trial["prompt"], device)
                network(**inputs, use_cache=False)
                if len(captured) != 1:
                    raise ValueError("Unsupported architecture: selected block must run exactly once per prompt.")
                features[index] = captured[0]
    finally:
        hook.remove()
    if not np.isfinite(features).all():
        raise ValueError("Model produced nonfinite residual activations.")
    metadata = dict(provenance, kind="prompt_activations", n_trials=len(trials),
                    label="calendar salience:condition:event=1,neutral=0",
                    event_valences=sorted({row["event_valence"] for row in trials}))
    _atomic_npz(output_path, features=features,
                labels=np.asarray([int(row["condition"] == "event") for row in trials], dtype=np.int8),
                trial_ids=np.asarray([row["trial_id"] for row in trials]),
                cluster_ids=np.asarray([row["cluster_id"] for row in trials]),
                splits=np.asarray([row["split"] for row in trials]),
                trials=_json_array(trials), metadata=_json_array(metadata))
    return dict(metadata, output=str(output_path))


def _ridge_fit(np, x, labels, ridge):
    target = 2.0 * labels - 1.0
    mean_x, mean_y = x.mean(axis=0), float(target.mean())
    centered_x, centered_y = x - mean_x, target - mean_y
    n, width = x.shape
    if width <= n:
        gram = centered_x.T @ centered_x
        gram.flat[::width + 1] += ridge
        coefficient = np.linalg.solve(gram, centered_x.T @ centered_y)
    else:
        gram = centered_x @ centered_x.T
        gram.flat[::n + 1] += ridge
        coefficient = centered_x.T @ np.linalg.solve(gram, centered_y)
    return coefficient, float(mean_y - mean_x @ coefficient)


def _scores(np, x, labels, clusters, coefficient, intercept):
    predicted = (x @ coefficient + intercept >= 0).astype(np.int8)
    correct = predicted == labels
    cluster_accuracy = [float(correct[clusters == key].mean()) for key in np.unique(clusters)]
    balanced = [float(correct[labels == value].mean()) for value in (0, 1)]
    return {"n": int(len(labels)), "n_clusters": int(len(cluster_accuracy)),
            "accuracy": float(correct.mean()), "balanced_accuracy": float(np.mean(balanced)),
            "cluster_macro_accuracy": float(np.mean(cluster_accuracy))}


def fit_probe(activations_path, output_path, *, ridge=1.0):
    """Fit event-vs-neutral ridge classifier with strictly train-only scaling.

    Dev/test are reported without tuning. A fixed-seed cluster-level label flip
    and training-majority classifier are negative/baseline controls, not a
    calibrated permutation significance test. A fit accepts one event valence.
    Decodability does not establish mediation of financial decisions.
    """
    np = _numpy()
    if isinstance(ridge, bool) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and strictly positive.")
    try:
        with np.load(activations_path, allow_pickle=False) as data:
            x = np.asarray(data["features"], dtype=np.float64)
            labels = np.asarray(data["labels"])
            ids = np.asarray(data["trial_ids"])
            clusters, splits = np.asarray(data["cluster_ids"]), np.asarray(data["splits"])
            metadata = json.loads(str(data["metadata"].item()))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid activation artifact: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("kind") != "prompt_activations" or metadata.get("format_version") != _FORMAT or metadata.get("representation") != _REPRESENTATION:
        raise ValueError("Expected supported prompt-only activation artifact.")
    valences = metadata.get("event_valences")
    if valences not in (["auspicious"], ["avoidance"]):
        raise ValueError("Fit one event_valence at a time: subset whole trial pairs to auspicious or avoidance before extraction.")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Features must be a nonempty finite matrix.")
    if x.shape[1] != metadata.get("width"):
        raise ValueError("Activation metadata width does not match features.")
    if any(values.shape != (x.shape[0],) for values in (labels, ids, clusters, splits)):
        raise ValueError("Activation metadata arrays must match the number of rows.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate trial IDs in activation artifact.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Probe labels must be event=1 or neutral=0.")
    if not np.isin(splits, ["train", "dev", "test"]).all():
        raise ValueError("Every activation must have train, dev, or test split.")
    if any(not str(value).strip() for value in clusters):
        raise ValueError("Cluster IDs cannot be empty.")
    owners = {}
    for cluster, split in zip(clusters.tolist(), splits.tolist()):
        if cluster in owners and owners[cluster] != split:
            raise ValueError(f"Cluster leakage: {cluster!r} occurs in multiple splits.")
        owners[cluster] = split
    masks = {name: splits == name for name in ("train", "dev", "test")}
    for name, mask in masks.items():
        if not mask.any() or len(np.unique(labels[mask])) != 2:
            raise ValueError(f"Split {name!r} must contain both event and neutral observations.")
    train = masks["train"]
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (x - mean) / scale
    coefficient, intercept = _ridge_fit(np, standardized[train], labels[train], ridge)
    # Permute whole clusters' paired condition coding, rather than pretending
    # translated/paraphrased repetitions are independent exchangeable samples.
    shuffled = labels[train].copy()
    rng = np.random.default_rng(42)
    train_clusters = clusters[train]
    flipped = 0
    for cluster in np.unique(train_clusters):
        if rng.integers(0, 2):
            mask = train_clusters == cluster
            shuffled[mask] = 1 - shuffled[mask]
            flipped += 1
    null_coefficient, null_intercept = _ridge_fit(np, standardized[train], shuffled, ridge)
    majority = int(labels[train].mean() >= 0.5)
    reports = {}
    for name, mask in masks.items():
        reports[name] = _scores(np, standardized[mask], labels[mask], clusters[mask], coefficient, intercept)
        reports[name]["majority_baseline_accuracy"] = float((labels[mask] == majority).mean())
        reports[name]["cluster_label_flip_control"] = _scores(
            np, standardized[mask], labels[mask], clusters[mask], null_coefficient, null_intercept
        )
    direction = coefficient / scale
    raw_intercept = float(intercept - mean @ direction)
    if not np.isfinite(direction).all() or not math.isfinite(raw_intercept):
        raise ValueError("Probe fit produced nonfinite coefficients.")
    summary = {
        "ridge": float(ridge), "splits": reports, "control_seed": 42,
        "event_valence": valences[0], "label": "calendar salience:event=1,neutral=0",
        "control_flipped_train_clusters": flipped, "train_only_scaling": True,
        "direction_l2_norm": float(np.linalg.norm(direction)),
        "interpretation": "Correlational condition decodability; not mediation or evidence of a universal mechanism.",
        "positive_direction": "event versus neutral calendar salience, not auspiciousness or necessarily increased investment",
        "steering_positions": "all_tokens", "selection": "fixed ridge; no dev/test tuning",
    }
    provenance = {key: metadata.get(key) for key in (
        "format_version", "model", "revision", "resolved_revision", "local_model_sha256", "layer", "width",
        "representation", "prompt_format", "trust_remote_code"
    )}
    artifact_metadata = dict(provenance, kind="ridge_condition_probe", report=summary,
                             activation_sha256=_artifact_digest(activations_path))
    _atomic_npz(output_path, direction=direction, intercept=np.asarray(raw_intercept),
                scaler_mean=mean, scaler_scale=scale, standardized_coefficient=coefficient,
                standardized_intercept=np.asarray(intercept), metadata=_json_array(artifact_metadata))
    return dict(summary, output=str(output_path), provenance=provenance)
