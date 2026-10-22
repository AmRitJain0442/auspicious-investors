"""Offline invariants; optional runtime tests use a real, tiny random Llama.

Synthetic fixtures below test numerical behavior only. They are not observations,
validated calendar stimuli, or evidence about any trained model's behavior.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from auspicious import openweights as ow

try:
    import numpy as np
except ImportError:
    np = None


def provenance(width=3):
    return {
        "format_version": 1, "model": "local-fixture", "revision": "fixture-revision",
        "resolved_revision": "fixture-commit", "layer": 1, "width": width,
        "local_model_sha256": None,
        "representation": "decoder_block_output_prompt_final_token",
        "prompt_format": "tokenizer_chat_template_if_available_else_plain",
        "trust_remote_code": False,
    }


def activation_fixture(path):
    rows, labels, clusters, splits = [], [], [], []
    for split, count in (("train", 3), ("dev", 2), ("test", 2)):
        for cluster in range(count):
            for label in (0, 1, 0, 1):
                nuisance = cluster if split == "train" else 10000 + cluster
                rows.append([2 * (2 * label - 1), nuisance, 7])
                labels.append(label)
                clusters.append(f"{split}-{cluster}")
                splits.append(split)
    arrays = {
        "features": np.asarray(rows, dtype=float), "labels": np.asarray(labels),
        "cluster_ids": np.asarray(clusters), "splits": np.asarray(splits),
        "trial_ids": np.asarray([f"trial-{i}" for i in range(len(rows))]),
        "metadata": np.asarray(json.dumps(dict(provenance(), kind="prompt_activations", event_valences=["auspicious"]))),
    }
    np.savez(path, **arrays)
    return arrays


class ProviderConfigurationTests(unittest.TestCase):
    def test_import_and_constructor_are_lazy(self):
        script = (
            "import sys; from auspicious.openweights import OpenWeightProvider; "
            "p=OpenWeightProvider('explicit-model'); "
            "assert all(x not in sys.modules for x in ('torch','transformers','numpy')); "
            "assert p.identity['requested_model']=='explicit-model'"
        )
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_configuration_rejects_unbacked_interventions(self):
        for kwargs in ({"strength": 1}, {"random_direction_seed": 0},
                       {"temperature": float("nan")}, {"max_output_tokens": 0},
                       {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ow.OpenWeightProvider("explicit-model", **kwargs)

    def test_unsupported_architecture_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
            ow._decoder_layers(object())

    def test_layer_bounds_and_negative_resolution(self):
        self.assertEqual(ow._layer_index(-1, 3), 2)
        self.assertEqual(ow._layer_index(-3, 3), 0)
        for index in (3, -4, True, 1.5):
            with self.assertRaises(ValueError):
                ow._layer_index(index, 3)

    def test_local_model_and_tokenizer_bytes_bind_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.safetensors").write_bytes(b"synthetic-weight-fixture")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text('{"fixture":1}', encoding="utf-8")
            first = ow.OpenWeightProvider(folder).identity
            self.assertEqual(first, ow.OpenWeightProvider(folder).identity)
            tokenizer.write_text('{"fixture":2}', encoding="utf-8")
            second = ow.OpenWeightProvider(folder).identity
            self.assertNotEqual(first["local_model_sha256"], second["local_model_sha256"])
            (root / "model.safetensors").write_bytes(b"changed-weight-fixture")
            third = ow.OpenWeightProvider(folder).identity
            self.assertNotEqual(second["local_model_sha256"], third["local_model_sha256"])


@unittest.skipIf(np is None, "numpy optional extra is not installed")
class ProbeInvariantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "activations.npz"
        self.output = Path(self.temp.name) / "probe.npz"
        self.arrays = activation_fixture(self.source)

    def test_train_only_scaling_and_raw_direction_equivalence(self):
        report = ow.fit_probe(self.source, self.output)
        with np.load(self.output, allow_pickle=False) as artifact:
            np.testing.assert_allclose(artifact["scaler_mean"], [0, 1, 7])
            np.testing.assert_allclose(artifact["scaler_scale"], [2, np.sqrt(2 / 3), 1])
            standardized = (self.arrays["features"] - artifact["scaler_mean"]) / artifact["scaler_scale"]
            raw_scores = self.arrays["features"] @ artifact["direction"] + artifact["intercept"]
            scaled_scores = standardized @ artifact["standardized_coefficient"] + artifact["standardized_intercept"]
            np.testing.assert_allclose(raw_scores, scaled_scores)
            metadata = json.loads(artifact["metadata"].item())
            self.assertEqual(metadata["activation_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        for split in ("train", "dev", "test"):
            self.assertEqual(report["splits"][split]["accuracy"], 1.0)
            self.assertEqual(report["splits"][split]["majority_baseline_accuracy"], 0.5)
            self.assertIn("cluster_label_flip_control", report["splits"][split])
        self.assertIn("not mediation", report["interpretation"])

    def test_no_cluster_leakage_even_with_distinct_trial_ids(self):
        self.arrays["cluster_ids"][self.arrays["splits"] == "test"] = "train-0"
        np.savez(self.source, **self.arrays)
        with self.assertRaisesRegex(ValueError, "Cluster leakage"):
            ow.fit_probe(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_reject_duplicate_trials_nonfinite_and_missing_class(self):
        mutations = (
            ("duplicate", lambda a: a["trial_ids"].__setitem__(1, a["trial_ids"][0])),
            ("nonfinite", lambda a: a["features"].__setitem__((0, 0), np.nan)),
            ("missing_class", lambda a: a["labels"].__setitem__(a["splits"] == "dev", 1)),
        )
        for name, change in mutations:
            with self.subTest(name=name):
                arrays = activation_fixture(self.source)
                change(arrays)
                np.savez(self.source, **arrays)
                with self.assertRaises(ValueError):
                    ow.fit_probe(self.source, self.output)

    def test_heldout_labels_do_not_change_fitted_direction(self):
        ow.fit_probe(self.source, self.output)
        with np.load(self.output, allow_pickle=False) as artifact:
            original = artifact["direction"].copy()
        heldout = self.arrays["splits"] != "train"
        self.arrays["labels"][heldout] = 1 - self.arrays["labels"][heldout]
        np.savez(self.source, **self.arrays)
        report = ow.fit_probe(self.source, self.output)
        with np.load(self.output, allow_pickle=False) as artifact:
            np.testing.assert_array_equal(original, artifact["direction"])
        self.assertEqual(report["splits"]["test"]["accuracy"], 0.0)

    def test_dual_and_primal_ridge_agree_with_direct_solution(self):
        rng = np.random.default_rng(4)
        for n, width in ((8, 3), (5, 9)):
            x = rng.normal(size=(n, width))
            labels = np.arange(n) % 2
            coefficient, intercept = ow._ridge_fit(np, x, labels, 0.7)
            centered = x - x.mean(axis=0)
            target = 2 * labels - 1
            expected = np.linalg.solve(centered.T @ centered + 0.7 * np.eye(width),
                                       centered.T @ (target - target.mean()))
            np.testing.assert_allclose(coefficient, expected, atol=1e-12)
            self.assertAlmostEqual(intercept, target.mean() - x.mean(axis=0) @ expected)

    def test_mixed_valence_is_not_presented_as_auspiciousness(self):
        metadata = json.loads(self.arrays["metadata"].item())
        metadata["event_valences"] = ["auspicious", "avoidance"]
        self.arrays["metadata"] = np.asarray(json.dumps(metadata))
        np.savez(self.source, **self.arrays)
        with self.assertRaisesRegex(ValueError, "one event_valence"):
            ow.fit_probe(self.source, self.output)

    def test_provenance_mismatches_all_fail(self):
        ow.fit_probe(self.source, self.output)
        for field, value in (("model", "other"), ("revision", None),
                             ("resolved_revision", "new-commit"), ("local_model_sha256", "changed-bytes"), ("layer", 0),
                             ("width", 4), ("representation", "generated_answer")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "provenance mismatch"):
                ow._read_steering(self.output, dict(provenance(), **{field: value}))

    def test_random_control_is_seeded_and_norm_matched(self):
        ow.fit_probe(self.source, self.output)
        learned = ow._read_steering(self.output, provenance())
        random1 = ow._read_steering(self.output, provenance(), 7)
        random2 = ow._read_steering(self.output, provenance(), 7)
        other = ow._read_steering(self.output, provenance(), 8)
        np.testing.assert_array_equal(random1, random2)
        self.assertFalse(np.allclose(random1, learned))
        self.assertFalse(np.allclose(random1, other))
        self.assertAlmostEqual(float(np.linalg.norm(random1)), float(np.linalg.norm(learned)))

    def test_probe_refuses_nonpositive_ridge(self):
        for ridge in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                ow.fit_probe(self.source, self.output, ridge=ridge)


@unittest.skipIf(np is None, "numpy optional extra is not installed")
class RealDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError as exc:
            raise unittest.SkipTest(f"Optional torch/transformers unavailable: {exc}")
        cls.torch = torch
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2, max_position_embeddings=128,
                             bos_token_id=1, eos_token_id=None, pad_token_id=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            cls.network = LlamaForCausalLM(config).eval()

        class CharacterTokenizer:
            chat_template = None
            pad_token_id = 0
            eos_token_id = None

            def __call__(self, text, **kwargs):
                ids = torch.tensor([[1] + [2 + ord(char) % 29 for char in text]])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

            def decode(self, tokens, **kwargs):
                # Actual generated token IDs, deliberately not a fabricated task answer.
                return json.dumps(tokens.tolist())

        cls.tokenizer = CharacterTokenizer()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.loader = patch.object(ow, "_load_runtime", return_value=(self.torch, self.tokenizer, self.network))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def _steering_artifact(self):
        metadata = ow._provenance(self.network, "local-fixture", None, 1)
        metadata["kind"] = "ridge_condition_probe"
        direction = np.linspace(-0.4, 0.7, 16)
        path = self.root / "steer.npz"
        np.savez(path, direction=direction, metadata=np.asarray(json.dumps(metadata)))
        return path, direction

    def test_real_residual_hook_signed_zero_and_tuple_preservation(self):
        torch = self.torch
        inputs = ow._inputs(self.tokenizer, "fixture", "cpu")
        block = self.network.model.layers[-1]
        captured = []
        capture = lambda module, args, output: captured.append(ow._hidden(output).detach().clone())
        hook = block.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                baseline_logits = self.network(**inputs).logits.clone()
        finally:
            hook.remove()
        baseline = captured.pop()
        direction = torch.arange(16, dtype=torch.float32) / 16
        for strength in (-2.0, 0.0, 2.0):
            steering = block.register_forward_hook(ow._ResidualSteering(direction, strength))
            capture_hook = block.register_forward_hook(capture)
            try:
                with torch.inference_mode():
                    logits = self.network(**inputs).logits
            finally:
                steering.remove()
                capture_hook.remove()
            shifted = captured.pop()
            torch.testing.assert_close(shifted, baseline + strength * direction)
            if strength == 0:
                torch.testing.assert_close(logits, baseline_logits, atol=0, rtol=0)
            else:
                self.assertFalse(torch.equal(logits, baseline_logits))
        marker = object()
        output = ow._ResidualSteering(direction, 1)(None, None, (baseline, marker))
        self.assertIs(output[1], marker)
        torch.testing.assert_close(output[0], baseline + direction)
        self.assertEqual(len(block._forward_hooks), 0)

    def test_provider_generates_and_zero_control_matches_baseline(self):
        path, _ = self._steering_artifact()
        plain = ow.OpenWeightProvider("local-fixture", max_output_tokens=3)
        zero = ow.OpenWeightProvider("local-fixture", max_output_tokens=3, steering_path=path, strength=0)
        before = self.torch.random.get_rng_state().clone()
        baseline, control = plain.generate("abc"), zero.generate("abc")
        self.assertEqual(baseline["raw_text"], control["raw_text"])
        self.assertEqual(baseline["usage"], {"input_tokens": 4, "output_tokens": 3})
        self.assertEqual(baseline["status"], "invalid")
        self.assertEqual(baseline["generation"]["resolved_layer"], 1)
        self.torch.testing.assert_close(before, self.torch.random.get_rng_state())
        self.assertEqual(len(self.network.model.layers[1]._forward_hooks), 0)

    def test_fast_tokenizer_segment_ids_do_not_break_causal_generation(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        backend = Tokenizer(WordLevel({"[UNK]": 0, "financial": 1, "decision": 2}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]")
        prompt = "financial decision"
        with self.torch.inference_mode():
            expected = self.network.generate(
                **tokenizer(prompt, return_tensors="pt", return_token_type_ids=False),
                max_new_tokens=3, do_sample=False, pad_token_id=0,
            )
        with patch.object(ow, "_load_runtime", return_value=(self.torch, tokenizer, self.network)):
            result = ow.OpenWeightProvider("local-fixture", max_output_tokens=3).generate(prompt)
        self.assertEqual(result["raw_text"], tokenizer.decode(expected[0, 2:], skip_special_tokens=True))
        self.assertEqual(result["usage"]["input_tokens"], 2)
        self.assertEqual(result["status"], "invalid")

    def test_generation_failure_removes_hook(self):
        path, _ = self._steering_artifact()
        provider = ow.OpenWeightProvider("local-fixture", steering_path=path, strength=-1)
        with patch.object(self.network, "generate", side_effect=RuntimeError("intentional failure")):
            with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                provider.generate("abc")
        self.assertEqual(len(self.network.model.layers[1]._forward_hooks), 0)

    def test_changed_artifact_is_not_used(self):
        path, _ = self._steering_artifact()
        provider = ow.OpenWeightProvider("local-fixture", steering_path=path)
        path.write_bytes(b"changed artifact")
        with self.assertRaisesRegex(ValueError, "changed after"):
            provider.generate("abc")

    def test_trial_sampling_seeds_differ_and_reproduce_without_splitting_settings(self):
        provider = ow.OpenWeightProvider("local-fixture", temperature=0.8, max_output_tokens=3, seed=19)
        first = provider.generate_trial({"trial_id": "replicate-1", "prompt": "abc"})
        second = provider.generate_trial({"trial_id": "replicate-2", "prompt": "abc"})
        replay = provider.generate_trial({"trial_id": "replicate-1", "prompt": "abc"})
        self.assertNotEqual(first["sampling_seed"], second["sampling_seed"])
        self.assertEqual(first["sampling_seed"], replay["sampling_seed"])
        self.assertEqual(first["raw_text"], replay["raw_text"])
        self.assertEqual(first["generation"], second["generation"])
        self.assertEqual(first["generation"]["seed"], 19)
        self.assertIn("steering_sha256", first["generation"])

    def test_extraction_is_prompt_only_and_matches_final_token(self):
        from auspicious.core import write_jsonl
        trials = []
        for condition, prompt in (("event", "festival fixture"), ("neutral", "ordinary fixture")):
            trials.append({
                "trial_id": hashlib.sha256(prompt.encode()).hexdigest(), "pair_id": "fixture-pair",
                "cluster_id": "fixture-scenario", "scenario_id": "fixture-scenario", "split": "train",
                "language": "en", "festival": "dhanteras", "condition": condition,
                "event_valence": "auspicious", "cue": "festival_named", "role": "investor",
                "persona_id": "fixture-persona", "paraphrase_id": "0", "replicate": 0,
                "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "review_status": "draft", "review": {},
            })
        source, output = self.root / "trials.jsonl", self.root / "activations.npz"
        write_jsonl(source, trials)
        with patch.object(self.network, "generate", side_effect=AssertionError("Extraction must not generate")):
            report = ow.extract_activations(source, output, model="local-fixture", layer=-1)
        self.assertEqual(report["layer"], 1)
        expected = []
        hook = self.network.model.layers[1].register_forward_hook(
            lambda module, args, output: expected.append(ow._hidden(output)[0, -1].detach().clone())
        )
        try:
            with self.torch.inference_mode():
                for trial in trials:
                    self.network(**ow._inputs(self.tokenizer, trial["prompt"], "cpu"), use_cache=False)
        finally:
            hook.remove()
        with np.load(output, allow_pickle=False) as artifact:
            np.testing.assert_allclose(artifact["features"], self.torch.stack(expected).numpy())
            np.testing.assert_array_equal(artifact["labels"], [1, 0])
            self.assertEqual(json.loads(artifact["trials"].item()), trials)
        self.assertEqual(len(self.network.model.layers[1]._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
