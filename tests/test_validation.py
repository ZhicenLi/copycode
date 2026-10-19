"""Regression tests for the validation holes the external audit found."""

from __future__ import annotations

import unittest

import numpy as np

from fusion.config import Config
from fusion.data.manifest import Manifest, build_manifest, validate
from fusion.data.specs import get_spec

try:
    import sklearn  # noqa: F401
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - depends on the environment
    HAS_SKLEARN = False


def _base(**overrides):
    raw = {
        "data": {"dataset": "cifar10", "num_clients": 10, "dirichlet_alpha": 0.5},
        "runtime": {"arms": ["fedds"], "exp_name": "x"},
    }
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)
    return raw


class TestConfigValidation(unittest.TestCase):
    def test_unknown_section_is_rejected(self):
        with self.assertRaises(KeyError):
            Config.from_dict({**_base(), "fedd": {"rounds": 5}})

    def test_last_k_must_be_positive_and_bounded(self):
        with self.assertRaises(ValueError):
            Config.from_dict(_base(fedds={"last_k": 0}))
        with self.assertRaises(ValueError):
            Config.from_dict(_base(fedds={"rounds": 5, "last_k": 6}))
        cfg = Config.from_dict(_base(fedds={"rounds": 200, "last_k": 40}))
        self.assertEqual(cfg.fedds.last_k, 40)

    def test_iteration1_must_be_positive(self):
        with self.assertRaises(ValueError):
            Config.from_dict(_base(fedcorr={"iteration1": 0}))

    def test_lid_k_must_be_at_least_two(self):
        with self.assertRaises(ValueError):
            Config.from_dict(_base(fedcorr={"lid_k": 1}))

    def test_single_client_is_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict({
                "data": {"dataset": "cifar10", "num_clients": 1},
                "runtime": {"arms": ["fedavg"], "exp_name": "x"},
            })

    def test_fedds_arm_needs_two_annotators(self):
        with self.assertRaises(ValueError):
            Config.from_dict({
                "data": {"dataset": "cifar10", "num_clients": 8},
                "fedds": {"frac": 0.1},          # int(0.8) -> 1 annotator
                "runtime": {"arms": ["fedds"], "exp_name": "x"},
            })

    def test_frac_map_must_be_in_unit_interval(self):
        with self.assertRaises(ValueError):
            Config.from_dict(_base(fedds={"frac_by_num_clients": {10: -0.5}}))

    def test_integer_fields_reject_floats(self):
        with self.assertRaises(ValueError):
            Config.from_dict(_base(fedds={"rounds": 1.5}))

    def test_non_finite_alpha_is_rejected(self):
        with self.assertRaises(ValueError):
            Config.from_dict(_base(data={"dirichlet_alpha": float("nan")}))

    def test_cli_style_strings_are_parsed(self):
        cfg = Config.from_dict(_base(
            data={"noise_levels": "0.2,0.5"},
            fedds={"frac_by_num_clients": "10:0.5,50:0.2"},
            runtime={"arms": "fedds,fedavg"},
        ))
        self.assertEqual(cfg.data.noise_levels, (0.2, 0.5))
        self.assertEqual(cfg.fedds.frac_by_num_clients, {10: 0.5, 50: 0.2})
        self.assertEqual(cfg.runtime.arms, ("fedds", "fedavg"))

    def test_partition_and_preprocess_are_validated(self):
        cfg = Config.from_dict(_base(
            data={"partition": "iid"},
            fedcorr={"preprocess": "standard"},
            fedds={"preprocess": "raw"},
        ))
        self.assertEqual(cfg.data.partition, "iid")
        self.assertEqual(cfg.fedcorr.preprocess, "standard")
        self.assertEqual(cfg.fedds.preprocess, "raw")   # the two stages may differ
        with self.assertRaises(ValueError):
            Config.from_dict(_base(data={"partition": "noniid"}))
        with self.assertRaises(ValueError):
            Config.from_dict(_base(fedds={"preprocess": "imagenet"}))
        with self.assertRaises(ValueError):
            Config.from_dict(_base(fedcorr={"preprocess": "imagenet"}))


class TestPaperProtocol(unittest.TestCase):
    """The FedDS paper's data pipeline: IID split + raw x/255 preprocessing."""

    def test_iid_partition_is_equal_size_and_covers_everything(self):
        from fusion.data.manifest import iid_partition

        rng = np.random.default_rng(0)
        parts = iid_partition(1000, 100, rng)
        sizes = [len(part) for part in parts]
        self.assertEqual(sum(sizes), 1000)
        self.assertEqual(len(set(sizes)), 1)          # exactly equal when it divides
        joined = np.sort(np.concatenate(parts))
        np.testing.assert_array_equal(joined, np.arange(1000))

    def test_iid_partition_gives_remainder_to_last_client(self):
        from fusion.data.manifest import iid_partition

        parts = iid_partition(1003, 100, np.random.default_rng(0))
        self.assertEqual(len(parts[-1]), 13)
        self.assertTrue(all(len(p) == 10 for p in parts[:-1]))

    def test_raw_preprocess_keeps_pixels_in_unit_interval(self):
        from PIL import Image

        from fusion.data.loaders import build_transform

        spec = get_spec("cifar10")
        transform = build_transform(spec, "train", "raw")
        image = Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8))
        tensor = transform(image)
        self.assertEqual(tuple(tensor.shape), (3, 32, 32))
        self.assertAlmostEqual(float(tensor.mean()), 200 / 255, places=5)
        self.assertAlmostEqual(float(tensor.max()), 200 / 255, places=5)

    def test_standard_preprocess_centres_the_same_pixels(self):
        from PIL import Image

        from fusion.data.loaders import build_transform

        spec = get_spec("cifar10")
        transform = build_transform(spec, "eval", "standard")
        image = Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8))
        tensor = transform(image)
        self.assertGreater(float(tensor.abs().max()), 0.5)   # normalisation applied

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is not installed")
    def test_sklearn_public_split_is_stratified_and_disjoint(self):
        from fusion.data.manifest import sklearn_public_split

        labels = np.repeat(np.arange(10), 5000)
        public, rest = sklearn_public_split(labels, 0.1, 42)
        self.assertEqual(len(public), 5000)
        self.assertEqual(len(rest), 45000)
        self.assertEqual(np.bincount(labels[public], minlength=10).tolist(), [500] * 10)
        self.assertEqual(len(np.intersect1d(public, rest)), 0)
        again, _ = sklearn_public_split(labels, 0.1, 42)
        np.testing.assert_array_equal(public, again)         # deterministic

    def test_explicit_noise_rates_are_used_verbatim(self):
        import json
        import tempfile

        from fusion.data.manifest import noise_report

        labels = np.repeat(np.arange(10), 200)               # 2000 samples
        rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        spec = get_spec("cifar10")
        manifest = build_manifest(
            clean_labels=labels, spec=spec, num_clients=10, dirichlet_alpha=0.5,
            public_ratio=0.1, noise_levels=[0.5], split_seed=42, noise_seed=1,
            min_client_samples=5, partition="iid", noise_rates=rates,
        )
        self.assertEqual(manifest.meta["client_noise_rates"], rates)
        self.assertEqual([row["requested_noise"] for row in noise_report(manifest)], rates)

    def test_noise_flip_count_truncates_like_the_official_code(self):
        from fusion.data.manifest import assign_noise

        labels = np.zeros(10, dtype=np.int64)
        client_id = np.zeros(10, dtype=np.int32)
        # 0.29 * 10 = 2.9 -> int() = 2, round() would give 3
        noisy = assign_noise(labels, client_id, [0.29], num_classes=10,
                             rng=np.random.default_rng(0))
        self.assertEqual(int((noisy != labels).sum()), 2)

    def test_noise_rates_length_is_validated(self):
        with self.assertRaises(ValueError):
            Config.from_dict(_base(data={"noise_rates": [0.1, 0.2]}))   # 10 clients


class TestManifestValidation(unittest.TestCase):
    def _build(self, n=2000, clients=10):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 10, n)
        return build_manifest(
            clean_labels=labels, spec=get_spec("cifar10"), num_clients=clients,
            dirichlet_alpha=0.5, public_ratio=0.1, noise_levels=[0.5],
            split_seed=1, noise_seed=2, min_client_samples=5,
        )

    def test_checksum_covers_clean_labels(self):
        manifest = self._build()
        before = manifest.digest()
        manifest.clean_label = manifest.clean_label.copy()
        manifest.clean_label[0] = (manifest.clean_label[0] + 1) % 10
        self.assertNotEqual(manifest.digest(), before)

    def test_checksum_covers_noise_rate(self):
        manifest = self._build()
        before = manifest.digest()
        manifest.noise_rate = manifest.noise_rate.copy()
        manifest.noise_rate[0] = 0.9
        self.assertNotEqual(manifest.digest(), before)

    def test_inconsistent_client_id_is_rejected(self):
        manifest = self._build()
        manifest.client_id = manifest.client_id.copy()
        public_row = int(manifest.public_indices()[0])
        manifest.client_id[public_row] = 0          # public row claims a client
        problems = validate(manifest, strict=False)
        self.assertTrue(any("disagree" in p for p in problems), problems)

    def test_out_of_range_client_id_is_rejected(self):
        manifest = self._build()
        manifest.client_id = manifest.client_id.copy()
        manifest.client_id[0] = 999
        problems = validate(manifest, strict=False)
        self.assertTrue(any(">= num_clients" in p for p in problems), problems)

    def test_empty_public_split_is_impossible(self):
        rng = np.random.default_rng(1)
        labels = np.arange(10)                      # one sample per class
        with self.assertRaises(ValueError):
            build_manifest(
                clean_labels=labels, spec=get_spec("cifar10"), num_clients=2,
                dirichlet_alpha=0.5, public_ratio=0.9, noise_levels=[0.5],
                split_seed=1, noise_seed=2, min_client_samples=1,
            )

    def test_short_rates_are_rejected(self):
        from fusion.data.manifest import assign_noise

        labels = np.zeros(10, dtype=np.int64)
        client_id = np.array([-1, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            assign_noise(labels, client_id, [0.5], num_classes=10, rng=np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
