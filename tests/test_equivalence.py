"""Equivalence tests against the official FedCorr / FedDS source files.

These load the *original* modules straight from `code/fedcorr` and
`code/fedds_repro` and compare numerical output with this project's
re-implementations. A green run is the concrete answer to "is the port correct",
rather than "it looks right".

Skipped automatically when the original sources or scipy are unavailable.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORIGINAL_ROOT = os.path.join(os.path.dirname(ROOT), "code")

try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _original(path: str):
    full = os.path.join(ORIGINAL_ROOT, path)
    return full if os.path.exists(full) else None


class _OgridProxy:
    """numpy >= 2 returns a tuple from ``np.ogrid``, while the official FedCorr
    code assigns into it (``idx[1] = ...``) and therefore crashes. This proxy
    restores the old list behaviour so the *formula* can still be compared."""

    def __init__(self, real):
        self._real = real

    def __getitem__(self, key):
        return list(self._real[key])


def _official_lid(module, features, k):
    try:
        return module.lid_term(features, features, k=k)
    except TypeError:                     # numpy >= 2: ogrid returns a tuple
        real = np.ogrid
        np.ogrid = _OgridProxy(real)
        try:
            return module.lid_term(features, features, k=k)
        finally:
            np.ogrid = real


@unittest.skipUnless(HAS_SCIPY, "scipy is required for the official LID implementation")
class TestLIDEquivalence(unittest.TestCase):
    def test_matches_official_lid_term(self):
        from fusion.core.lid import lid_scores

        path = _original(os.path.join("fedcorr", "util", "util.py"))
        if path is None:
            self.skipTest("original FedCorr sources not present")
        original = _load(path, "orig_fedcorr_util")

        rng = np.random.default_rng(0)
        for shape, k in [((40, 10), 20), ((80, 5), 10), ((25, 32), 5)]:
            features = rng.random(shape).astype(np.float32)
            expected = _official_lid(original, features, k)
            actual = lid_scores(features, k=k)
            np.testing.assert_allclose(
                actual, expected, rtol=1e-4, atol=1e-6,
                err_msg=f"LID mismatch for shape={shape}, k={k}",
            )


@unittest.skipUnless(HAS_SCIPY, "torch model comparison needs the same runtime")
class TestModelEquivalence(unittest.TestCase):
    def test_resnet18_matches_official_architecture(self):
        from fusion.models import build_model

        path = _original(os.path.join("fedcorr", "model", "model_resnet.py"))
        if path is None:
            self.skipTest("original FedCorr sources not present")
        original = _load(path, "orig_fedcorr_resnet")

        reference = original.ResNet18(num_classes=10)
        mine = build_model("resnet18", 10)

        ref_keys = set(reference.state_dict())
        my_keys = set(mine.state_dict())
        self.assertEqual(ref_keys, my_keys, "state_dict keys differ")

        mine.load_state_dict(reference.state_dict())
        mine.eval()
        reference.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected = reference(x)
            actual = mine(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_resnet34_matches_official_architecture(self):
        from fusion.models import build_model

        path = _original(os.path.join("fedcorr", "model", "model_resnet.py"))
        if path is None:
            self.skipTest("original FedCorr sources not present")
        original = _load(path, "orig_fedcorr_resnet2")

        reference = original.ResNet34(num_classes=100)
        mine = build_model("resnet34", 100)
        mine.load_state_dict(reference.state_dict())
        mine.eval()
        reference.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(mine(x), reference(x), rtol=1e-5, atol=1e-6)


class TestFedAvgEquivalence(unittest.TestCase):
    def test_weighted_average_matches_official_fedavg(self):
        from fusion.stages.fedds_train import weighted_average

        path = _original(os.path.join("fedcorr", "util", "fedavg.py"))
        if path is None:
            self.skipTest("original FedCorr sources not present")
        original = _load(path, "orig_fedcorr_fedavg")

        torch.manual_seed(0)
        states = [{"w": torch.randn(4), "b": torch.randn(2)} for _ in range(5)]
        sizes = [10, 20, 30, 40, 50]
        expected = original.FedAvg(states, sizes)
        actual = weighted_average(states, np.asarray(sizes, dtype=np.float64))
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-8)


class TestDawidSkeneEquivalence(unittest.TestCase):
    def test_matches_official_em(self):
        from fusion.core.dawid_skene import dawid_skene

        path = _original(os.path.join("fedds_repro", "dawid skene.py"))
        if path is None:
            self.skipTest("original FedDS sources not present")
        original = _load(path, "orig_fedds_ds")

        rng = np.random.default_rng(7)
        num_classes, num_items = 10, 800
        truth = rng.integers(0, num_classes, num_items)
        columns = []
        for accuracy in (0.9, 0.7, 0.5):
            keep = rng.random(num_items) < accuracy
            wrong = (truth + rng.integers(1, num_classes, num_items)) % num_classes
            columns.append(np.where(keep, truth, wrong))
        predictions = np.stack(columns, axis=1)

        _, expected_confusion = original.dawid_skene_standard(
            predictions.tolist(), num_classes=num_classes
        )
        result = dawid_skene(predictions, num_classes=num_classes)

        np.testing.assert_allclose(
            result.confusion, np.asarray(expected_confusion), rtol=1e-6, atol=1e-8
        )

    def test_single_annotator_degenerates(self):
        from fusion.core.dawid_skene import dawid_skene

        rng = np.random.default_rng(11)
        predictions = rng.integers(0, 10, (300, 1))
        result = dawid_skene(predictions, num_classes=10)
        np.testing.assert_allclose(result.reliability, [1.0])


@unittest.skipUnless(HAS_SCIPY, "torch model comparison needs the same runtime")
class TestCNNCifarEquivalence(unittest.TestCase):
    def test_matches_official_fedds_cnn(self):
        """The FedDS stage model must be the same network as the official one."""
        from types import SimpleNamespace

        from fusion.models import build_model

        path = _original(os.path.join("fedds_repro", "model.py"))
        if path is None:
            self.skipTest("original FedDS sources not present")
        original = _load(path, "orig_fedds_model")

        args = SimpleNamespace(num_channels=3, num_classes=10)
        reference = original.CNNCifar10(args)
        mine = build_model("cnncifar", 10)

        self.assertEqual(set(reference.state_dict()), set(mine.state_dict()))
        mine.load_state_dict(reference.state_dict())
        mine.eval()
        reference.eval()

        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(mine(x), reference(x), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
