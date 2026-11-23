"""Synthetic software checks; no study data or subject identifiers."""
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat

from fnirs_subtyping.features import activation_beta, background_fc, extract_subject, fc, hrf, malff
from fnirs_subtyping.clustering import FAMILIES, candidate, fused_from_features, scale_view
from fnirs_subtyping.statistics import (behavior_tests, classify_profiles, profile_tests,
                                        signed_auc, subtype_permutation)


class SmokeTests(unittest.TestCase):
    def test_feature_shapes(self):
        rng = np.random.default_rng(14)
        membership = np.repeat(np.arange(1, 23), 2)
        channels = rng.normal(size=(300, 44))
        valid = np.ones(44, dtype=bool)
        events = pd.DataFrame({"condition": ["Naming"] * 5,
                               "start": [20, 70, 120, 170, 220],
                               "end": [35, 85, 135, 185, 235]})
        self.assertEqual(activation_beta(channels, valid, membership, 1.0, events, ("Naming",)).shape, (1, 22))
        self.assertEqual(background_fc(channels, valid, membership, 1.0, events, ("Naming",)).shape, (231,))
        self.assertEqual(fc(channels[:, ::2]).shape, (231,))
        self.assertEqual(malff(channels[:, ::2], 1.0).shape, (22,))
        self.assertTrue(np.isfinite(hrf(1.0)).all())

    def test_end_to_end_feature_extraction(self):
        rng = np.random.default_rng(16)
        membership = np.repeat(np.arange(1, 23), 2)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for run in ("rest", "naming", "fluency"):
                signal = rng.normal(size=(400, 44))
                savemat(folder / f"{run}.mat", {"nirsdata": {
                    "oxyData": signal, "dxyData": signal, "totalData": signal,
                    "T": 1.0, "exception_channel": np.zeros(44)}})
            rows = []
            for i in range(5):
                rows.append(("naming", "Naming", 20 + i * 65, 40 + i * 65))
                rows.append(("fluency", "Phonemic", 20 + i * 35, 35 + i * 35))
                rows.append(("fluency", "Semantic", 205 + i * 35, 220 + i * 35))
            events = pd.DataFrame(rows, columns=["run", "condition", "start", "end"])
            result = extract_subject(folder, events, membership)
            self.assertEqual({k: len(v) for k, v in result.items()},
                             {"intrinsic": 253, "activation": 66, "reconfiguration": 462,
                              "profiling": 106, "edge_profiles": 693})

    def test_clustering_shapes(self):
        rng = np.random.default_rng(52)
        features = {name: rng.normal(size=(79, sum(sizes))) for name, sizes in FAMILIES.items()}
        self.assertEqual(scale_view(features["intrinsic"], FAMILIES["intrinsic"]).shape, (79, 253))
        fused = fused_from_features(features)
        labels, gap = candidate(fused, 3)
        self.assertEqual(fused.shape, (79, 79))
        self.assertEqual(len(labels), 79)
        self.assertTrue(np.isfinite(gap))

    def test_statistics(self):
        self.assertAlmostEqual(signed_auc(np.array([3., 4.]), np.array([1., 2.])), 1.0)
        rng = np.random.default_rng(2)
        labels = np.repeat([1, 2, 3], [32, 17, 30])
        subtypes = rng.choice(["a", "b", "c"], size=79)
        result = subtype_permutation(labels, subtypes, repetitions=10)
        self.assertTrue(0 <= result["p_monte_carlo"] <= 1)
        patient = rng.normal(size=(79, 106))
        control = rng.normal(size=(17, 106))
        names = [f"f{i + 1:03d}" for i in range(106)]
        frames = profile_tests(patient, control, labels, names, bootstrap_reps=2)
        self.assertEqual(len(frames["cluster_vs_control"]), 318)
        self.assertEqual(len(frames["patient_pairwise"]), 318)
        classes = classify_profiles(frames, patient, control, labels, names, repetitions=2)
        self.assertEqual(len(classes), 318)
        scores = pd.DataFrame(rng.normal(size=(96, 3)), columns=["naming", "phonemic", "semantic"])
        comparison = behavior_tests(scores, np.r_[labels, np.zeros(17, int)])
        self.assertEqual(len(comparison["behavior_pairwise"]), 18)


if __name__ == "__main__":
    unittest.main()
