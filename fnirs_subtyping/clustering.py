"""Family-balanced SNF, deterministic spectral clustering, and null testing."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

FAMILIES = {"intrinsic": (22, 231), "activation": (22, 22, 22),
            "reconfiguration": (231, 231)}


def scale_view(x: np.ndarray, family_sizes: tuple[int, ...]) -> np.ndarray:
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != sum(family_sizes) or not np.isfinite(x).all():
        raise ValueError("Invalid feature matrix")
    sd = x.std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError("Constant feature in patient cohort")
    z = (x - x.mean(axis=0)) / sd
    start = 0
    for size in family_sizes:
        z[:, start:start + size] /= np.sqrt(size)
        start += size
    return z


def affinity(x: np.ndarray, neighbors: int = 20, sigma: float = 0.5) -> np.ndarray:
    if len(x) <= neighbors:
        raise ValueError("Need more participants than neighbors")
    diff = squareform(pdist(x, metric="euclidean"))
    means = np.sort(diff, axis=0).T[:, 1:neighbors + 1].mean(axis=1) + np.finfo(float).eps
    local = (means[:, None] + means[None, :]) / 3 + diff / 3 + np.finfo(float).eps
    scale = sigma * np.maximum(local, np.finfo(float).eps)
    density = np.exp(-0.5 * (diff / scale) ** 2) / (np.sqrt(2 * np.pi) * scale)
    return (density + density.T) / 2


def _normalize(x: np.ndarray) -> np.ndarray:
    denominator = x.sum(axis=1) - np.diag(x)
    if np.any(denominator <= 0):
        raise ValueError("Disconnected affinity")
    out = x / (2 * denominator[:, None])
    np.fill_diagonal(out, 0.5)
    return (out + out.T) / 2


def _dominant(x: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(x)
    for i, row in enumerate(x):
        keep = np.argsort(row, kind="stable")[-k:]
        out[i, keep] = row[keep]
    return out / out.sum(axis=1)[:, None]


def snf(views: list[np.ndarray], neighbors: int = 20, iterations: int = 20) -> np.ndarray:
    if len(views) < 2:
        raise ValueError("SNF requires at least two views")
    p = [_normalize(w) for w in views]
    s = [_dominant(w, neighbors) for w in p]
    for _ in range(iterations):
        old = p
        p = [_normalize(s[v] @ np.mean([old[j] for j in range(len(old)) if j != v], axis=0) @ s[v])
             for v in range(len(old))]
    return _normalize(np.mean(p, axis=0))


def fused_from_features(features: dict[str, np.ndarray], neighbors: int = 20,
                        iterations: int = 20) -> np.ndarray:
    return snf([affinity(scale_view(features[name], sizes), neighbors)
                for name, sizes in FAMILIES.items()], neighbors, iterations)


def eigensystem(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.allclose(w, w.T, atol=1e-12):
        raise ValueError("Asymmetric network")
    degree = w.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("Isolated participant")
    inv = degree ** -0.5
    laplacian = np.eye(len(w)) - inv[:, None] * w * inv[None, :]
    return np.linalg.eigh(laplacian)


def discretize(x: np.ndarray) -> np.ndarray:
    x = x / np.linalg.norm(x, axis=1)[:, None]
    n, k = x.shape
    rotation = np.zeros((k, k))
    rotation[:, 0] = x[int(np.rint(n / 2)) - 1]
    cumulative = np.zeros(n)
    for j in range(1, k):
        cumulative += np.abs(x @ rotation[:, j - 1])
        rotation[:, j] = x[np.argmin(cumulative)]
    last = 0.0
    for _ in range(20):
        labels = np.argmax(x @ rotation, axis=1)
        discrete = np.eye(k)[labels]
        u, singular, vh = np.linalg.svd(discrete.T @ x)
        objective = 2 * (n - singular.sum())
        if abs(objective - last) < np.finfo(float).eps:
            break
        last = objective
        rotation = vh.T @ u.T
    labels = np.argmax(x @ rotation, axis=1) + 1
    if len(np.unique(labels)) != k:
        raise ValueError("Empty spectral cluster")
    return labels


def candidate(w: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    values, vectors = eigensystem(w)
    order = np.argsort(np.abs(values), kind="stable")
    return discretize(vectors[:, order[:k]]), float(values[k] - values[k - 1])


def permutation_eigengaps(features: dict[str, np.ndarray], repetitions: int = 5000,
                          seed: int = 20261111) -> tuple[np.ndarray, np.ndarray]:
    observed_w = fused_from_features(features)
    observed = np.array([candidate(observed_w, k)[1] for k in range(2, 7)])
    rng = np.random.default_rng(seed)
    exceed = np.zeros(5, int)
    for _ in range(repetitions):
        shuffled = {name: matrix.copy() for name, matrix in features.items()}
        # Preserve the joint within-participant mALFF block under the null.
        shuffled["intrinsic"][:, :22] = features["intrinsic"][rng.permutation(len(observed_w)), :22]
        for name, matrix in shuffled.items():
            start = 22 if name == "intrinsic" else 0
            for j in range(start, matrix.shape[1]):
                matrix[:, j] = matrix[rng.permutation(len(matrix)), j]
        w = fused_from_features(shuffled)
        exceed += np.array([candidate(w, k)[1] >= observed[i] for i, k in enumerate(range(2, 7))])
    p = (exceed + 1) / (repetitions + 1)
    from statsmodels.stats.multitest import multipletests
    return observed, multipletests(p, method="holm")[1]


def subsampling_ari(features: dict[str, np.ndarray], labels: np.ndarray,
                    repetitions: int = 1000, seed: int = 20261111) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(labels)
    size = round(n * 0.8)
    scores = np.empty(repetitions)
    for i in range(repetitions):
        indices = np.sort(rng.choice(n, size=size, replace=False))
        w = fused_from_features({name: x[indices] for name, x in features.items()})
        sampled, _ = candidate(w, 3)
        scores[i] = adjusted_rand_score(labels[indices], sampled)
    return scores


def robustness(features: dict[str, np.ndarray], labels: np.ndarray) -> list[dict]:
    """Parameter grid, view ablation, mean fusion and leave-one-out refits."""
    from sklearn.metrics import adjusted_rand_score
    rows = []
    def agreement(reference: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
        overlap = np.zeros((3, 3))
        for a in range(1, 4):
            for b in range(1, 4):
                intersection = np.sum((reference == a) & (fitted == b))
                union = np.sum((reference == a) | (fitted == b))
                overlap[a - 1, b - 1] = intersection / union if union else 0
        row, col = linear_sum_assignment(-overlap)
        matched = {int(a + 1): float(overlap[a, b]) for a, b in zip(row, col)}
        return {"ari": adjusted_rand_score(reference, fitted),
                **{f"jaccard_c{k}": matched[k] for k in range(1, 4)}}
    for neighbors in (10, 15, 20, 25, 30):
        for iterations in (10, 20, 30, 50):
            if (neighbors, iterations) == (20, 20):
                continue
            fitted, _ = candidate(fused_from_features(features, neighbors, iterations), 3)
            rows.append({"analysis": "parameter_grid", "detail": f"K={neighbors};T={iterations}",
                         **agreement(labels, fitted)})
    for omitted in FAMILIES:
        remaining = [affinity(scale_view(features[name], sizes))
                     for name, sizes in FAMILIES.items() if name != omitted]
        fitted, _ = candidate(snf(remaining), 3)
        rows.append({"analysis": "leave_one_view_out", "detail": omitted,
                     **agreement(labels, fitted)})
    views = [affinity(scale_view(features[name], sizes)) for name, sizes in FAMILIES.items()]
    fitted, _ = candidate(np.mean(views, axis=0), 3)
    rows.append({"analysis": "arithmetic_mean_fusion", "detail": "equal_view_weights",
                 **agreement(labels, fitted)})
    for removed in range(len(labels)):
        keep = np.arange(len(labels)) != removed
        fitted, _ = candidate(fused_from_features({name: x[keep] for name, x in features.items()}), 3)
        rows.append({"analysis": "leave_one_patient_out", "detail": str(removed + 1),
                     **agreement(labels[keep], fitted)})
    return rows
