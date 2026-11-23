"""Fixed-label neurophysiological and clinical comparisons."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, rankdata
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm


def signed_auc(a: np.ndarray, b: np.ndarray) -> float:
    return 2 * mannwhitneyu(a, b, alternative="two-sided").statistic / (len(a) * len(b)) - 1


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    pooled = np.sqrt(((n - 1) * a.var(ddof=1) + (m - 1) * b.var(ddof=1)) / (n + m - 2))
    return float((a.mean() - b.mean()) / pooled * (1 - 3 / (4 * (n + m) - 9))) if pooled else np.nan


def bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
              repetitions: int = 2000) -> tuple[float, float]:
    draws = np.empty(repetitions)
    for i in range(repetitions):
        draws[i] = a[rng.integers(len(a), size=len(a))].mean() - b[rng.integers(len(b), size=len(b))].mean()
    return tuple(np.quantile(draws, (0.025, 0.975)))


def profile_tests(patient: np.ndarray, controls: np.ndarray, labels: np.ndarray,
                  names: list[str], seed: int = 20261111,
                  bootstrap_reps: int = 2000) -> dict[str, pd.DataFrame]:
    if patient.shape != (len(labels), len(names)) or controls.shape[1] != len(names):
        raise ValueError("Profile feature dimensions differ")
    if not np.isfinite(patient).all() or not np.isfinite(controls).all():
        raise ValueError("Missing profile measurements")
    groups = np.unique(labels)
    if len(groups) != 3:
        raise ValueError("Expected three patient clusters")
    rng = np.random.default_rng(seed)
    hc_rows, omnibus_rows, pair_rows = [], [], []
    for j, name in enumerate(names):
        samples = {group: patient[labels == group, j] for group in groups}
        control = controls[:, j]
        omnibus_rows.append({"feature": name, "p": kruskal(*samples.values()).pvalue})
        for group, sample in samples.items():
            ci = bootstrap(sample, control, rng, bootstrap_reps) if bootstrap_reps else (np.nan, np.nan)
            hc_rows.append({"feature": name, "cluster": int(group),
                            "p": mannwhitneyu(sample, control).pvalue,
                            "signed_auc": signed_auc(sample, control),
                            "mean_difference": sample.mean() - control.mean(),
                            "hedges_g": hedges_g(sample, control),
                            "ci95_low": ci[0], "ci95_high": ci[1]})
        for a, b in itertools.combinations(groups, 2):
            pair_rows.append({"feature": name, "cluster_a": int(a), "cluster_b": int(b),
                              "p": mannwhitneyu(samples[a], samples[b]).pvalue,
                              "signed_auc": signed_auc(samples[a], samples[b])})
    frames = {"cluster_vs_control": pd.DataFrame(hc_rows),
              "patient_omnibus": pd.DataFrame(omnibus_rows),
              "patient_pairwise": pd.DataFrame(pair_rows)}
    for frame in frames.values():
        frame["q_bh"] = multipletests(frame.p, method="fdr_bh")[1]
    return frames


def classify_profiles(frames: dict[str, pd.DataFrame], patient: np.ndarray,
                      controls: np.ndarray, labels: np.ndarray,
                      names: list[str], repetitions: int = 2000,
                      seed: int = 20261111) -> pd.DataFrame:
    """Apply the manuscript's confirmatory and directional selection rules."""
    hc = frames["cluster_vs_control"].set_index(["feature", "cluster"])
    omni = frames["patient_omnibus"].set_index("feature")
    pair = frames["patient_pairwise"]
    groups = sorted(np.unique(labels))
    rng = np.random.default_rng(seed)
    rows = []
    for j, name in enumerate(names):
        samples = {g: patient[labels == g, j] for g in groups}
        control = controls[:, j]
        for target in groups:
            others = [g for g in groups if g != target]
            pair_rows = [pair.loc[(pair.feature == name) &
                                  (pair.cluster_a.eq(min(target, other))) &
                                  (pair.cluster_b.eq(max(target, other)))].iloc[0]
                         for other in others]
            directional = []
            for other, row in zip(others, pair_rows):
                effect = row.signed_auc if target < other else -row.signed_auc
                directional.append(effect)
            same_direction = np.sign(directional[0]) == np.sign(directional[1]) and directional[0] != 0
            significant_pairs = all(row.q_bh < 0.05 for row in pair_rows)
            characteristic = (hc.loc[(name, target), "q_bh"] < 0.05 and
                              omni.loc[name, "q_bh"] < 0.05 and significant_pairs)
            recurrence = 0.0
            if same_direction and repetitions:
                direction = np.sign(directional[0])
                matches = 0
                for _ in range(repetitions):
                    means = {g: samples[g][rng.integers(len(samples[g]), size=len(samples[g]))].mean()
                             for g in groups}
                    hc_mean = control[rng.integers(len(control), size=len(control))].mean()
                    matches += all(direction * (means[target] - means[other]) > 0 for other in others) and \
                               direction * (means[target] - hc_mean) > 0
                recurrence = matches / repetitions
            differentiating = (not characteristic and omni.loc[name, "q_bh"] < 0.05 and
                               significant_pairs and same_direction and
                               min(abs(v) for v in directional) >= 0.35 and recurrence >= 0.80)
            rows.append({"feature": name, "cluster": int(target),
                         "classification": "cluster_characteristic" if characteristic else
                         "cluster_differentiating" if differentiating else "other",
                         "bootstrap_extremeness_recurrence": recurrence})
    return pd.DataFrame(rows)


def clinical_tests(table: pd.DataFrame, labels: np.ndarray,
                   columns: list[str]) -> dict[str, pd.DataFrame]:
    if len(table) != len(labels):
        raise ValueError("Clinical rows and labels differ")
    groups = np.unique(labels)
    overall, pairwise = [], []
    for column in columns:
        values = pd.to_numeric(table[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing clinical values: {column}")
        samples = {group: values[labels == group] for group in groups}
        overall.append({"variable": column, "p": kruskal(*samples.values()).pvalue})
        for a, b in itertools.combinations(groups, 2):
            pairwise.append({"variable": column, "cluster_a": int(a), "cluster_b": int(b),
                             "p": mannwhitneyu(samples[a], samples[b]).pvalue})
    frames = {"clinical_omnibus": pd.DataFrame(overall), "clinical_pairwise": pd.DataFrame(pairwise)}
    for frame in frames.values():
        frame["p_holm"] = multipletests(frame.p, method="holm")[1]
    return frames


def behavior_tests(table: pd.DataFrame, labels: np.ndarray,
                   columns: tuple[str, ...] = ("naming", "phonemic", "semantic")) -> dict[str, pd.DataFrame]:
    """Four-group task score tests; group 0 denotes healthy controls."""
    if len(table) != len(labels) or set(np.unique(labels)) != {0, 1, 2, 3}:
        raise ValueError("Expected controls and three patient clusters")
    omnibus, pairwise = [], []
    for column in columns:
        values = table[column].to_numpy(float)
        samples = {group: values[labels == group] for group in range(4)}
        omnibus.append({"task": column, "p": kruskal(*samples.values()).pvalue})
        for a, b in itertools.combinations(range(4), 2):
            pairwise.append({"task": column, "group_a": a, "group_b": b,
                             "p": mannwhitneyu(samples[a], samples[b]).pvalue})
    frames = {"behavior_omnibus": pd.DataFrame(omnibus),
              "behavior_pairwise": pd.DataFrame(pairwise)}
    for frame in frames.values():
        frame["p_holm"] = multipletests(frame.p, method="holm")[1]
    return frames


def subtype_permutation(labels: np.ndarray, subtypes: np.ndarray,
                        repetitions: int = 100000, seed: int = 20261111) -> dict[str, float]:
    observed = pd.crosstab(labels, subtypes).to_numpy()
    statistic = chi2_contingency(observed, correction=False).statistic
    n = len(labels)
    effect = np.sqrt(statistic / (n * min(observed.shape[0] - 1, observed.shape[1] - 1)))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(repetitions):
        perm = pd.crosstab(labels, rng.permutation(subtypes)).to_numpy()
        exceed += chi2_contingency(perm, correction=False).statistic >= statistic
    return {"chi2": statistic, "cramers_v": effect, "p_monte_carlo": (exceed + 1) / (repetitions + 1)}


def adjusted_patient_model(values: np.ndarray, labels: np.ndarray, age: np.ndarray,
                           sex: np.ndarray | None = None) -> pd.DataFrame:
    """Patient-only inverse-normal rank model with HC3 covariance."""
    values, age = np.asarray(values, float), np.asarray(age, float)
    if not np.isfinite(values).all() or not np.isfinite(age).all():
        raise ValueError("Missing covariate data")
    from scipy.stats import norm
    y = norm.ppf((rankdata(values) - 0.5) / len(values))
    design = pd.get_dummies(pd.Series(labels, name="cluster"), drop_first=True, dtype=float)
    design["age"] = age
    if sex is not None:
        design["sex"] = np.asarray(sex, float)
    fit = sm.OLS(y, sm.add_constant(design)).fit(cov_type="HC3")
    return pd.DataFrame({"term": fit.params.index, "coefficient": fit.params.values,
                         "se_hc3": fit.bse.values, "p": fit.pvalues.values})
