"""Command line entry points; all participant files stay outside this repository."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import extract_subject
from .clustering import candidate, fused_from_features, permutation_eigengaps, subsampling_metrics, robustness
from .statistics import (adjusted_patient_model, behavior_tests, classify_profiles, clr_malff,
                         clinical_tests, profile_tests, subtype_permutation)


def _external(path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    path = path.resolve()
    if path == root or root in path.parents:
        raise ValueError("Data and output paths must be outside the public repository")
    return path


def _write(frame: pd.DataFrame, folder: Path, name: str) -> None:
    frame.to_csv(folder / f"{name}.csv", index=False)


def extract(args: argparse.Namespace) -> None:
    data = _external(args.data)
    output = _external(args.output)
    cohort = pd.read_csv(_external(args.cohort), dtype=str)
    events = pd.read_csv(_external(args.events))
    roi = pd.read_csv(_external(args.roi_map))
    if not {"subject", "group"} <= set(cohort) or not {"channel", "roi"} <= set(roi):
        raise ValueError("Invalid cohort or ROI map schema")
    if cohort.subject.duplicated().any() or cohort.group.value_counts().to_dict() != {"PAT": 79, "HC": 17}:
        raise ValueError("Expected 79 unique patients and 17 unique controls")
    roi = roi.sort_values("channel")
    membership = roi.roi.to_numpy(int)
    if not np.array_equal(roi.channel.to_numpy(int), np.arange(1, len(roi) + 1)):
        raise ValueError("ROI map channels must be numbered consecutively from 1")
    output.mkdir(parents=True, exist_ok=True)
    rows = {"intrinsic": [], "activation": [], "reconfiguration": [], "profiling": [], "edge_profiles": []}
    for record in cohort.itertuples(index=False):
        selected = events.loc[events.subject.astype(str) == record.subject]
        if selected.empty:
            raise ValueError(f"Missing events for {record.subject}")
        values = extract_subject(data / record.subject, selected, membership)
        for name, vector in values.items():
            rows[name].append({"subject": record.subject, "group": record.group,
                               **{f"f{i + 1:03d}": float(v) for i, v in enumerate(vector)}})
    for name, records in rows.items():
        _write(pd.DataFrame(records), output, name)


def _load_features(folder: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    matrices = {}
    reference = None
    for name, expected in (("intrinsic", 253), ("activation", 66), ("reconfiguration", 462)):
        frame = pd.read_csv(folder / f"{name}.csv", dtype={"subject": str, "group": str})
        if reference is None:
            reference = frame[["subject", "group"]]
        elif not frame[["subject", "group"]].equals(reference):
            raise ValueError("Feature row order differs")
        values = frame.drop(columns=["subject", "group"]).to_numpy(float)
        if values.shape[1] != expected or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} features")
        matrices[name] = values
    assert reference is not None
    if reference.subject.duplicated().any() or reference.group.value_counts().to_dict() != {"PAT": 79, "HC": 17}:
        raise ValueError("Expected 79 patients and 17 controls")
    mask = reference.group.eq("PAT").to_numpy()
    return {name: x[mask] for name, x in matrices.items()}, reference


def cluster(args: argparse.Namespace) -> None:
    input_dir, output = _external(args.input), _external(args.output)
    matrices, cohort = _load_features(input_dir)
    output.mkdir(parents=True, exist_ok=True)
    network = fused_from_features(matrices)
    labels, _ = candidate(network, 3)
    subjects = cohort.loc[cohort.group.eq("PAT"), "subject"].tolist()
    _write(pd.DataFrame({"subject": subjects, "cluster": labels}), output, "cluster_labels")
    _write(pd.DataFrame(network, index=subjects, columns=subjects), output, "fused_network")
    if args.permutations:
        gaps, adjusted = permutation_eigengaps(matrices, args.permutations, args.seed)
        _write(pd.DataFrame({"k": np.arange(2, 7), "eigengap": gaps, "p_holm": adjusted}),
               output, "candidate_eigengaps")
    if args.subsamples:
        _write(pd.DataFrame(subsampling_metrics(matrices, labels, args.subsamples, args.seed)),
               output, "subsampling_stability")
    if args.robustness:
        _write(pd.DataFrame(robustness(matrices, labels)), output, "robustness")


def analyze(args: argparse.Namespace) -> None:
    input_dir, output = _external(args.input), _external(args.output)
    profile = pd.read_csv(input_dir / "profiling.csv", dtype={"subject": str, "group": str})
    labels = pd.read_csv(_external(args.labels), dtype={"subject": str})
    if labels.subject.duplicated().any() or profile.subject.duplicated().any():
        raise ValueError("Duplicate subject")
    patients = profile.loc[profile.group.eq("PAT")].merge(labels, on="subject", validate="one_to_one")
    controls = profile.loc[profile.group.eq("HC")]
    if len(patients) != 79 or len(controls) != 17:
        raise ValueError("Profile cohort mismatch")
    names = [c for c in profile if c.startswith("f")]
    if len(names) != 106:
        raise ValueError("Expected 106 profile features")
    output.mkdir(parents=True, exist_ok=True)
    frames = profile_tests(patients[names].to_numpy(float), controls[names].to_numpy(float),
                           patients.cluster.to_numpy(int), names, args.seed)
    for name, frame in frames.items():
        _write(frame, output, name)
    _write(classify_profiles(frames, patients[names].to_numpy(float),
                             controls[names].to_numpy(float), patients.cluster.to_numpy(int),
                             names, seed=args.seed), output, "profile_classification")
    if args.clr:
        clr_names = [f"clr_{name}" for name in names[:22]]
        clr_frames = profile_tests(clr_malff(patients[names[:22]].to_numpy(float)),
                                   clr_malff(controls[names[:22]].to_numpy(float)),
                                   patients.cluster.to_numpy(int), clr_names, args.seed,
                                   bootstrap_reps=0)
        for name, frame in clr_frames.items():
            _write(frame, output, "clr_malff_" + name)
    if args.edges:
        edge = pd.read_csv(input_dir / "edge_profiles.csv", dtype={"subject": str, "group": str})
        edge_pat = patients[["subject", "cluster"]].merge(edge.loc[edge.group.eq("PAT")],
                                                            on="subject", validate="one_to_one")
        edge_hc = edge.loc[edge.group.eq("HC")]
        edge_names = [c for c in edge if c.startswith("f")]
        if len(edge_pat) != 79 or len(edge_hc) != 17 or len(edge_names) != 693:
            raise ValueError("Edge profile cohort or dimensions differ")
        edge_frames = profile_tests(edge_pat[edge_names].to_numpy(float),
                                    edge_hc[edge_names].to_numpy(float),
                                    edge_pat.cluster.to_numpy(int), edge_names,
                                    args.seed, bootstrap_reps=0)
        for name, frame in edge_frames.items():
            _write(frame, output, "exploratory_edges_" + name)
    if args.clinical:
        clinical = pd.read_csv(_external(args.clinical), dtype={"subject": str})
        clinical = patients[["subject", "cluster"]].merge(clinical, on="subject", validate="one_to_one")
        for family, columns in (("demographic", args.demographic_columns.split(",")),
                                ("clinical", args.clinical_columns.split(","))):
            for name, frame in clinical_tests(clinical, clinical.cluster.to_numpy(int), columns).items():
                _write(frame, output, f"{family}_{name}")
        if "age" in clinical:
            for j, name in enumerate(names[88:106], start=88):
                age = clinical.age.to_numpy(float)
                sex = clinical.sex.to_numpy(float) if "sex" in clinical else None
                for model, covariate in (("age", None), ("age_sex", sex)):
                    if model == "age_sex" and sex is None:
                        continue
                    result = adjusted_patient_model(patients[name].to_numpy(float),
                                                    patients.cluster.to_numpy(int),
                                                    age, covariate)
                    result.insert(0, "feature", name)
                    result.insert(1, "model", model)
                    _write(result, output, f"covariate_{name}_{model}")
        if "subtype" in clinical:
            (output / "subtype_test.json").write_text(json.dumps(subtype_permutation(
                clinical.cluster.to_numpy(int), clinical.subtype.to_numpy(str),
                args.subtype_permutations, args.seed), indent=2), encoding="utf-8")
    if args.behavior:
        behavior = pd.read_csv(_external(args.behavior), dtype={"subject": str, "group": str})
        expected = pd.concat((patients[["subject", "cluster"]],
                              controls[["subject"]].assign(cluster=0)), ignore_index=True)
        behavior = expected.merge(behavior, on="subject", validate="one_to_one")
        if len(behavior) != 96:
            raise ValueError("Behavioral cohort mismatch")
        for name, frame in behavior_tests(behavior, behavior.cluster.to_numpy(int)).items():
            _write(frame, output, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    for flag in ("data", "cohort", "events", "roi-map", "output"):
        e.add_argument(f"--{flag}", type=Path, required=True)
    e.set_defaults(func=extract)
    c = sub.add_parser("cluster")
    c.add_argument("--input", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--permutations", type=int, default=5000)
    c.add_argument("--subsamples", type=int, default=1000)
    c.add_argument("--seed", type=int, default=20261111)
    c.add_argument("--robustness", action="store_true", help="Run parameter, view, fusion and leave-one-out refits")
    c.set_defaults(func=cluster)
    a = sub.add_parser("analyze")
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--labels", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--clinical", type=Path)
    a.add_argument("--behavior", type=Path, help="Four-group naming and fluency task scores")
    a.add_argument("--demographic-columns", default="age,post_onset_duration,education")
    a.add_argument("--clinical-columns", default="WAB,BDAE")
    a.add_argument("--subtype-permutations", type=int, default=100000)
    a.add_argument("--seed", type=int, default=20261111)
    a.add_argument("--edges", action="store_true", help="Run exploratory FC edge comparisons")
    a.add_argument("--clr", action="store_true", help="Run mALFF centered log-ratio sensitivity")
    a.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
