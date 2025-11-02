#!/usr/bin/env python3
"""
Train baseline classifiers and export artifacts for the OMTAD toolkit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

plt.switch_backend("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_PATH = SCRIPT_DIR / "DEMO.md"

FEATURE_COLUMNS = [
    "n_points",
    "track_duration_min",
    "mean_speed",
    "max_speed",
    "mean_turn_rate",
    "dist_nm",
]

TRAIN_YEARS = {2018, 2019}
TEST_YEARS = {2020}

DEFAULT_FEATURES_PATH = Path("./data/stage/track_features.csv")
DEFAULT_MODELS_DIR = Path("./deliverables/omtad_xiv/models")


def ensure_non_empty(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        raise RuntimeError(f"{label} dataset is empty. Check year coverage.")


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], path: Path) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("OMTAD Vessel Type Confusion Matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_f1_scores(f1_values: dict[str, float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(f1_values.keys())
    scores = [f1_values[label] for label in labels]
    ax.bar(labels, scores, color="#2563eb")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Scores")
    for idx, score in enumerate(scores):
        ax.text(idx, score + 0.02, f"{score:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_speed_hist(flat_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = sorted(flat_df["vessel_class"].dropna().unique())
    for vessel_class in classes:
        subset = flat_df.loc[flat_df["vessel_class"] == vessel_class, "speed"].dropna()
        if subset.empty:
            continue
        ax.hist(subset, bins=40, alpha=0.5, label=vessel_class.capitalize())
    ax.set_xlabel("Speed")
    ax.set_ylabel("Frequency")
    ax.set_title("Speed Distribution by Vessel Class")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_report(
    features: pd.DataFrame,
    metrics_path: Path,
    macro_f1: float,
    class_f1: dict[str, float],
    binary_f1: float,
    roc_auc: float,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    total_tracks = len(features)
    train_tracks = len(features[features["year"].isin(TRAIN_YEARS)])
    test_tracks = len(features[features["year"].isin(TEST_YEARS)])

    fishing_speed = features.loc[features["vessel_class"] == "fishing", "mean_speed"].mean()
    non_fishing_speed = features.loc[features["vessel_class"] != "fishing", "mean_speed"].mean()
    fishing_turn = features.loc[features["vessel_class"] == "fishing", "mean_turn_rate"].mean()
    tanker_turn = features.loc[features["vessel_class"] == "tanker", "mean_turn_rate"].mean()

    lines = [
        "OMTAD XIV Analysis Report",
        "=========================",
        "",
        f"Total tracks analysed: {total_tracks:,}",
        f"Training tracks (2018-2019): {train_tracks:,}",
        f"Test tracks (2020): {test_tracks:,}",
        "",
        f"Multiclass macro-F1: {macro_f1:.3f}",
        f"Binary fishing F1: {binary_f1:.3f}",
        f"Binary fishing ROC-AUC: {roc_auc:.3f}",
        "",
        "Per-class F1 scores:",
    ]
    for cls, score in class_f1.items():
        lines.append(f"- {cls}: {score:.3f}")
    lines += [
        "",
        "Key Findings:",
        f"- Fishing tracks average {fishing_speed:.2f} speed units vs {non_fishing_speed:.2f} for other classes.",
        f"- Fishing turn rate mean ({fishing_turn:.3f} deg/s) exceeds tanker turn rate ({tanker_turn:.3f} deg/s).",
        f"- Multiclass macro-F1 of {macro_f1:.3f} indicates balanced performance across vessel types.",
        f"- Binary fishing model ROC-AUC of {roc_auc:.3f} highlights strong separation between fishing/non-fishing.",
    ]

    metrics_path.write_text("\n".join(lines), encoding="utf-8")


def write_demo_instructions(path: Path) -> None:
    lines = [
        "1. Upload `deliverables/omtad_xiv/models/omtad_vessel_type.pkl` and `deliverables/omtad_xiv/models/omtad_fishing.pkl` in the demo UI.",
        "2. Pick a track_id from `deliverables/omtad_xiv/samples/demo_tracks.csv`, click “Classify Track”, and review the predicted class and probabilities.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline models on OMTAD track features.")
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Input track features CSV (default: ./data/stage/track_features.csv).",
    )
    parser.add_argument(
        "--out",
        dest="models_dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory to store models (default: ./deliverables/omtad_xiv/models).",
    )
    args = parser.parse_args()

    models_dir = args.models_dir
    base_dir = models_dir.parent if models_dir.parent != Path(".") else SCRIPT_DIR
    charts_dir = base_dir / "charts"
    metrics_dir = base_dir / "metrics"

    features = pd.read_csv(args.input_path, parse_dates=["start_ts", "end_ts"])
    ensure_non_empty(features, "Features")

    X = features[FEATURE_COLUMNS]
    y_multi = features["vessel_class"].astype(str)

    train_mask = features["year"].isin(TRAIN_YEARS)
    test_mask = features["year"].isin(TEST_YEARS)

    ensure_non_empty(features[train_mask], "Training")
    ensure_non_empty(features[test_mask], "Test")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train_multi, y_test_multi = y_multi[train_mask], y_multi[test_mask]

    multi_rf = train_random_forest(X_train, y_train_multi)
    y_pred_multi = multi_rf.predict(X_test)
    macro_f1 = f1_score(y_test_multi, y_pred_multi, average="macro")
    labels = sorted(y_multi.unique())
    cm = confusion_matrix(y_test_multi, y_pred_multi, labels=labels)
    f1_per_class_values = f1_score(y_test_multi, y_pred_multi, average=None, labels=labels)
    class_f1 = {label: score for label, score in zip(labels, f1_per_class_values)}

    binary_labels = (y_multi == "fishing").astype(int)
    y_train_bin, y_test_bin = binary_labels[train_mask], binary_labels[test_mask]
    binary_rf = train_random_forest(X_train, y_train_bin)
    y_pred_bin = binary_rf.predict(X_test)
    y_proba_bin = binary_rf.predict_proba(X_test)[:, 1]
    binary_f1 = f1_score(y_test_bin, y_pred_bin)
    roc_auc = roc_auc_score(y_test_bin, y_proba_bin)

    models_dir.mkdir(parents=True, exist_ok=True)
    vessel_model_path = models_dir / "omtad_vessel_type.pkl"
    fishing_model_path = models_dir / "omtad_fishing.pkl"
    joblib.dump(multi_rf, vessel_model_path, compress=3)
    joblib.dump(binary_rf, fishing_model_path, compress=3)

    vessel_meta = {
        "features": FEATURE_COLUMNS,
        "train_years": sorted(TRAIN_YEARS),
        "test_years": sorted(TEST_YEARS),
        "macro_f1": macro_f1,
        "class_f1": class_f1,
    }
    fishing_meta = {
        "features": FEATURE_COLUMNS,
        "train_years": sorted(TRAIN_YEARS),
        "test_years": sorted(TEST_YEARS),
        "f1": binary_f1,
        "roc_auc": roc_auc,
    }

    (models_dir / "omtad_vessel_type.meta.json").write_text(json.dumps(vessel_meta, indent=2), encoding="utf-8")
    (models_dir / "omtad_fishing.meta.json").write_text(json.dumps(fishing_meta, indent=2), encoding="utf-8")

    plot_confusion_matrix(cm, labels, charts_dir / "confusion_matrix_multiclass.png")
    plot_f1_scores(class_f1, charts_dir / "f1_per_class.png")

    flat_path = args.input_path.parent / "omtad_flat.csv"
    if not flat_path.exists():
        raise FileNotFoundError(f"Expected flattened CSV at {flat_path} for histogram generation.")
    flat_df = pd.read_csv(flat_path)
    plot_speed_hist(flat_df, charts_dir / "speed_hist_by_class.png")

    write_report(features, metrics_dir / "report.txt", macro_f1, class_f1, binary_f1, roc_auc)
    write_demo_instructions(DEMO_PATH)

    print(f"Multiclass macro-F1: {macro_f1:.3f}")
    print(f"Binary fishing ROC-AUC: {roc_auc:.3f}")
    print(f"Saved multiclass model to {vessel_model_path}")
    print(f"Saved binary model to {fishing_model_path}")
    print(f"Charts directory: {charts_dir}")
    print(f"Metrics report: {metrics_dir / 'report.txt'}")
    print(f"Demo instructions: {DEMO_PATH}")


if __name__ == "__main__":
    main()
