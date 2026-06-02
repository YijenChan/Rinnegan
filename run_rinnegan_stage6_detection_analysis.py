# -*- coding: utf-8 -*-
"""
Rinnegan Stage-6: Analyze R-GAE detection results.

This script analyzes Stage-5 outputs:
1. present-only threshold metrics;
2. strict clean-universe metrics;
3. TP/FP/FN entity lists;
4. corruption loss and remediation recovery;
5. recovered-edge reliability analysis;
6. report-ready comparison tables.

Default usage from the repository root:
    python scripts/run_stage6_detection_analysis.py \
        --stage5-root outputs/stage5_rgae_detection \
        --stage4-root outputs/stage4_rgae_data \
        --out-dir outputs/stage6_detection_analysis

Dependencies:
    pip install pandas numpy
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# =========================
# Config
# =========================

STAGE5_ROOT = Path("outputs/stage5_rgae_detection")
STAGE4_ROOT = Path("outputs/stage4_rgae_data")
OUT_DIR = Path("outputs/stage6_detection_analysis")

TEST_NAMES = ["clean", "corrupted", "remediated"]


# =========================
# Arguments
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rinnegan Stage-6 detection result analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage5-root", type=Path, default=STAGE5_ROOT)
    parser.add_argument("--stage4-root", type=Path, default=STAGE4_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--test-names",
        type=str,
        default=",".join(TEST_NAMES),
        help="Comma-separated test graph names whose *_canonical_scores.csv files exist in Stage-5 output.",
    )
    return parser.parse_args()


# =========================
# IO helpers
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def safe_float(x, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def text_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=str)
    return df[col].astype(str)


def parse_test_names(s: str) -> List[str]:
    out = [x.strip() for x in safe_str(s).split(",") if x.strip()]
    if not out:
        raise ValueError("--test-names cannot be empty.")
    return out


def is_recovered_source(x: str) -> bool:
    x = safe_str(x).lower()
    return x == "recovered" or x.endswith("_recovered") or x in {
        "llm_recovered",
        "mock_recovered",
        "cached_recovered",
        "openai_recovered",
    }


# =========================
# Load Stage-5 outputs
# =========================

def load_delta(stage5_root: Path) -> float:
    calibration_path = stage5_root / "calibration.json"
    if calibration_path.exists():
        cal = read_json(calibration_path)
        return float(cal.get("delta", 0.0))

    metrics_path = stage5_root / "detection_metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        return float(metrics.get("calibration", {}).get("delta", 0.0))

    raise FileNotFoundError(
        f"Cannot find calibration.json or detection_metrics.json under {stage5_root}"
    )


def load_canonical_scores(stage5_root: Path, name: str) -> pd.DataFrame:
    path = stage5_root / f"{name}_canonical_scores.csv"
    df = read_csv(path)

    required = ["canonical_node_id", "canonical_entity", "score", "is_malicious"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    df = df.copy()
    df["canonical_node_id"] = df["canonical_node_id"].astype(str)
    df["canonical_entity"] = df["canonical_entity"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["is_malicious"] = pd.to_numeric(df["is_malicious"], errors="coerce").fillna(0).astype(int)

    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    if "best_node_id" not in df.columns:
        df["best_node_id"] = ""
    if "best_node_source" not in df.columns:
        df["best_node_source"] = ""
    if "entity_type" not in df.columns:
        df["entity_type"] = ""
    if "host" not in df.columns:
        df["host"] = ""

    return df


def load_all_scores(stage5_root: Path, test_names: List[str]) -> Dict[str, pd.DataFrame]:
    return {name: load_canonical_scores(stage5_root, name) for name in test_names}


# =========================
# Metric functions
# =========================

def compute_metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "num_entities": int(len(y_true)),
        "num_malicious": int(y_true.sum()),
        "num_predicted": int(y_pred.sum()),
    }


def compute_present_only_metrics(df: pd.DataFrame, delta: float) -> Dict[str, Any]:
    y_true = df["is_malicious"].astype(int).values
    y_pred = (df["score"].values > delta).astype(int)
    return compute_metrics_from_arrays(y_true, y_pred)


def build_strict_universe_df(clean_df: pd.DataFrame, target_df: pd.DataFrame, delta: float) -> pd.DataFrame:
    """
    Strict clean-universe evaluation.

    Present-only evaluation ignores malicious entities that disappear from
    corrupted graphs. Strict evaluation uses clean canonical entities as the
    reference universe and assigns pred=0 to missing target entities.
    """
    clean_cols = [
        "canonical_node_id",
        "canonical_entity",
        "is_malicious",
        "entity_type",
        "host",
    ]
    clean_cols = [c for c in clean_cols if c in clean_df.columns]
    clean_base = clean_df[clean_cols].copy()
    clean_base = clean_base.rename(columns={
        "canonical_entity": "canonical_entity_clean",
        "is_malicious": "is_malicious_clean",
        "entity_type": "entity_type_clean",
        "host": "host_clean",
    })

    target_cols = [
        "canonical_node_id",
        "canonical_entity",
        "score",
        "confidence",
        "best_node_id",
        "best_node_source",
        "entity_type",
        "host",
        "is_malicious",
    ]
    target_cols = [c for c in target_cols if c in target_df.columns]
    target_base = target_df[target_cols].copy()
    target_base = target_base.rename(columns={
        "canonical_entity": "canonical_entity_target",
        "score": "score_target",
        "confidence": "confidence_target",
        "best_node_id": "best_node_id_target",
        "best_node_source": "best_node_source_target",
        "entity_type": "entity_type_target",
        "host": "host_target",
        "is_malicious": "is_malicious_target",
    })

    merged = clean_base.merge(target_base, on="canonical_node_id", how="outer")

    if "canonical_entity_clean" not in merged.columns:
        merged["canonical_entity_clean"] = ""
    if "canonical_entity_target" not in merged.columns:
        merged["canonical_entity_target"] = ""

    merged["canonical_entity_final"] = merged["canonical_entity_clean"].fillna(
        merged["canonical_entity_target"]
    )
    merged["is_present_in_target"] = merged["score_target"].notna().astype(int)

    merged["score_eval"] = pd.to_numeric(merged["score_target"], errors="coerce")
    merged["score_eval"] = merged["score_eval"].fillna(float("-inf"))

    if "is_malicious_clean" not in merged.columns:
        merged["is_malicious_clean"] = np.nan
    if "is_malicious_target" not in merged.columns:
        merged["is_malicious_target"] = 0

    merged["is_malicious_eval"] = merged["is_malicious_clean"].fillna(
        merged["is_malicious_target"]
    )
    merged["is_malicious_eval"] = pd.to_numeric(
        merged["is_malicious_eval"], errors="coerce"
    ).fillna(0).astype(int)

    merged["pred_anomalous_eval"] = (
        (merged["is_present_in_target"] == 1) & (merged["score_eval"] > delta)
    ).astype(int)

    if "entity_type_clean" not in merged.columns:
        merged["entity_type_clean"] = ""
    if "entity_type_target" not in merged.columns:
        merged["entity_type_target"] = ""
    if "host_clean" not in merged.columns:
        merged["host_clean"] = ""
    if "host_target" not in merged.columns:
        merged["host_target"] = ""

    merged["entity_type_final"] = merged["entity_type_clean"].fillna(
        merged["entity_type_target"]
    )
    merged["host_final"] = merged["host_clean"].fillna(merged["host_target"])

    return merged


def compute_strict_metrics(
    clean_df: pd.DataFrame,
    target_df: pd.DataFrame,
    delta: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    strict_df = build_strict_universe_df(clean_df, target_df, delta)
    y_true = strict_df["is_malicious_eval"].astype(int).values
    y_pred = strict_df["pred_anomalous_eval"].astype(int).values

    metrics = compute_metrics_from_arrays(y_true, y_pred)
    metrics["num_missing_from_target"] = int((strict_df["is_present_in_target"] == 0).sum())
    metrics["num_missing_malicious_from_target"] = int(
        ((strict_df["is_present_in_target"] == 0) & (strict_df["is_malicious_eval"] == 1)).sum()
    )
    return metrics, strict_df


def topk_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if len(df) == 0:
        return {"k": 0, "tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    y_true = df["is_malicious"].astype(int).values
    k = int(y_true.sum())

    if k <= 0:
        return {"k": 0, "tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tmp = df.sort_values("score", ascending=False).reset_index(drop=True)
    pred = np.zeros(len(tmp), dtype=int)
    pred[:k] = 1
    y = tmp["is_malicious"].astype(int).values

    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "k": k,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


# =========================
# Entity error analysis
# =========================

def annotate_errors(df: pd.DataFrame, delta: float) -> pd.DataFrame:
    out = df.copy()
    out["pred_anomalous_eval"] = (out["score"] > delta).astype(int)

    def label_row(r):
        y = safe_int(r.get("is_malicious", 0))
        p = safe_int(r.get("pred_anomalous_eval", 0))
        if y == 1 and p == 1:
            return "TP"
        if y == 0 and p == 1:
            return "FP"
        if y == 1 and p == 0:
            return "FN"
        return "TN"

    out["error_type"] = out.apply(label_row, axis=1)
    return out


def export_error_lists(scores: Dict[str, pd.DataFrame], delta: float, out_dir: Path):
    error_dir = out_dir / "entity_error_lists"
    ensure_dir(error_dir)

    for name, df in scores.items():
        ann = annotate_errors(df, delta)
        ann.to_csv(error_dir / f"{name}_all_entities_annotated.csv", index=False, encoding="utf-8-sig")

        for etype in ["TP", "FP", "FN"]:
            sub = ann[ann["error_type"] == etype].copy()
            sub = sub.sort_values("score", ascending=False)
            sub.to_csv(error_dir / f"{name}_{etype}.csv", index=False, encoding="utf-8-sig")


def build_entity_comparison(scores: Dict[str, pd.DataFrame], delta: float, test_names: List[str]) -> pd.DataFrame:
    if "clean" not in scores:
        raise ValueError("Entity comparison requires a clean score file.")

    clean_cols = [
        "canonical_node_id",
        "canonical_entity",
        "is_malicious",
        "entity_type",
        "host",
    ]
    clean_cols = [c for c in clean_cols if c in scores["clean"].columns]
    base = scores["clean"][clean_cols].copy()

    base = base.rename(columns={
        "is_malicious": "is_malicious_clean",
        "entity_type": "entity_type_clean",
        "host": "host_clean",
    })

    out = base

    for name in test_names:
        df = scores[name].copy()
        cols = [
            "canonical_node_id",
            "score",
            "confidence",
            "best_node_id",
            "best_node_source",
            "is_malicious",
        ]
        cols = [c for c in cols if c in df.columns]
        df = df[cols].copy()
        df[f"{name}_present"] = 1
        df[f"{name}_pred"] = (df["score"] > delta).astype(int)
        df = df.rename(columns={
            "score": f"{name}_score",
            "confidence": f"{name}_confidence",
            "best_node_id": f"{name}_best_node_id",
            "best_node_source": f"{name}_best_node_source",
            "is_malicious": f"{name}_is_malicious",
        })
        out = out.merge(df, on="canonical_node_id", how="outer")

    for name in test_names:
        for col, default in [
            (f"{name}_present", 0),
            (f"{name}_pred", 0),
            (f"{name}_score", np.nan),
            (f"{name}_confidence", np.nan),
            (f"{name}_best_node_id", ""),
            (f"{name}_best_node_source", ""),
            (f"{name}_is_malicious", 0),
        ]:
            if col not in out.columns:
                out[col] = default

        out[f"{name}_present"] = out[f"{name}_present"].fillna(0).astype(int)
        out[f"{name}_pred"] = out[f"{name}_pred"].fillna(0).astype(int)
        out[f"{name}_score"] = pd.to_numeric(out[f"{name}_score"], errors="coerce")

    if "is_malicious_clean" not in out.columns:
        out["is_malicious_clean"] = 0

    out["is_malicious_eval"] = pd.to_numeric(
        out["is_malicious_clean"], errors="coerce"
    ).fillna(0).astype(int)

    clean_pred = out["clean_pred"] if "clean_pred" in out.columns else 0
    corrupted_pred = out["corrupted_pred"] if "corrupted_pred" in out.columns else 0
    remediated_pred = out["remediated_pred"] if "remediated_pred" in out.columns else 0

    out["lost_by_corruption"] = (
        (clean_pred == 1)
        & (corrupted_pred == 0)
        & (out["is_malicious_eval"] == 1)
    ).astype(int)

    out["restored_by_remediation"] = (
        (corrupted_pred == 0)
        & (remediated_pred == 1)
        & (out["is_malicious_eval"] == 1)
    ).astype(int)

    out["still_missed_after_remediation"] = (
        (remediated_pred == 0)
        & (out["is_malicious_eval"] == 1)
    ).astype(int)

    out["new_fp_after_remediation"] = (
        (remediated_pred == 1)
        & (out["is_malicious_eval"] == 0)
    ).astype(int)

    score_cols = [f"{name}_score" for name in ["remediated", "clean", "corrupted"] if f"{name}_score" in out.columns]
    if score_cols:
        sort_score = out[score_cols].bfill(axis=1).iloc[:, 0].fillna(-1)
    else:
        sort_score = pd.Series([-1] * len(out), index=out.index)

    out["_sort_score"] = sort_score
    out = out.sort_values(
        ["is_malicious_eval", "restored_by_remediation", "lost_by_corruption", "_sort_score"],
        ascending=[False, False, False, False],
    ).drop(columns=["_sort_score"])

    return out


# =========================
# Recovered edge reliability analysis
# =========================

def bucket_reliability(x: float) -> str:
    if x < 0.4:
        return "low_<0.4"
    if x < 0.7:
        return "mid_0.4-0.7"
    if x < 1.0:
        return "high_0.7-1.0"
    return "observed_1.0"


def analyze_edge_reliability(stage5_root: Path, out_dir: Path) -> Dict[str, Any]:
    path = stage5_root / "remediated_edge_errors.csv"
    if not path.exists():
        return {"available": False, "reason": "remediated_edge_errors.csv not found"}

    df = read_csv(path)
    if len(df) == 0:
        return {"available": False, "reason": "remediated_edge_errors.csv empty"}

    if "source" not in df.columns:
        df["source"] = "unknown"

    df["source"] = df["source"].astype(str)
    df["reliability"] = numeric_series(df, "reliability", 1.0)
    df["recon_error"] = numeric_series(df, "recon_error", 0.0)

    if "recon_error" not in df.columns:
        return {"available": False, "reason": "recon_error column missing"}

    df["is_attack_edge"] = numeric_series(df, "is_attack_edge", 0).astype(int)
    df["matched_removed_event"] = numeric_series(df, "matched_removed_event", 0).astype(int)
    df["reliability_bucket"] = df["reliability"].map(bucket_reliability)
    df["is_recovered_source"] = df["source"].map(is_recovered_source).astype(int)

    df.to_csv(out_dir / "remediated_edge_reliability_details.csv", index=False, encoding="utf-8-sig")

    group_cols = ["source", "reliability_bucket"]
    summary = df.groupby(group_cols).agg(
        edge_count=("recon_error", "count"),
        reliability_mean=("reliability", "mean"),
        recon_error_mean=("recon_error", "mean"),
        recon_error_median=("recon_error", "median"),
        attack_edges=("is_attack_edge", "sum"),
        matched_removed_events=("matched_removed_event", "sum"),
    ).reset_index()

    summary.to_csv(out_dir / "remediated_edge_reliability_summary.csv", index=False, encoding="utf-8-sig")

    recovered = df[df["is_recovered_source"] == 1].copy()
    top_recovered = recovered.sort_values("recon_error", ascending=False).head(50)
    top_recovered.to_csv(out_dir / "top_recovered_edges_by_error.csv", index=False, encoding="utf-8-sig")

    observed = df[df["source"].astype(str) == "observed"].copy()

    return {
        "available": True,
        "num_edges": int(len(df)),
        "num_recovered_edges": int(len(recovered)),
        "recovered_recon_error_mean": float(recovered["recon_error"].mean()) if len(recovered) else None,
        "observed_recon_error_mean": float(observed["recon_error"].mean()) if len(observed) else None,
        "summary_path": str(out_dir / "remediated_edge_reliability_summary.csv"),
        "details_path": str(out_dir / "remediated_edge_reliability_details.csv"),
    }


def analyze_recovered_copies(stage5_root: Path, stage4_root: Path, out_dir: Path) -> Dict[str, Any]:
    node_scores_path = stage5_root / "remediated_node_instance_scores.csv"
    copy_mapping_path = stage4_root / "test_remediated_graph" / "copy_mapping.csv"

    if not node_scores_path.exists() or not copy_mapping_path.exists():
        return {"available": False, "reason": "node scores or copy mapping missing"}

    node_scores = read_csv(node_scores_path)
    copy_map = read_csv(copy_mapping_path)

    if len(copy_map) == 0 or len(node_scores) == 0:
        return {"available": False, "reason": "copy mapping or node scores empty"}

    if "copy_node_id" not in copy_map.columns or "node_id" not in node_scores.columns:
        return {"available": False, "reason": "copy mapping or node score ID columns missing"}

    merged = copy_map.merge(
        node_scores,
        left_on="copy_node_id",
        right_on="node_id",
        how="left",
        suffixes=("_map", "_score"),
    )

    merged["node_score"] = numeric_series(merged, "node_score", 0.0)

    if "reliability_map" in merged.columns:
        merged["reliability_map"] = numeric_series(merged, "reliability_map", 0.0)
    elif "reliability" in merged.columns:
        merged["reliability_map"] = numeric_series(merged, "reliability", 0.0)
    else:
        merged["reliability_map"] = 0.0

    merged = merged.sort_values("node_score", ascending=False)
    merged.to_csv(out_dir / "recovered_copy_node_scores.csv", index=False, encoding="utf-8-sig")

    return {
        "available": True,
        "num_copy_nodes": int(len(merged)),
        "score_mean": float(merged["node_score"].mean()) if len(merged) else None,
        "score_max": float(merged["node_score"].max()) if len(merged) else None,
        "path": str(out_dir / "recovered_copy_node_scores.csv"),
    }


# =========================
# Report tables
# =========================

def build_metric_table(
    present_metrics: Dict[str, Dict[str, Any]],
    strict_metrics: Dict[str, Dict[str, Any]],
    topk: Dict[str, Dict[str, Any]],
    test_names: List[str],
) -> pd.DataFrame:
    rows = []
    display = {
        "clean": "Complete logs",
        "corrupted": "Corrupted logs",
        "remediated": "Rinnegan-remediated",
    }

    for name in test_names:
        p = present_metrics[name]
        s = strict_metrics[name]
        k = topk[name]
        rows.append({
            "Graph": display.get(name, name),
            "Present-P": p["precision"],
            "Present-R": p["recall"],
            "Present-F1": p["f1"],
            "Strict-P": s["precision"],
            "Strict-R": s["recall"],
            "Strict-F1": s["f1"],
            "TopK-F1": k["f1"],
            "TP": s["tp"],
            "FP": s["fp"],
            "FN": s["fn"],
            "MissingMal": s.get("num_missing_malicious_from_target", 0),
        })

    return pd.DataFrame(rows)


def latex_escape(s: str) -> str:
    s = safe_str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def write_latex_table(metric_table: pd.DataFrame, out_path: Path):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Node-level APT detection diagnostics on the artifact dataset.}")
    lines.append(r"\label{tab:artifact_detection}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Graph & Present-F1 & Strict-F1 & TopK-F1 & TP & FP & FN \\")
    lines.append(r"\midrule")

    for _, r in metric_table.iterrows():
        lines.append(
            f"{latex_escape(r['Graph'])} & "
            f"{float(r['Present-F1']):.4f} & "
            f"{float(r['Strict-F1']):.4f} & "
            f"{float(r['TopK-F1']):.4f} & "
            f"{int(r['TP'])} & {int(r['FP'])} & {int(r['FN'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    write_text(out_path, "\n".join(lines))


# =========================
# Main
# =========================

def main():
    args = parse_args()

    stage5_root = args.stage5_root
    stage4_root = args.stage4_root
    out_dir = args.out_dir
    test_names = parse_test_names(args.test_names)

    ensure_dir(out_dir)

    print("=== Rinnegan Stage-6: Detection Result Analysis ===")
    print(f"STAGE5_ROOT = {stage5_root}")
    print(f"STAGE4_ROOT = {stage4_root}")
    print(f"OUT_DIR     = {out_dir}")
    print(f"TEST_NAMES  = {test_names}")
    print("")

    if "clean" not in test_names:
        raise ValueError("Stage-6 strict evaluation requires 'clean' in --test-names.")

    delta = load_delta(stage5_root)
    scores = load_all_scores(stage5_root, test_names)

    present_metrics = {}
    strict_metrics = {}
    strict_dfs = {}
    topk = {}

    clean_df = scores["clean"]

    for name in test_names:
        present_metrics[name] = compute_present_only_metrics(scores[name], delta)
        strict_metrics[name], strict_dfs[name] = compute_strict_metrics(clean_df, scores[name], delta)
        topk[name] = topk_metrics(scores[name])

        strict_dfs[name].to_csv(out_dir / f"{name}_strict_universe_scores.csv", index=False, encoding="utf-8-sig")

    export_error_lists(scores, delta, out_dir=out_dir)

    entity_comparison = build_entity_comparison(scores, delta, test_names=test_names)
    entity_comparison.to_csv(out_dir / "entity_detection_comparison.csv", index=False, encoding="utf-8-sig")

    restored = entity_comparison[entity_comparison["restored_by_remediation"] == 1].copy()
    lost = entity_comparison[entity_comparison["lost_by_corruption"] == 1].copy()
    still_missed = entity_comparison[entity_comparison["still_missed_after_remediation"] == 1].copy()
    new_fp = entity_comparison[entity_comparison["new_fp_after_remediation"] == 1].copy()

    restored.to_csv(out_dir / "entities_restored_by_remediation.csv", index=False, encoding="utf-8-sig")
    lost.to_csv(out_dir / "entities_lost_by_corruption.csv", index=False, encoding="utf-8-sig")
    still_missed.to_csv(out_dir / "entities_still_missed_after_remediation.csv", index=False, encoding="utf-8-sig")
    new_fp.to_csv(out_dir / "entities_new_fp_after_remediation.csv", index=False, encoding="utf-8-sig")

    edge_rel_info = analyze_edge_reliability(stage5_root, out_dir)
    copy_info = analyze_recovered_copies(stage5_root, stage4_root, out_dir)

    metric_table = build_metric_table(present_metrics, strict_metrics, topk, test_names=test_names)
    metric_table.to_csv(out_dir / "detection_comparison_table.csv", index=False, encoding="utf-8-sig")
    write_latex_table(metric_table, out_dir / "detection_comparison_table.tex")

    summary = {
        "delta": delta,
        "test_names": test_names,
        "present_only_metrics": present_metrics,
        "strict_clean_universe_metrics": strict_metrics,
        "topk_metrics": topk,
        "entity_changes": {
            "lost_by_corruption": int(len(lost)),
            "restored_by_remediation": int(len(restored)),
            "still_missed_after_remediation": int(len(still_missed)),
            "new_fp_after_remediation": int(len(new_fp)),
        },
        "edge_reliability_analysis": edge_rel_info,
        "recovered_copy_analysis": copy_info,
        "outputs": {
            "metric_table": str(out_dir / "detection_comparison_table.csv"),
            "latex_table": str(out_dir / "detection_comparison_table.tex"),
            "entity_comparison": str(out_dir / "entity_detection_comparison.csv"),
            "restored_entities": str(out_dir / "entities_restored_by_remediation.csv"),
            "lost_entities": str(out_dir / "entities_lost_by_corruption.csv"),
            "still_missed_entities": str(out_dir / "entities_still_missed_after_remediation.csv"),
            "entity_error_lists": str(out_dir / "entity_error_lists"),
        }
    }
    write_json(out_dir / "analysis_summary.json", summary)

    lines = []
    lines.append("Rinnegan Stage-6: Detection Result Analysis")
    lines.append("")
    lines.append(f"delta: {delta:.6f}")
    lines.append("")
    lines.append("Present-only metrics")
    for name in test_names:
        m = present_metrics[name]
        lines.append(
            f"  {name}: P={m['precision']}, R={m['recall']}, F1={m['f1']}, "
            f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, "
            f"entities={m['num_entities']}, malicious={m['num_malicious']}"
        )

    lines.append("")
    lines.append("Strict clean-universe metrics")
    for name in test_names:
        m = strict_metrics[name]
        lines.append(
            f"  {name}: P={m['precision']}, R={m['recall']}, F1={m['f1']}, "
            f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, "
            f"missing_malicious={m.get('num_missing_malicious_from_target', 0)}"
        )

    lines.append("")
    lines.append("Top-K ranking metrics")
    for name in test_names:
        m = topk[name]
        lines.append(
            f"  {name}: K={m['k']}, P={m['precision']}, R={m['recall']}, F1={m['f1']}, "
            f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}"
        )

    lines.append("")
    lines.append("Entity-level changes")
    lines.append(f"  lost_by_corruption: {len(lost)}")
    lines.append(f"  restored_by_remediation: {len(restored)}")
    lines.append(f"  still_missed_after_remediation: {len(still_missed)}")
    lines.append(f"  new_fp_after_remediation: {len(new_fp)}")

    if edge_rel_info.get("available"):
        lines.append("")
        lines.append("Recovered-edge reliability analysis")
        lines.append(f"  num_edges: {edge_rel_info['num_edges']}")
        lines.append(f"  num_recovered_edges: {edge_rel_info['num_recovered_edges']}")
        lines.append(f"  recovered_recon_error_mean: {edge_rel_info['recovered_recon_error_mean']}")
        lines.append(f"  observed_recon_error_mean: {edge_rel_info['observed_recon_error_mean']}")
    else:
        lines.append("")
        lines.append("Recovered-edge reliability analysis")
        lines.append(f"  unavailable: {edge_rel_info.get('reason', 'unknown reason')}")

    if copy_info.get("available"):
        lines.append("")
        lines.append("Recovered-copy node analysis")
        lines.append(f"  num_copy_nodes: {copy_info['num_copy_nodes']}")
        lines.append(f"  score_mean: {copy_info['score_mean']}")
        lines.append(f"  score_max: {copy_info['score_max']}")
    else:
        lines.append("")
        lines.append("Recovered-copy node analysis")
        lines.append(f"  unavailable: {copy_info.get('reason', 'unknown reason')}")

    lines.append("")
    lines.append("Detection comparison table")
    for _, r in metric_table.iterrows():
        lines.append(
            f"  {r['Graph']}: Present-F1={r['Present-F1']}, "
            f"Strict-F1={r['Strict-F1']}, TopK-F1={r['TopK-F1']}"
        )

    lines.append("")
    lines.append("Important interpretation")
    lines.append("  Present-only metrics evaluate only entities that appear in each graph.")
    lines.append("  Strict clean-universe metrics additionally penalize malicious entities that disappear from corrupted graphs.")
    lines.append("  For reporting robustness under log removal, strict metrics are more conservative.")
    lines.append("")
    lines.append("Outputs")
    lines.append(f"  {out_dir / 'analysis_summary.txt'}")
    lines.append(f"  {out_dir / 'detection_comparison_table.csv'}")
    lines.append(f"  {out_dir / 'detection_comparison_table.tex'}")
    lines.append(f"  {out_dir / 'entity_detection_comparison.csv'}")
    lines.append(f"  {out_dir / 'entity_error_lists'}")

    summary_text = "\n".join(lines)
    write_text(out_dir / "analysis_summary.txt", summary_text)

    print(summary_text)
    print("")
    print("[DONE] Stage-6 detection result analysis completed.")


if __name__ == "__main__":
    main()