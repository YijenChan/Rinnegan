# -*- coding: utf-8 -*-
"""
Rinnegan Stage-4: Prepare R-GAE training / validation / test graph data.

This script prepares graph-level inputs for the reliability-aware GAE stage.

It creates:
    benign_background_graph/
    train_benign_graph/
    val_benign_graph/
    test_clean_graph/
    test_corrupted_graph/
    test_remediated_graph/
    split_summary.txt
    split_summary.json

Default usage from the repository root:
    python scripts/run_stage4_prepare_rgae_data.py \
        --data-root OfficeFog_desensitized_version/full-dataset \
        --stage3-root outputs/stage3_graph_init \
        --out-dir outputs/stage4_rgae_data

Dependencies:
    pip install pandas
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Set, Tuple

import pandas as pd


# =========================
# Config
# =========================

DATA_ROOT = Path("OfficeFog_desensitized_version/full-dataset")
STAGE3_ROOT = Path("outputs/stage3_graph_init")
OUT_DIR = Path("outputs/stage4_rgae_data")

TRAIN_RATIO = 0.70

# Recommended for benign-only IDS training.
EXCLUDE_MALICIOUS_ENDPOINTS = True

# By default, benign background events inside attack windows are kept.
EXCLUDE_ATTACK_WINDOWS_FOR_TRAIN_VAL = False


# =========================
# Arguments
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rinnegan Stage-4 R-GAE data preparation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--stage3-root", type=Path, default=STAGE3_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--attack-related-dir", type=Path, default=None)

    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument(
        "--keep-malicious-endpoints",
        action="store_true",
        help="Do not remove edges connected to diagnostic malicious endpoints from benign train/val graphs.",
    )
    parser.add_argument(
        "--exclude-attack-windows-for-train-val",
        action="store_true",
        help="Remove all events in attack windows from benign train/val graphs.",
    )
    parser.add_argument(
        "--copy-vocab",
        action="store_true",
        default=True,
        help="Copy relation/node-type vocab files from Stage-3 if available.",
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
    return pd.read_csv(path, encoding="utf-8-sig")


def read_optional_csv(path: Path, default_df: pd.DataFrame = None) -> pd.DataFrame:
    if default_df is None:
        default_df = pd.DataFrame()

    if not path.exists():
        return default_df

    if path.stat().st_size == 0:
        return default_df

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return default_df


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


def to_int_series(s: pd.Series, default: int = 0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(int)


def window_sort_key(w: str) -> int:
    w = safe_str(w)
    if w.startswith("W"):
        try:
            return int(w[1:])
        except Exception:
            return 999999
    return 999999


def copy_dir(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Missing graph folder: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Empty tables
# =========================

def empty_copy_mapping() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "copy_node_id",
        "canonical_node_id",
        "canonical_entity",
        "endpoint_role",
        "hypothesis_id",
        "event_id",
        "reliability",
    ])


def empty_node_features() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "node_id",
        "canonical_id",
        "raw_entity",
        "semantic_text",
        "entity_type",
        "host",
        "source",
        "is_recovered_copy",
        "hypothesis_id",
        "reliability",
        "src_observed",
        "src_recovered_copy",
        "type_process",
        "type_file",
        "type_socket",
        "type_unknown",
    ])


# =========================
# Load attack identifiers
# =========================

def load_attack_event_ids(
    attack_audit_events_path: Path,
    event_labels_path: Path,
    removed_events_path: Path,
) -> Set[str]:
    """
    Attack event IDs are used only to remove attack edges from benign-only
    train/validation graphs.
    """
    attack_ids = set()

    if attack_audit_events_path.exists():
        df = read_csv(attack_audit_events_path)
        if "event_id" in df.columns:
            attack_ids.update(df["event_id"].astype(str).tolist())

    if event_labels_path.exists():
        df = read_csv(event_labels_path)
        if "event_id" in df.columns and "is_attack_event" in df.columns:
            tmp = df[to_int_series(df["is_attack_event"]) == 1]
            attack_ids.update(tmp["event_id"].astype(str).tolist())

    if removed_events_path.exists():
        df = read_csv(removed_events_path)
        if "original_event_id" in df.columns:
            attack_ids.update(df["original_event_id"].astype(str).tolist())

    return {x for x in attack_ids if x}


def load_attack_windows(
    attack_audit_events_path: Path,
    event_labels_path: Path,
    window_labels_path: Path,
) -> Set[str]:
    """
    Attack windows are used only when --exclude-attack-windows-for-train-val is enabled.
    """
    windows = set()

    if attack_audit_events_path.exists():
        df = read_csv(attack_audit_events_path)
        if "window_id" in df.columns:
            windows.update(df["window_id"].astype(str).tolist())

    if event_labels_path.exists():
        df = read_csv(event_labels_path)
        if "window_id" in df.columns and "is_attack_event" in df.columns:
            tmp = df[to_int_series(df["is_attack_event"]) == 1]
            windows.update(tmp["window_id"].astype(str).tolist())

    if window_labels_path.exists():
        df = read_csv(window_labels_path)
        if "window_id" in df.columns:
            if "attack_stages" in df.columns:
                tmp = df[df["attack_stages"].astype(str).str.lower() != "none"]
                windows.update(tmp["window_id"].astype(str).tolist())
            elif "removed_stage" in df.columns:
                tmp = df[df["removed_stage"].astype(str).str.lower() != "none"]
                windows.update(tmp["window_id"].astype(str).tolist())

    return {x for x in windows if x}


def load_malicious_node_ids(clean_nodes: pd.DataFrame) -> Set[str]:
    if "is_malicious" not in clean_nodes.columns:
        return set()

    tmp = clean_nodes[to_int_series(clean_nodes["is_malicious"]) == 1]
    return set(tmp["node_id"].astype(str).tolist())


# =========================
# Graph table loading
# =========================

def load_stage3_graph(graph_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = read_csv(graph_dir / "nodes.csv")
    edges = read_csv(graph_dir / "edges.csv")

    node_features = read_optional_csv(graph_dir / "node_features.csv", default_df=empty_node_features())
    copy_mapping = read_optional_csv(graph_dir / "copy_mapping.csv", default_df=empty_copy_mapping())

    return nodes, edges, node_features, copy_mapping


def validate_stage3_root(stage3_root: Path):
    required_dirs = [
        stage3_root / "clean_graph",
        stage3_root / "corrupted_graph",
        stage3_root / "remediated_graph",
    ]
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing Stage-3 graph folder: {d}")
        for fname in ["nodes.csv", "edges.csv"]:
            if not (d / fname).exists():
                raise FileNotFoundError(f"Missing Stage-3 graph file: {d / fname}")


# =========================
# Benign background filtering
# =========================

def filter_benign_background_edges(
    clean_edges: pd.DataFrame,
    clean_nodes: pd.DataFrame,
    attack_event_ids: Set[str],
    attack_windows: Set[str],
    exclude_malicious_endpoints: bool,
    exclude_attack_windows_for_train_val: bool,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    edges = clean_edges.copy()
    original_count = len(edges)

    if "source" in edges.columns:
        edges = edges[edges["source"].astype(str) == "observed"].copy()
    count_after_source = len(edges)

    if "is_attack_edge" in edges.columns:
        edges = edges[to_int_series(edges["is_attack_edge"]) == 0].copy()
    count_after_attack_label = len(edges)

    if "label" in edges.columns:
        edges = edges[edges["label"].astype(str).str.lower() != "malicious"].copy()
    count_after_label = len(edges)

    if "event_id" in edges.columns:
        edges = edges[~edges["event_id"].astype(str).isin(attack_event_ids)].copy()
    count_after_attack_ids = len(edges)

    if exclude_attack_windows_for_train_val and "window_id" in edges.columns:
        edges = edges[~edges["window_id"].astype(str).isin(attack_windows)].copy()
    count_after_attack_windows = len(edges)

    if exclude_malicious_endpoints:
        mal_nodes = load_malicious_node_ids(clean_nodes)
        if "src_node" in edges.columns and "dst_node" in edges.columns:
            edges = edges[
                (~edges["src_node"].astype(str).isin(mal_nodes))
                & (~edges["dst_node"].astype(str).isin(mal_nodes))
            ].copy()
    count_after_malicious_endpoint = len(edges)

    stats = {
        "original_clean_edges": int(original_count),
        "after_source_observed": int(count_after_source),
        "after_is_attack_edge_filter": int(count_after_attack_label),
        "after_label_filter": int(count_after_label),
        "after_attack_event_id_filter": int(count_after_attack_ids),
        "after_attack_window_filter": int(count_after_attack_windows),
        "after_malicious_endpoint_filter": int(count_after_malicious_endpoint),
        "removed_by_all_filters": int(original_count - count_after_malicious_endpoint),
    }

    edges = sort_edges(edges)
    return edges, stats


def sort_edges(edges: pd.DataFrame) -> pd.DataFrame:
    edges = edges.copy()

    if len(edges) == 0:
        return edges.reset_index(drop=True)

    if "timestamp" in edges.columns:
        edges["_timestamp_sort"] = pd.to_datetime(edges["timestamp"], errors="coerce")
    else:
        edges["_timestamp_sort"] = pd.NaT

    sort_cols = []

    if "window_id" in edges.columns:
        edges["_window_sort"] = edges["window_id"].astype(str).map(window_sort_key)
        sort_cols.append("_window_sort")

    sort_cols.append("_timestamp_sort")

    if "event_id" in edges.columns:
        sort_cols.append("event_id")

    edges = edges.sort_values(sort_cols)
    edges = edges.drop(columns=[c for c in ["_timestamp_sort", "_window_sort"] if c in edges.columns])
    return edges.reset_index(drop=True)


# =========================
# Train / validation split
# =========================

def split_train_val_by_window(
    benign_edges: pd.DataFrame,
    train_ratio: float,
) -> Tuple[List[str], List[str], pd.DataFrame, pd.DataFrame]:
    if "window_id" not in benign_edges.columns:
        raise ValueError("edges.csv must contain window_id for chronological split.")

    windows = sorted(benign_edges["window_id"].astype(str).unique().tolist(), key=window_sort_key)

    if len(windows) == 0:
        raise ValueError("No benign windows found after filtering.")

    if len(windows) == 1:
        train_windows = windows
        val_windows = windows
    else:
        n_train = int(len(windows) * train_ratio)
        n_train = max(1, min(n_train, len(windows) - 1))
        train_windows = windows[:n_train]
        val_windows = windows[n_train:]

    train_edges = benign_edges[benign_edges["window_id"].astype(str).isin(train_windows)].copy()
    val_edges = benign_edges[benign_edges["window_id"].astype(str).isin(val_windows)].copy()

    train_edges = sort_edges(train_edges)
    val_edges = sort_edges(val_edges)

    return train_windows, val_windows, train_edges, val_edges


# =========================
# Graph table export
# =========================

def subset_nodes_by_edges(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    if len(edges) == 0:
        return nodes.iloc[0:0].copy()

    if "src_node" not in edges.columns or "dst_node" not in edges.columns:
        raise ValueError("edges.csv must contain src_node and dst_node columns.")

    used = set(edges["src_node"].astype(str).tolist()) | set(edges["dst_node"].astype(str).tolist())
    out = nodes[nodes["node_id"].astype(str).isin(used)].copy()

    in_deg = Counter(edges["dst_node"].astype(str).tolist())
    out_deg = Counter(edges["src_node"].astype(str).tolist())

    out["in_degree"] = out["node_id"].astype(str).map(lambda x: int(in_deg.get(x, 0)))
    out["out_degree"] = out["node_id"].astype(str).map(lambda x: int(out_deg.get(x, 0)))
    out["total_degree"] = out["in_degree"] + out["out_degree"]

    sort_cols = []
    if "is_recovered_copy" in out.columns:
        sort_cols.append("is_recovered_copy")
    if "is_malicious" in out.columns:
        sort_cols.append("is_malicious")
    if "total_degree" in out.columns:
        sort_cols.append("total_degree")

    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    return out.reset_index(drop=True)


def build_node_features_from_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-input node features from graph nodes.

    Diagnostic label columns such as is_malicious are deliberately excluded.
    They remain in nodes.csv for evaluation but should not be used as model inputs.
    """
    keep_cols = [
        "node_id",
        "canonical_id",
        "raw_entity",
        "semantic_text",
        "entity_type",
        "host",
        "source",
        "is_recovered_copy",
        "hypothesis_id",
        "reliability",
        "src_observed",
        "src_recovered_copy",
        "type_process",
        "type_file",
        "type_socket",
        "type_unknown",
    ]

    cols = [c for c in keep_cols if c in nodes.columns]
    out = nodes[cols].copy()

    for c in empty_node_features().columns:
        if c not in out.columns:
            out[c] = ""

    return out[empty_node_features().columns].copy()


def subset_node_features_by_nodes(node_features: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    if len(nodes) == 0:
        return empty_node_features()

    if len(node_features) == 0 or "node_id" not in node_features.columns:
        return build_node_features_from_nodes(nodes)

    used = set(nodes["node_id"].astype(str).tolist())
    out = node_features[node_features["node_id"].astype(str).isin(used)].copy()

    # Remove diagnostic labels if a previous Stage-3 version exported them in node_features.csv.
    label_like_cols = [
        "is_malicious",
        "label_stage",
        "label_description",
        "attack_step_id",
        "is_attack",
    ]
    out = out.drop(columns=[c for c in label_like_cols if c in out.columns], errors="ignore")

    # Ensure all expected feature columns exist.
    for c in empty_node_features().columns:
        if c not in out.columns:
            out[c] = ""

    return out[empty_node_features().columns].reset_index(drop=True)


def subset_copy_mapping_by_nodes(copy_mapping: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    if len(copy_mapping) == 0:
        return empty_copy_mapping()

    used = set(nodes["node_id"].astype(str).tolist())
    if "copy_node_id" not in copy_mapping.columns:
        return empty_copy_mapping()

    out = copy_mapping[copy_mapping["copy_node_id"].astype(str).isin(used)].copy()

    for c in empty_copy_mapping().columns:
        if c not in out.columns:
            out[c] = ""

    return out[empty_copy_mapping().columns].reset_index(drop=True)


def graph_stats(nodes: pd.DataFrame, edges: pd.DataFrame, graph_name: str) -> Dict[str, Any]:
    stats = {
        "graph_name": graph_name,
        "num_nodes": int(len(nodes)),
        "num_edges": int(len(edges)),
        "num_windows": int(edges["window_id"].nunique()) if "window_id" in edges.columns and len(edges) else 0,
        "windows": sorted(edges["window_id"].astype(str).unique().tolist(), key=window_sort_key)
        if "window_id" in edges.columns and len(edges) else [],
        "num_attack_edges_diagnostic": int(to_int_series(edges["is_attack_edge"]).sum())
        if "is_attack_edge" in edges.columns and len(edges) else 0,
        "num_malicious_nodes_diagnostic": int(to_int_series(nodes["is_malicious"]).sum())
        if "is_malicious" in nodes.columns and len(nodes) else 0,
        "relation_counts": edges["relation"].astype(str).value_counts().to_dict()
        if "relation" in edges.columns and len(edges) else {},
        "node_type_counts": nodes["entity_type"].astype(str).value_counts().to_dict()
        if "entity_type" in nodes.columns and len(nodes) else {},
        "edge_source_counts": edges["source"].astype(str).value_counts().to_dict()
        if "source" in edges.columns and len(edges) else {},
        "node_source_counts": nodes["source"].astype(str).value_counts().to_dict()
        if "source" in nodes.columns and len(nodes) else {},
    }

    if "timestamp" in edges.columns and len(edges):
        ts = pd.to_datetime(edges["timestamp"], errors="coerce")
        stats["time_min"] = str(ts.min())
        stats["time_max"] = str(ts.max())
    else:
        stats["time_min"] = ""
        stats["time_max"] = ""

    return stats


def export_graph_tables(
    out_dir: Path,
    graph_name: str,
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    node_features: pd.DataFrame,
    copy_mapping: pd.DataFrame,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    nodes.to_csv(out_dir / "nodes.csv", index=False, encoding="utf-8-sig")
    edges.to_csv(out_dir / "edges.csv", index=False, encoding="utf-8-sig")
    node_features.to_csv(out_dir / "node_features.csv", index=False, encoding="utf-8-sig")
    copy_mapping.to_csv(out_dir / "copy_mapping.csv", index=False, encoding="utf-8-sig")

    stats = graph_stats(nodes, edges, graph_name)
    write_json(out_dir / "graph_stats.json", stats)

    lines = []
    lines.append(f"Graph: {graph_name}")
    lines.append(f"Nodes: {stats['num_nodes']}")
    lines.append(f"Edges: {stats['num_edges']}")
    lines.append(f"Windows: {stats['num_windows']}")
    lines.append(f"Time range: {stats['time_min']} -> {stats['time_max']}")
    lines.append(f"Attack edges diagnostic: {stats['num_attack_edges_diagnostic']}")
    lines.append(f"Malicious nodes diagnostic: {stats['num_malicious_nodes_diagnostic']}")
    lines.append("")
    lines.append("Windows:")
    lines.append("  " + ", ".join(stats["windows"]))
    lines.append("")
    lines.append("Relation counts:")
    for k, v in stats["relation_counts"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Node type counts:")
    for k, v in stats["node_type_counts"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Edge source counts:")
    for k, v in stats["edge_source_counts"].items():
        lines.append(f"  {k}: {v}")

    write_text(out_dir / "graph_stats.txt", "\n".join(lines))
    return stats


# =========================
# Main
# =========================

def main():
    args = parse_args()

    data_root = args.data_root
    stage3_root = args.stage3_root
    out_dir = args.out_dir
    attack_related_dir = args.attack_related_dir or (data_root / "attack-related-data")

    train_ratio = args.train_ratio
    exclude_malicious_endpoints = not args.keep_malicious_endpoints
    exclude_attack_windows = args.exclude_attack_windows_for_train_val

    if not (0.0 < train_ratio <= 1.0):
        raise ValueError("--train-ratio must be within (0, 1].")

    event_labels = data_root / "labels" / "event_labels.csv"
    entity_labels = data_root / "labels" / "entity_labels.csv"
    removed_events = data_root / "labels" / "removed_events.csv"
    window_labels = data_root / "labels" / "window_labels.csv"
    attack_audit_events = attack_related_dir / "audit_attack_events.csv"

    stage3_clean_graph = stage3_root / "clean_graph"
    stage3_corrupted_graph = stage3_root / "corrupted_graph"
    stage3_remediated_graph = stage3_root / "remediated_graph"

    ensure_dir(out_dir)
    validate_stage3_root(stage3_root)

    print("=== Rinnegan Stage-4: Prepare R-GAE Data ===")
    print(f"DATA_ROOT   = {data_root}")
    print(f"STAGE3_ROOT = {stage3_root}")
    print(f"OUT_DIR     = {out_dir}")
    print(f"TRAIN_RATIO = {train_ratio}")
    print(f"EXCLUDE_MALICIOUS_ENDPOINTS = {exclude_malicious_endpoints}")
    print(f"EXCLUDE_ATTACK_WINDOWS_FOR_TRAIN_VAL = {exclude_attack_windows}")
    print("")

    # Copy vocab files from Stage-3.
    if args.copy_vocab:
        for fname in ["relation_vocab.json", "node_type_vocab.json"]:
            src = stage3_root / fname
            if src.exists():
                shutil.copy2(src, out_dir / fname)

    # Load Stage-3 clean graph.
    clean_nodes, clean_edges, clean_node_features, _ = load_stage3_graph(stage3_clean_graph)

    attack_event_ids = load_attack_event_ids(
        attack_audit_events_path=attack_audit_events,
        event_labels_path=event_labels,
        removed_events_path=removed_events,
    )

    attack_windows = load_attack_windows(
        attack_audit_events_path=attack_audit_events,
        event_labels_path=event_labels,
        window_labels_path=window_labels,
    )

    benign_edges, filter_stats = filter_benign_background_edges(
        clean_edges=clean_edges,
        clean_nodes=clean_nodes,
        attack_event_ids=attack_event_ids,
        attack_windows=attack_windows,
        exclude_malicious_endpoints=exclude_malicious_endpoints,
        exclude_attack_windows_for_train_val=exclude_attack_windows,
    )

    benign_nodes = subset_nodes_by_edges(clean_nodes, benign_edges)
    benign_node_features = subset_node_features_by_nodes(clean_node_features, benign_nodes)
    benign_copy_mapping = empty_copy_mapping()

    benign_stats = export_graph_tables(
        out_dir=out_dir / "benign_background_graph",
        graph_name="benign_background_graph",
        nodes=benign_nodes,
        edges=benign_edges,
        node_features=benign_node_features,
        copy_mapping=benign_copy_mapping,
    )

    train_windows, val_windows, train_edges, val_edges = split_train_val_by_window(
        benign_edges=benign_edges,
        train_ratio=train_ratio,
    )

    train_nodes = subset_nodes_by_edges(clean_nodes, train_edges)
    train_node_features = subset_node_features_by_nodes(clean_node_features, train_nodes)
    train_copy_mapping = empty_copy_mapping()

    val_nodes = subset_nodes_by_edges(clean_nodes, val_edges)
    val_node_features = subset_node_features_by_nodes(clean_node_features, val_nodes)
    val_copy_mapping = empty_copy_mapping()

    train_stats = export_graph_tables(
        out_dir=out_dir / "train_benign_graph",
        graph_name="train_benign_graph",
        nodes=train_nodes,
        edges=train_edges,
        node_features=train_node_features,
        copy_mapping=train_copy_mapping,
    )

    val_stats = export_graph_tables(
        out_dir=out_dir / "val_benign_graph",
        graph_name="val_benign_graph",
        nodes=val_nodes,
        edges=val_edges,
        node_features=val_node_features,
        copy_mapping=val_copy_mapping,
    )

    # Copy full test graphs from Stage-3.
    copy_dir(stage3_clean_graph, out_dir / "test_clean_graph")
    copy_dir(stage3_corrupted_graph, out_dir / "test_corrupted_graph")
    copy_dir(stage3_remediated_graph, out_dir / "test_remediated_graph")

    test_clean_stats = load_json_if_exists(out_dir / "test_clean_graph" / "graph_stats.json")
    test_corrupted_stats = load_json_if_exists(out_dir / "test_corrupted_graph" / "graph_stats.json")
    test_remediated_stats = load_json_if_exists(out_dir / "test_remediated_graph" / "graph_stats.json")

    split_summary = {
        "config": {
            "train_ratio": train_ratio,
            "exclude_malicious_endpoints": exclude_malicious_endpoints,
            "exclude_attack_windows_for_train_val": exclude_attack_windows,
        },
        "input_paths": {
            "data_root": str(data_root),
            "stage3_root": str(stage3_root),
            "attack_related_dir": str(attack_related_dir),
        },
        "attack_event_ids_count": len(attack_event_ids),
        "attack_windows": sorted(list(attack_windows), key=window_sort_key),
        "filter_stats": filter_stats,
        "benign_background_windows": benign_stats["windows"],
        "train_windows": train_windows,
        "val_windows": val_windows,
        "graphs": {
            "benign_background_graph": benign_stats,
            "train_benign_graph": train_stats,
            "val_benign_graph": val_stats,
            "test_clean_graph": test_clean_stats,
            "test_corrupted_graph": test_corrupted_stats,
            "test_remediated_graph": test_remediated_stats,
        },
        "output_paths": {
            "benign_background_graph": str(out_dir / "benign_background_graph"),
            "train_benign_graph": str(out_dir / "train_benign_graph"),
            "val_benign_graph": str(out_dir / "val_benign_graph"),
            "test_clean_graph": str(out_dir / "test_clean_graph"),
            "test_corrupted_graph": str(out_dir / "test_corrupted_graph"),
            "test_remediated_graph": str(out_dir / "test_remediated_graph"),
        },
        "notes": {
            "diagnostic_labels": "Diagnostic labels may appear in nodes.csv/edges.csv for evaluation, but node_features.csv excludes label columns.",
            "training_protocol": "R-GAE should train only on train_benign_graph and calibrate delta on val_benign_graph.",
        }
    }

    write_json(out_dir / "split_summary.json", split_summary)

    lines = []
    lines.append("Rinnegan Stage-4: R-GAE Data Preparation Summary")
    lines.append("")
    lines.append("Configuration")
    lines.append(f"  train_ratio: {train_ratio}")
    lines.append(f"  exclude_malicious_endpoints: {exclude_malicious_endpoints}")
    lines.append(f"  exclude_attack_windows_for_train_val: {exclude_attack_windows}")
    lines.append("")
    lines.append("Attack filtering")
    lines.append(f"  attack_event_ids_count: {len(attack_event_ids)}")
    lines.append(f"  attack_windows: {', '.join(sorted(list(attack_windows), key=window_sort_key))}")
    lines.append("")
    lines.append("Clean graph filtering")
    for k, v in filter_stats.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Benign background graph")
    lines.append(f"  nodes: {benign_stats['num_nodes']}")
    lines.append(f"  edges: {benign_stats['num_edges']}")
    lines.append(f"  windows: {', '.join(benign_stats['windows'])}")
    lines.append("")
    lines.append("Train benign graph")
    lines.append(f"  nodes: {train_stats['num_nodes']}")
    lines.append(f"  edges: {train_stats['num_edges']}")
    lines.append(f"  windows: {', '.join(train_windows)}")
    lines.append("")
    lines.append("Validation benign graph")
    lines.append(f"  nodes: {val_stats['num_nodes']}")
    lines.append(f"  edges: {val_stats['num_edges']}")
    lines.append(f"  windows: {', '.join(val_windows)}")
    lines.append("")
    lines.append("Test graphs copied from Stage-3")
    lines.append(f"  test_clean_graph:      {out_dir / 'test_clean_graph'}")
    lines.append(f"  test_corrupted_graph:  {out_dir / 'test_corrupted_graph'}")
    lines.append(f"  test_remediated_graph: {out_dir / 'test_remediated_graph'}")
    lines.append("")
    lines.append("Feature safety")
    lines.append("  node_features.csv excludes diagnostic label columns such as is_malicious.")
    lines.append("")
    lines.append("Next step")
    lines.append("  Train R-GAE on train_benign_graph.")
    lines.append("  Calibrate delta on val_benign_graph.")
    lines.append("  Evaluate on test_clean_graph, test_corrupted_graph, and test_remediated_graph.")

    summary_text = "\n".join(lines)
    write_text(out_dir / "split_summary.txt", summary_text)

    print(summary_text)
    print("")
    print("[DONE] Stage-4 R-GAE data preparation completed.")


if __name__ == "__main__":
    main()