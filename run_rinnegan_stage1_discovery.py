# -*- coding: utf-8 -*-
"""
Rinnegan Stage-1: Cross-modal corrupted-window discovery.

This script implements the Stage-1 discovery module for the anonymized
Rinnegan artifact package. It extracts aligned audit-log, network-flow,
host-metric, and inconsistency-clue features, trains a cross-modal Transformer
verifier, and exports window-level corruption scores for downstream remediation.

Default usage from the repository root:
    python scripts/run_stage1_discovery.py \
        --data-root OfficeFog_desensitized_version/full-dataset \
        --out-dir outputs/stage1_discovery

The default scoring protocol uses the Transformer consistency score:
    corruption_score = 1 - p_t
where p_t is the learned log-side consistency probability. A hybrid diagnostic
score can be enabled with --score-mode hybrid.
"""

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

DATA_ROOT = Path("OfficeFog_desensitized_version/full-dataset")
OUT_DIR = Path("outputs/stage1_discovery")

WINDOW_SIZE_MINUTES = 10
TIME_START = pd.Timestamp("2026-01-20 09:00:00")
TIME_END = pd.Timestamp("2026-01-20 15:00:00")

SEED = 42

# Methodology setting.
GAMMA = 0.60

# Synthetic corrupted variants per clean window.
K_CORRUPTIONS = 5

# Lightweight configuration for the desensitized artifact package.
D_MODEL = 32
N_HEADS = 4
N_LAYERS = 1
FF_DIM = 96
DROPOUT = 0.20

EPOCHS = 100
LR = 0.001
WEIGHT_DECAY = 1e-4
LAMBDA_CL = 0.30
TAU = 0.20
PATIENCE = 25

# Optional score calibration between neural consistency and explicit clues.
SCORE_MODE = "transformer"  # choices: transformer, clue, hybrid
HYBRID_TRANSFORMER_WEIGHT = 0.50
HYBRID_CLUE_WEIGHT = 1.0 - HYBRID_TRANSFORMER_WEIGHT

DEVICE = "cpu"

# Context export for Stage-2 remediation.
MAX_AUDIT_ROWS_PER_CONTEXT = 80
MAX_FLOW_ROWS_PER_CONTEXT = 80
MAX_METRIC_ROWS_PER_CONTEXT = 30
MAX_METRIC_EVENT_ROWS_PER_CONTEXT = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rinnegan Stage-1 corrupted-window discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="Path to the desensitized dataset root.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Directory for Stage-1 outputs.")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Threshold for selecting candidate corrupted windows.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum training epochs.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Computation device.")
    parser.add_argument(
        "--score-mode",
        type=str,
        default=SCORE_MODE,
        choices=["transformer", "clue", "hybrid"],
        help="Window corruption score used for candidate selection.",
    )
    parser.add_argument(
        "--hybrid-transformer-weight",
        type=float,
        default=HYBRID_TRANSFORMER_WEIGHT,
        help="Transformer weight used only when --score-mode hybrid.",
    )
    parser.add_argument(
        "--context-fallback-top-n",
        type=int,
        default=6,
        help="Number of top-ranked windows exported if no score exceeds gamma.",
    )
    return parser.parse_args()


# =========================
# Feature schema
# =========================

LOG_FEATURES = [
    "log_event_count",
    "log_read_count",
    "log_write_count",
    "log_execute_count",
    "log_spawn_count",
    "log_connect_count",
    "log_accept_count",
    "log_chmod_count",
    "log_unlink_count",
    "log_rename_count",
    "log_unique_process_count",
    "log_unique_src_count",
    "log_unique_dst_count",
    "log_file_dst_count",
    "log_socket_dst_count",
    "log_external_connect_count",
    "log_internal_connect_count",
    "log_root_user_count",
]

NET_FEATURES = [
    "flow_count",
    "flow_external_count",
    "flow_internal_count",
    "flow_unique_dst_ip",
    "flow_unique_dst_port",
    "flow_443_count",
    "flow_80_count",
    "flow_22_count",
    "flow_bytes_out_sum",
    "flow_bytes_in_sum",
    "flow_bytes_out_max",
    "flow_bytes_in_max",
    "flow_packets_out_sum",
    "flow_packets_in_sum",
    "flow_duration_sum",
    "flow_beacon_like_count",
    "flow_payload_like_count",
    "flow_exfil_like_count",
]

METRIC_FEATURES = [
    "metric_sample_count",
    "metric_cpu_mean",
    "metric_cpu_max",
    "metric_mem_mean",
    "metric_mem_max",
    "metric_disk_read_sum",
    "metric_disk_write_sum",
    "metric_disk_write_max",
    "metric_net_in_sum",
    "metric_net_out_sum",
    "metric_net_out_max",
    "metric_proc_count_mean",
    "metric_proc_count_max",
    "metric_open_file_mean",
    "metric_open_file_max",
    "metric_event_count",
    "metric_event_z_mean",
    "metric_event_z_max",
    "metric_event_unique_hosts",
]

CLUE_FEATURES = [
    "clue_unsupported_flow_count",
    "clue_unsupported_external_flow_count",
    "clue_unsupported_internal_flow_count",
    "clue_unsupported_flow_bytes",
    "clue_unsupported_metric_count",
    "clue_unsupported_metric_zsum",
    "clue_unsupported_metric_zmax",
    "clue_flow_log_gap",
    "clue_external_flow_log_gap",
    "clue_metric_log_gap",
    "clue_side_without_log_score",
]

ALL_FEATURES = LOG_FEATURES + NET_FEATURES + METRIC_FEATURES + CLUE_FEATURES

MODALITY_LOG = 0
MODALITY_NET = 1
MODALITY_METRIC = 2
MODALITY_CLUE = 3
NUM_MODALITIES = 4

FEATURE_MODALITIES_LIST = []
for f in ALL_FEATURES:
    if f in LOG_FEATURES:
        FEATURE_MODALITIES_LIST.append(MODALITY_LOG)
    elif f in NET_FEATURES:
        FEATURE_MODALITIES_LIST.append(MODALITY_NET)
    elif f in METRIC_FEATURES:
        FEATURE_MODALITIES_LIST.append(MODALITY_METRIC)
    else:
        FEATURE_MODALITIES_LIST.append(MODALITY_CLUE)

FEATURE_MODALITIES = torch.tensor(FEATURE_MODALITIES_LIST, dtype=torch.long)


# =========================
# Utilities
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


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


def numeric_series(df: pd.DataFrame, col_name: str, default: float = 0.0) -> pd.Series:
    if col_name not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col_name], errors="coerce").fillna(default)


def text_series(df: pd.DataFrame, col_name: str, default: str = "") -> pd.Series:
    if col_name not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=str)
    return df[col_name].astype(str)


def window_ids() -> List[str]:
    total_min = int((TIME_END - TIME_START).total_seconds() // 60)
    n = total_min // WINDOW_SIZE_MINUTES
    return [f"W{i:03d}" for i in range(n)]


def window_sort_key(w: str) -> int:
    w = safe_str(w)
    try:
        return int(w.replace("W", ""))
    except Exception:
        return 999999


def is_external_ip(ip: str) -> bool:
    ip = safe_str(ip)
    if ip == "":
        return False
    return not ip.startswith("10.20.")


def is_external_socket(socket: str) -> bool:
    socket = safe_str(socket)
    if ":" not in socket:
        return False
    ip = socket.split(":", 1)[0]
    return is_external_ip(ip)


def log_transform_feature(name: str, value: float) -> float:
    if any(k in name for k in ["bytes", "packet", "count", "sum", "duration", "unique", "gap"]):
        return math.log1p(max(0.0, float(value)))
    return float(value)


def bounded01(x: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


# =========================
# Load data
# =========================

def load_dataset() -> Dict[str, pd.DataFrame]:
    clean_audit = read_csv(DATA_ROOT / "clean" / "audit_log.csv")
    corrupted_audit = read_csv(DATA_ROOT / "corrupted" / "audit_log.csv")

    network_flow = read_csv(DATA_ROOT / "corrupted" / "network_flow.csv")
    host_metrics = read_csv(DATA_ROOT / "corrupted" / "host_metrics.csv")
    host_metric_events = read_csv(DATA_ROOT / "corrupted" / "host_metric_events.csv")

    window_labels = read_csv(DATA_ROOT / "labels" / "window_labels.csv")
    removed_events = read_csv(DATA_ROOT / "labels" / "removed_events.csv")

    for df in [clean_audit, corrupted_audit]:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "start_time" in network_flow.columns:
        network_flow["start_time"] = pd.to_datetime(network_flow["start_time"], errors="coerce")
    if "end_time" in network_flow.columns:
        network_flow["end_time"] = pd.to_datetime(network_flow["end_time"], errors="coerce")

    if "timestamp" in host_metrics.columns:
        host_metrics["timestamp"] = pd.to_datetime(host_metrics["timestamp"], errors="coerce")

    if "start_time" in host_metric_events.columns:
        host_metric_events["start_time"] = pd.to_datetime(host_metric_events["start_time"], errors="coerce")
    if "end_time" in host_metric_events.columns:
        host_metric_events["end_time"] = pd.to_datetime(host_metric_events["end_time"], errors="coerce")

    return {
        "clean_audit": clean_audit,
        "corrupted_audit": corrupted_audit,
        "network_flow": network_flow,
        "host_metrics": host_metrics,
        "host_metric_events": host_metric_events,
        "window_labels": window_labels,
        "removed_events": removed_events,
    }


# =========================
# Feature extraction
# =========================

def get_relation_series(df: pd.DataFrame) -> pd.Series:
    if "relation" not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype=str)
    return df["relation"].astype(str)


def extract_log_features_raw(audit_w: pd.DataFrame) -> Dict[str, float]:
    d = {k: 0.0 for k in LOG_FEATURES}

    if len(audit_w) == 0:
        return d

    rel = get_relation_series(audit_w)
    dst_entity = text_series(audit_w, "dst_entity", "")
    dst_type = text_series(audit_w, "dst_type", "")

    d["log_event_count"] = float(len(audit_w))
    d["log_read_count"] = float((rel == "read").sum())
    d["log_write_count"] = float((rel == "write").sum())
    d["log_execute_count"] = float((rel == "execute").sum())
    d["log_spawn_count"] = float((rel == "spawn").sum())
    d["log_connect_count"] = float((rel == "connect").sum())
    d["log_accept_count"] = float((rel == "accept").sum())
    d["log_chmod_count"] = float((rel == "chmod").sum())
    d["log_unlink_count"] = float((rel == "unlink").sum())
    d["log_rename_count"] = float((rel == "rename").sum())

    if "process" in audit_w.columns:
        d["log_unique_process_count"] = float(audit_w["process"].astype(str).nunique())
    if "src_entity" in audit_w.columns:
        d["log_unique_src_count"] = float(audit_w["src_entity"].astype(str).nunique())
    if "dst_entity" in audit_w.columns:
        d["log_unique_dst_count"] = float(audit_w["dst_entity"].astype(str).nunique())

    d["log_file_dst_count"] = float((dst_type == "file").sum())
    d["log_socket_dst_count"] = float((dst_type == "socket").sum())

    connect_mask = (rel == "connect") & (dst_type == "socket")
    ext_mask = dst_entity.map(is_external_socket)
    d["log_external_connect_count"] = float((connect_mask & ext_mask).sum())
    d["log_internal_connect_count"] = float((connect_mask & (~ext_mask)).sum())

    if "user" in audit_w.columns:
        d["log_root_user_count"] = float((audit_w["user"].astype(str) == "root").sum())

    return d


def extract_network_features_raw(flow_w: pd.DataFrame) -> Dict[str, float]:
    d = {k: 0.0 for k in NET_FEATURES}

    if len(flow_w) == 0:
        return d

    dst_ip = text_series(flow_w, "dst_ip", "")
    dst_port = numeric_series(flow_w, "dst_port", -1).astype(int)
    external = dst_ip.map(is_external_ip)

    bytes_out = numeric_series(flow_w, "bytes_out", 0.0)
    bytes_in = numeric_series(flow_w, "bytes_in", 0.0)
    packets_out = numeric_series(flow_w, "packets_out", 0.0)
    packets_in = numeric_series(flow_w, "packets_in", 0.0)
    duration = numeric_series(flow_w, "duration_sec", 0.0)

    pattern = text_series(flow_w, "flow_pattern", "").str.lower()

    d["flow_count"] = float(len(flow_w))
    d["flow_external_count"] = float(external.sum())
    d["flow_internal_count"] = float((~external).sum())
    d["flow_unique_dst_ip"] = float(dst_ip.nunique())
    d["flow_unique_dst_port"] = float(dst_port.nunique())
    d["flow_443_count"] = float((dst_port == 443).sum())
    d["flow_80_count"] = float((dst_port == 80).sum())
    d["flow_22_count"] = float((dst_port == 22).sum())

    d["flow_bytes_out_sum"] = float(bytes_out.sum())
    d["flow_bytes_in_sum"] = float(bytes_in.sum())
    d["flow_bytes_out_max"] = float(bytes_out.max()) if len(bytes_out) else 0.0
    d["flow_bytes_in_max"] = float(bytes_in.max()) if len(bytes_in) else 0.0

    d["flow_packets_out_sum"] = float(packets_out.sum())
    d["flow_packets_in_sum"] = float(packets_in.sum())
    d["flow_duration_sum"] = float(duration.sum())

    d["flow_beacon_like_count"] = float(pattern.str.contains("beacon|checkin|periodic", regex=True).sum())
    d["flow_payload_like_count"] = float(pattern.str.contains("payload|download", regex=True).sum())
    d["flow_exfil_like_count"] = float(pattern.str.contains("exfil|upload", regex=True).sum())

    return d


def extract_metric_features_raw(metric_w: pd.DataFrame, metric_event_w: pd.DataFrame) -> Dict[str, float]:
    d = {k: 0.0 for k in METRIC_FEATURES}

    d["metric_sample_count"] = float(len(metric_w))

    cpu = numeric_series(metric_w, "cpu_percent", 0.0)
    mem = numeric_series(metric_w, "mem_percent", 0.0)
    disk_read = numeric_series(metric_w, "disk_read_bytes", 0.0)
    disk_write = numeric_series(metric_w, "disk_write_bytes", 0.0)
    net_in = numeric_series(metric_w, "net_in_bytes", 0.0)
    net_out = numeric_series(metric_w, "net_out_bytes", 0.0)
    proc_count = numeric_series(metric_w, "proc_count", 0.0)
    open_files = numeric_series(metric_w, "file_open_count", 0.0)

    if len(metric_w):
        d["metric_cpu_mean"] = float(cpu.mean())
        d["metric_cpu_max"] = float(cpu.max())
        d["metric_mem_mean"] = float(mem.mean())
        d["metric_mem_max"] = float(mem.max())
        d["metric_disk_read_sum"] = float(disk_read.sum())
        d["metric_disk_write_sum"] = float(disk_write.sum())
        d["metric_disk_write_max"] = float(disk_write.max())
        d["metric_net_in_sum"] = float(net_in.sum())
        d["metric_net_out_sum"] = float(net_out.sum())
        d["metric_net_out_max"] = float(net_out.max())
        d["metric_proc_count_mean"] = float(proc_count.mean())
        d["metric_proc_count_max"] = float(proc_count.max())
        d["metric_open_file_mean"] = float(open_files.mean())
        d["metric_open_file_max"] = float(open_files.max())

    d["metric_event_count"] = float(len(metric_event_w))
    if len(metric_event_w):
        z = numeric_series(metric_event_w, "z_score", 0.0)
        d["metric_event_z_mean"] = float(z.mean())
        d["metric_event_z_max"] = float(z.max())
        if "host" in metric_event_w.columns:
            d["metric_event_unique_hosts"] = float(metric_event_w["host"].astype(str).nunique())

    return d


# =========================
# Inconsistency clues
# =========================

def normalize_token_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-./:@]+", " ", safe_str(text).lower())


def extract_hint_tokens(hint: str) -> List[str]:
    parts = re.split(r"[/,\s]+", safe_str(hint).lower())
    return [p.strip() for p in parts if len(p.strip()) >= 3]


def build_audit_support_maps(audit_w: pd.DataFrame):
    audit_conn_by_host = defaultdict(set)
    audit_text_by_host = defaultdict(str)

    for _, r in audit_w.iterrows():
        host = safe_str(r.get("host"))
        relation = safe_str(r.get("relation"))
        dst_entity = safe_str(r.get("dst_entity"))
        dst_type = safe_str(r.get("dst_type"))

        text_fields = [
            r.get("process", ""),
            r.get("src_entity", ""),
            r.get("dst_entity", ""),
            r.get("relation", ""),
            r.get("command", ""),
            r.get("args", ""),
            r.get("stage", ""),
        ]
        audit_text_by_host[host] += " " + normalize_token_text(" ".join(map(safe_str, text_fields)))

        if relation == "connect" and dst_type == "socket":
            audit_conn_by_host[host].add(dst_entity)

    return audit_conn_by_host, audit_text_by_host


def compute_inconsistency_clues_from_windows(
    audit_w: pd.DataFrame,
    flow_w: pd.DataFrame,
    metric_event_w: pd.DataFrame,
) -> Dict[str, Any]:
    audit_conn_by_host, audit_text_by_host = build_audit_support_maps(audit_w)

    unsupported_flows = []
    unsupported_external = 0
    unsupported_internal = 0
    unsupported_bytes = 0.0

    for _, f in flow_w.iterrows():
        src_host = safe_str(f.get("src_host"))
        dst_ip = safe_str(f.get("dst_ip"))
        dst_port = safe_int(f.get("dst_port"), -1)
        target = f"{dst_ip}:{dst_port}"

        has_support = target in audit_conn_by_host.get(src_host, set())

        if not has_support:
            out_b = safe_float(f.get("bytes_out"), 0.0)
            in_b = safe_float(f.get("bytes_in"), 0.0)
            unsupported_bytes += out_b + in_b

            if is_external_ip(dst_ip):
                unsupported_external += 1
            else:
                unsupported_internal += 1

            unsupported_flows.append({
                "flow_id": safe_str(f.get("flow_id")),
                "src_host": src_host,
                "dst": target,
                "service": safe_str(f.get("service")),
                "bytes_out": int(out_b),
                "bytes_in": int(in_b),
                "flow_pattern": safe_str(f.get("flow_pattern")),
            })

    unsupported_metric_clues = []
    zsum = 0.0
    zmax = 0.0

    for _, m in metric_event_w.iterrows():
        host = safe_str(m.get("host"))
        hint = safe_str(m.get("related_hint"))
        tokens = extract_hint_tokens(hint)
        audit_text = audit_text_by_host.get(host, "")

        found = any(t in audit_text for t in tokens)
        if not found:
            z = safe_float(m.get("z_score"), 0.0)
            zsum += z
            zmax = max(zmax, z)
            unsupported_metric_clues.append({
                "metric_event_id": safe_str(m.get("metric_event_id")),
                "host": host,
                "event_type": safe_str(m.get("event_type")),
                "related_hint": hint,
                "z_score": z,
                "description": safe_str(m.get("description")),
            })

    return {
        "unsupported_flow_count": int(len(unsupported_flows)),
        "unsupported_external_flow_count": int(unsupported_external),
        "unsupported_internal_flow_count": int(unsupported_internal),
        "unsupported_flow_bytes": float(unsupported_bytes),
        "unsupported_metric_clue_count": int(len(unsupported_metric_clues)),
        "unsupported_metric_zsum": float(zsum),
        "unsupported_metric_zmax": float(zmax),
        "clue_unsupported_flows": unsupported_flows,
        "clue_unsupported_metric_clues": unsupported_metric_clues,
    }


def extract_clue_features_raw(
    log_d: Dict[str, float],
    net_d: Dict[str, float],
    metric_d: Dict[str, float],
    clue_d: Dict[str, Any],
) -> Dict[str, float]:
    flow_count = float(net_d.get("flow_count", 0.0))
    external_flow_count = float(net_d.get("flow_external_count", 0.0))
    metric_event_count = float(metric_d.get("metric_event_count", 0.0))

    log_event_count = float(log_d.get("log_event_count", 0.0))
    log_connect_count = float(log_d.get("log_connect_count", 0.0))
    log_external_connect_count = float(log_d.get("log_external_connect_count", 0.0))

    flow_log_gap = max(0.0, flow_count - log_connect_count)
    external_gap = max(0.0, external_flow_count - log_external_connect_count)
    metric_log_gap = max(0.0, metric_event_count - log_event_count * 0.05)

    side_activity = (
        0.7 * flow_count
        + 0.002 * float(net_d.get("flow_bytes_out_sum", 0.0))
        + 0.5 * metric_event_count
        + 0.2 * float(metric_d.get("metric_event_z_max", 0.0))
    )
    log_activity = (
        log_event_count
        + 1.5 * log_connect_count
        + float(log_d.get("log_write_count", 0.0))
        + float(log_d.get("log_execute_count", 0.0))
        + float(log_d.get("log_spawn_count", 0.0))
    )

    side_without_log_score = max(0.0, math.log1p(side_activity) - math.log1p(log_activity))

    return {
        "clue_unsupported_flow_count": float(clue_d.get("unsupported_flow_count", 0.0)),
        "clue_unsupported_external_flow_count": float(clue_d.get("unsupported_external_flow_count", 0.0)),
        "clue_unsupported_internal_flow_count": float(clue_d.get("unsupported_internal_flow_count", 0.0)),
        "clue_unsupported_flow_bytes": float(clue_d.get("unsupported_flow_bytes", 0.0)),
        "clue_unsupported_metric_count": float(clue_d.get("unsupported_metric_clue_count", 0.0)),
        "clue_unsupported_metric_zsum": float(clue_d.get("unsupported_metric_zsum", 0.0)),
        "clue_unsupported_metric_zmax": float(clue_d.get("unsupported_metric_zmax", 0.0)),
        "clue_flow_log_gap": flow_log_gap,
        "clue_external_flow_log_gap": external_gap,
        "clue_metric_log_gap": metric_log_gap,
        "clue_side_without_log_score": side_without_log_score,
    }


def compute_clue_score(raw_fd: Dict[str, float]) -> float:
    u_flow = float(raw_fd.get("clue_unsupported_flow_count", 0.0))
    u_ext = float(raw_fd.get("clue_unsupported_external_flow_count", 0.0))
    u_int = float(raw_fd.get("clue_unsupported_internal_flow_count", 0.0))
    u_bytes = float(raw_fd.get("clue_unsupported_flow_bytes", 0.0))
    u_metric = float(raw_fd.get("clue_unsupported_metric_count", 0.0))
    zsum = float(raw_fd.get("clue_unsupported_metric_zsum", 0.0))
    ext_gap = float(raw_fd.get("clue_external_flow_log_gap", 0.0))
    side_wo_log = float(raw_fd.get("clue_side_without_log_score", 0.0))

    weighted = (
        1.00 * u_flow
        + 1.60 * u_ext
        + 0.25 * u_int
        + 1.10 * u_metric
        + 0.28 * zsum
        + 0.18 * math.log1p(max(0.0, u_bytes))
        + 0.55 * ext_gap
        + 0.75 * side_wo_log
    )

    return bounded01(1.0 - math.exp(-weighted / 4.0))


def extract_raw_feature_dict(
    audit_w: pd.DataFrame,
    flow_w: pd.DataFrame,
    metric_w: pd.DataFrame,
    metric_event_w: pd.DataFrame,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    log_d = extract_log_features_raw(audit_w)
    net_d = extract_network_features_raw(flow_w)
    metric_d = extract_metric_features_raw(metric_w, metric_event_w)
    clue_d = compute_inconsistency_clues_from_windows(audit_w, flow_w, metric_event_w)
    clue_features = extract_clue_features_raw(log_d, net_d, metric_d, clue_d)

    raw = {}
    raw.update(log_d)
    raw.update(net_d)
    raw.update(metric_d)
    raw.update(clue_features)

    return raw, clue_d


def transform_feature_dict(raw: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k in ALL_FEATURES:
        out[k] = log_transform_feature(k, float(raw.get(k, 0.0)))
    return out


def extract_feature_dict(
    audit_w: pd.DataFrame,
    flow_w: pd.DataFrame,
    metric_w: pd.DataFrame,
    metric_event_w: pd.DataFrame,
) -> Dict[str, float]:
    raw, _ = extract_raw_feature_dict(audit_w, flow_w, metric_w, metric_event_w)
    return transform_feature_dict(raw)


# =========================
# Synthetic corruption generation
# =========================

def drop_rows_by_mask(audit_w: pd.DataFrame, mask: pd.Series, fallback_rate: float, rng: random.Random) -> pd.DataFrame:
    if len(audit_w) == 0:
        return audit_w.copy()

    idx = audit_w[mask].index.tolist()

    if not idx:
        n = max(1, int(len(audit_w) * fallback_rate))
        n = min(n, len(audit_w))
        idx = rng.sample(audit_w.index.tolist(), n)

    return audit_w.drop(index=idx).copy()


def make_synthetic_corruption(audit_w: pd.DataFrame, op_id: int, rng: random.Random) -> pd.DataFrame:
    if len(audit_w) == 0:
        return audit_w.copy()

    rel = text_series(audit_w, "relation", "")
    dst_type = text_series(audit_w, "dst_type", "")
    dst_entity = text_series(audit_w, "dst_entity", "")
    label = text_series(audit_w, "label", "")

    if op_id == 0:
        mask = (rel == "connect") & (dst_type == "socket") & dst_entity.map(is_external_socket)
        return drop_rows_by_mask(audit_w, mask, fallback_rate=0.10, rng=rng)

    if op_id == 1:
        mask = rel.isin(["spawn", "execute"])
        return drop_rows_by_mask(audit_w, mask, fallback_rate=0.12, rng=rng)

    if op_id == 2:
        mask = rel.isin(["write", "chmod", "rename", "unlink"])
        return drop_rows_by_mask(audit_w, mask, fallback_rate=0.12, rng=rng)

    if op_id == 3:
        mask = label.str.lower().eq("malicious")
        if mask.sum() == 0:
            mask = rel.isin(["connect", "spawn", "execute", "write", "chmod"])
        return drop_rows_by_mask(audit_w, mask, fallback_rate=0.18, rng=rng)

    n = max(1, int(len(audit_w) * 0.22))
    n = min(n, len(audit_w))

    if "timestamp" in audit_w.columns:
        idx_sorted = audit_w.sort_values("timestamp").index.tolist()
    else:
        idx_sorted = audit_w.index.tolist()

    if len(idx_sorted) <= n:
        remove_idx = idx_sorted
    else:
        start = rng.randrange(0, len(idx_sorted) - n + 1)
        remove_idx = idx_sorted[start:start + n]

    return audit_w.drop(index=remove_idx).copy()


# =========================
# Sample construction
# =========================

class FeatureNormalizer:
    def __init__(self):
        self.mean = {}
        self.std = {}

    def fit(self, feature_dicts: List[Dict[str, float]]):
        for k in ALL_FEATURES:
            vals = np.array([float(d.get(k, 0.0)) for d in feature_dicts], dtype=np.float32)
            self.mean[k] = float(vals.mean())
            sd = float(vals.std())
            self.std[k] = sd if sd > 1e-6 else 1.0

    def transform(self, d: Dict[str, float]) -> np.ndarray:
        vals = []
        for k in ALL_FEATURES:
            v = float(d.get(k, 0.0))
            vals.append((v - self.mean[k]) / self.std[k])
        return np.array(vals, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean, "std": self.std}


def build_samples(data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    clean_audit = data["clean_audit"]
    flow = data["network_flow"]
    metrics = data["host_metrics"]
    metric_events = data["host_metric_events"]

    samples = []
    rng = random.Random(SEED)

    for w in window_ids():
        audit_w = clean_audit[clean_audit["window_id"].astype(str) == w].copy()
        flow_w = flow[flow["window_id"].astype(str) == w].copy()
        metric_w = metrics[metrics["window_id"].astype(str) == w].copy()
        metric_event_w = metric_events[metric_events["window_id"].astype(str) == w].copy()

        raw_pos, _ = extract_raw_feature_dict(audit_w, flow_w, metric_w, metric_event_w)
        pos_features = transform_feature_dict(raw_pos)

        samples.append({
            "window_id": w,
            "variant": "clean",
            "label": 1,
            "features": pos_features,
            "clue_score": compute_clue_score(raw_pos),
        })

        for k in range(K_CORRUPTIONS):
            corrupted_audit_w = make_synthetic_corruption(audit_w, k, rng)
            raw_neg, _ = extract_raw_feature_dict(corrupted_audit_w, flow_w, metric_w, metric_event_w)
            neg_features = transform_feature_dict(raw_neg)

            samples.append({
                "window_id": w,
                "variant": f"synthetic_corrupted_{k}",
                "label": 0,
                "features": neg_features,
                "clue_score": compute_clue_score(raw_neg),
            })

    return samples


def split_samples_by_window(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str]]:
    wins = sorted(set(s["window_id"] for s in samples), key=window_sort_key)
    n_train = max(1, int(len(wins) * 0.70))
    n_train = min(n_train, len(wins) - 1)

    train_wins = set(wins[:n_train])
    val_wins = set(wins[n_train:])

    train_samples = [s for s in samples if s["window_id"] in train_wins]
    val_samples = [s for s in samples if s["window_id"] in val_wins]

    return train_samples, val_samples, sorted(train_wins, key=window_sort_key), sorted(val_wins, key=window_sort_key)


def samples_to_tensors(samples: List[Dict[str, Any]], normalizer: FeatureNormalizer) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    x = np.stack([normalizer.transform(s["features"]) for s in samples], axis=0)
    y = np.array([s["label"] for s in samples], dtype=np.float32)
    groups = [s["window_id"] for s in samples]

    return (
        torch.tensor(x, dtype=torch.float32, device=DEVICE),
        torch.tensor(y, dtype=torch.float32, device=DEVICE),
        groups,
    )


# =========================
# Model
# =========================

class CrossModalTransformerVerifier(nn.Module):
    def __init__(self, num_features: int, d_model: int, n_heads: int, n_layers: int, ff_dim: int, dropout: float):
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        self.value_proj = nn.Linear(1, d_model)
        self.feature_embed = nn.Embedding(num_features, d_model)
        self.modality_embed = nn.Embedding(NUM_MODALITIES, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.log_proj = nn.Linear(d_model, d_model)
        self.side_proj = nn.Linear(d_model, d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        feature_ids = torch.arange(num_features, dtype=torch.long)
        self.register_buffer("feature_ids", feature_ids)
        self.register_buffer("feature_modalities", FEATURE_MODALITIES.clone())

        log_mask = torch.tensor([m == MODALITY_LOG for m in FEATURE_MODALITIES.tolist()], dtype=torch.bool)
        side_mask = torch.tensor(
            [m in [MODALITY_NET, MODALITY_METRIC, MODALITY_CLUE] for m in FEATURE_MODALITIES.tolist()],
            dtype=torch.bool,
        )

        self.register_buffer("log_mask", log_mask)
        self.register_buffer("side_mask", side_mask)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_features = x.shape
        assert num_features == self.num_features

        values = x.unsqueeze(-1)
        h = self.value_proj(values)

        feat_emb = self.feature_embed(self.feature_ids).unsqueeze(0).expand(bsz, -1, -1)
        mod_emb = self.modality_embed(self.feature_modalities).unsqueeze(0).expand(bsz, -1, -1)

        h = h + feat_emb + mod_emb
        enc = self.encoder(h)

        log_tokens = enc[:, self.log_mask, :]
        side_tokens = enc[:, self.side_mask, :]

        h_log = self.log_proj(log_tokens.mean(dim=1))
        h_side = self.side_proj(side_tokens.mean(dim=1))

        pair = torch.cat([h_log, h_side, torch.abs(h_log - h_side), h_log * h_side], dim=-1)
        logits = self.head(pair).squeeze(-1)

        return logits, h_log, h_side


# =========================
# Losses and training
# =========================

def compute_infonce_loss(h_log: torch.Tensor, h_side: torch.Tensor, labels: torch.Tensor, groups: List[str]) -> torch.Tensor:
    total = []
    unique_groups = sorted(set(groups), key=window_sort_key)

    for g in unique_groups:
        idx = [i for i, x in enumerate(groups) if x == g]
        if not idx:
            continue

        pos_idx = [i for i in idx if float(labels[i].item()) == 1.0]
        neg_idx = [i for i in idx if float(labels[i].item()) == 0.0]

        if not pos_idx or not neg_idx:
            continue

        p = pos_idx[0]
        side = h_side[p:p + 1]

        pos_sim = F.cosine_similarity(h_log[p:p + 1], side, dim=-1) / TAU
        neg_log = h_log[neg_idx]
        neg_side = side.expand(len(neg_idx), -1)
        neg_sim = F.cosine_similarity(neg_log, neg_side, dim=-1) / TAU

        logits = torch.cat([pos_sim, neg_sim], dim=0).unsqueeze(0)
        target = torch.tensor([0], dtype=torch.long, device=DEVICE)
        loss = F.cross_entropy(logits, target)
        total.append(loss)

    if not total:
        return torch.tensor(0.0, dtype=torch.float32, device=DEVICE)

    return torch.stack(total).mean()


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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
    }


def evaluate_samples(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x)
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, y).item()
        pred = (prob >= 0.5).float()

    y_np = y.detach().cpu().numpy().astype(int)
    p_np = pred.detach().cpu().numpy().astype(int)
    m = compute_binary_metrics(y_np, p_np)

    return {
        "bce": bce,
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "accuracy": m["accuracy"],
    }


def train_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_groups: List[str],
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    val_groups: List[str],
) -> Tuple[CrossModalTransformerVerifier, pd.DataFrame]:
    model = CrossModalTransformerVerifier(
        num_features=len(ALL_FEATURES),
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_val_f1 = -1.0
    best_val_bce = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        logits, h_log, h_side = model(train_x)
        bce = F.binary_cross_entropy_with_logits(logits, train_y)
        cl = compute_infonce_loss(h_log, h_side, train_y, train_groups)
        loss = bce + LAMBDA_CL * cl

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            train_metrics = evaluate_samples(model, train_x, train_y)
            val_metrics = evaluate_samples(model, val_x, val_y)

            row = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "bce_loss": float(bce.item()),
                "contrastive_loss": float(cl.item()),
                "train_bce": train_metrics["bce"],
                "train_f1": train_metrics["f1"],
                "train_accuracy": train_metrics["accuracy"],
                "val_bce": val_metrics["bce"],
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
            }
            history.append(row)

            print(
                f"Epoch {epoch:04d} | loss={loss.item():.4f} "
                f"bce={bce.item():.4f} cl={cl.item():.4f} | "
                f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f}"
            )

            improved = False
            if val_metrics["f1"] > best_val_f1:
                improved = True
            elif val_metrics["f1"] == best_val_f1 and val_metrics["bce"] < best_val_bce:
                improved = True

            if improved:
                best_val_f1 = val_metrics["f1"]
                best_val_bce = val_metrics["bce"]
                best_epoch = epoch
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 10

            if epoch >= 30 and bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_val_f1={best_val_f1:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    history_df = pd.DataFrame(history)
    history_df["best_epoch"] = best_epoch
    history_df["best_val_f1"] = best_val_f1
    history_df["best_val_bce"] = best_val_bce

    return model, history_df


# =========================
# Test scoring
# =========================

def get_window_slices(
    window_id: str,
    audit_df: pd.DataFrame,
    data: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_w = audit_df[audit_df["window_id"].astype(str) == window_id].copy()
    flow_w = data["network_flow"][data["network_flow"]["window_id"].astype(str) == window_id].copy()
    metric_w = data["host_metrics"][data["host_metrics"]["window_id"].astype(str) == window_id].copy()
    metric_event_w = data["host_metric_events"][data["host_metric_events"]["window_id"].astype(str) == window_id].copy()
    return audit_w, flow_w, metric_w, metric_event_w


def build_window_feature_dict(window_id: str, audit_df: pd.DataFrame, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Any]]:
    audit_w, flow_w, metric_w, metric_event_w = get_window_slices(window_id, audit_df, data)
    raw_fd, clue_d = extract_raw_feature_dict(audit_w, flow_w, metric_w, metric_event_w)
    model_fd = transform_feature_dict(raw_fd)
    return raw_fd, model_fd, clue_d


def score_windows(
    model: nn.Module,
    audit_df: pd.DataFrame,
    data: Dict[str, pd.DataFrame],
    normalizer: FeatureNormalizer,
) -> pd.DataFrame:
    rows = []
    raw_dicts = []
    model_dicts = []
    clue_dicts = []

    for w in window_ids():
        raw_fd, model_fd, clue_d = build_window_feature_dict(w, audit_df, data)
        raw_dicts.append(raw_fd)
        model_dicts.append(model_fd)
        clue_dicts.append(clue_d)

    x_np = np.stack([normalizer.transform(fd) for fd in model_dicts], axis=0)
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.no_grad():
        logits, h_log, h_side = model(x)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        sim = F.cosine_similarity(h_log, h_side, dim=-1).detach().cpu().numpy()

    for i, w in enumerate(window_ids()):
        raw_fd = raw_dicts[i]
        clue_d = clue_dicts[i]

        transformer_score = bounded01(1.0 - float(p[i]))
        clue_score = compute_clue_score(raw_fd)
        hybrid_score = bounded01(
            HYBRID_TRANSFORMER_WEIGHT * transformer_score
            + HYBRID_CLUE_WEIGHT * clue_score
        )

        if SCORE_MODE == "transformer":
            final_score = transformer_score
        elif SCORE_MODE == "clue":
            final_score = clue_score
        else:
            final_score = hybrid_score

        rows.append({
            "window_id": w,
            "consistency_probability": float(p[i]),
            "transformer_corruption_score": transformer_score,
            "clue_score": clue_score,
            "hybrid_corruption_score": hybrid_score,
            "corruption_score": final_score,
            "score_mode": SCORE_MODE,
            "log_side_similarity": float(sim[i]),

            "unsupported_flow_count": int(clue_d.get("unsupported_flow_count", 0)),
            "unsupported_external_flow_count": int(clue_d.get("unsupported_external_flow_count", 0)),
            "unsupported_internal_flow_count": int(clue_d.get("unsupported_internal_flow_count", 0)),
            "unsupported_flow_bytes": float(clue_d.get("unsupported_flow_bytes", 0.0)),
            "unsupported_metric_clue_count": int(clue_d.get("unsupported_metric_clue_count", 0)),
            "unsupported_metric_zsum": float(clue_d.get("unsupported_metric_zsum", 0.0)),
            "unsupported_metric_zmax": float(clue_d.get("unsupported_metric_zmax", 0.0)),

            "clue_flow_log_gap": float(raw_fd.get("clue_flow_log_gap", 0.0)),
            "clue_external_flow_log_gap": float(raw_fd.get("clue_external_flow_log_gap", 0.0)),
            "clue_metric_log_gap": float(raw_fd.get("clue_metric_log_gap", 0.0)),
            "clue_side_without_log_score": float(raw_fd.get("clue_side_without_log_score", 0.0)),

            "clue_unsupported_flows": clue_d.get("clue_unsupported_flows", []),
            "clue_unsupported_metric_clues": clue_d.get("clue_unsupported_metric_clues", []),
        })

    return pd.DataFrame(rows)


# =========================
# Label attachment and evaluation
# =========================

def attach_labels(score_df: pd.DataFrame, window_labels: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "window_id",
        "is_corrupted",
        "removed_event_count",
        "attack_stages",
        "affected_hosts",
        "notes",
    ]
    available = [c for c in cols if c in window_labels.columns]
    out = score_df.merge(window_labels[available], on="window_id", how="left")
    out["is_corrupted"] = out["is_corrupted"].fillna(0).astype(int)
    out["removed_event_count"] = out["removed_event_count"].fillna(0).astype(int)
    return out


def threshold_sweep(score_df: pd.DataFrame, score_col: str = "corruption_score") -> pd.DataFrame:
    rows = []
    y = score_df["is_corrupted"].astype(int).values

    for th in np.linspace(0.05, 0.95, 19):
        pred = (score_df[score_col].values > th).astype(int)
        m = compute_binary_metrics(y, pred)
        rows.append({"threshold": round(float(th), 4), "score_col": score_col, **m})

    return pd.DataFrame(rows)


def evaluate_discovery(score_df: pd.DataFrame, gamma: float, score_col: str = "corruption_score") -> Dict[str, Any]:
    df = score_df.copy()
    y_true = df["is_corrupted"].astype(int).values

    pred_gamma = (df[score_col].values > gamma).astype(int)
    gamma_metrics = compute_binary_metrics(y_true, pred_gamma)
    gamma_windows = df[pred_gamma == 1]["window_id"].astype(str).tolist()

    k = int(y_true.sum())
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    pred_topk = np.zeros(len(ranked), dtype=int)
    if k > 0:
        pred_topk[:k] = 1

    topk_metrics = compute_binary_metrics(ranked["is_corrupted"].astype(int).values, pred_topk)
    topk_windows = ranked.head(k)["window_id"].astype(str).tolist() if k > 0 else []

    sweep = threshold_sweep(df, score_col=score_col)
    best = sweep.sort_values(["f1", "precision", "recall"], ascending=[False, False, False]).head(1).iloc[0].to_dict()

    return {
        "score_col": score_col,
        "gamma": gamma,
        "gamma_metrics": gamma_metrics,
        "gamma_windows": gamma_windows,
        "topk_k": k,
        "topk_metrics": topk_metrics,
        "topk_windows": topk_windows,
        "diagnostic_best_threshold_from_eval_sweep": best,
    }


# =========================
# Context export
# =========================

def shorten_records(rows: List[Any], max_rows: int = 20) -> List[Any]:
    if len(rows) <= max_rows:
        return rows
    return rows[:max_rows] + [f"... ({len(rows) - max_rows} more rows omitted)"]


def export_window_contexts(
    score_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    data: Dict[str, pd.DataFrame],
    out_dir: Path,
    fallback_top_n: int = 6,
):
    context_dir = out_dir / "window_contexts"
    ensure_dir(context_dir)

    selected_df = score_df[score_df["corruption_score"] > GAMMA].sort_values("corruption_score", ascending=False)
    if selected_df.empty:
        selected_df = score_df.sort_values("corruption_score", ascending=False).head(max(1, fallback_top_n))
    selected = selected_df["window_id"].astype(str).tolist()

    for w in selected:
        audit_w, flow_w, metric_w, metric_event_w = get_window_slices(w, audit_df, data)

        row = score_df[score_df["window_id"].astype(str) == w].iloc[0].to_dict()
        unsupported_flows = row.get("clue_unsupported_flows", [])
        unsupported_metrics = row.get("clue_unsupported_metric_clues", [])

        lines = []
        lines.append(f"# Window Context: {w}")
        lines.append("")
        lines.append(f"consistency_probability: {row.get('consistency_probability')}")
        lines.append(f"transformer_corruption_score: {row.get('transformer_corruption_score')}")
        lines.append(f"clue_score: {row.get('clue_score')}")
        lines.append(f"corruption_score: {row.get('corruption_score')}")
        lines.append(f"log_side_similarity: {row.get('log_side_similarity')}")
        lines.append(f"unsupported_flow_count: {row.get('unsupported_flow_count')}")
        lines.append(f"unsupported_external_flow_count: {row.get('unsupported_external_flow_count')}")
        lines.append(f"unsupported_metric_clue_count: {row.get('unsupported_metric_clue_count')}")
        lines.append(f"unsupported_metric_zsum: {row.get('unsupported_metric_zsum')}")
        lines.append("")

        lines.append("## Observed audit events in corrupted logs")
        audit_cols = ["event_id", "timestamp", "host", "user", "process", "relation", "dst_entity", "dst_type", "command", "args"]
        audit_cols = [c for c in audit_cols if c in audit_w.columns]
        for _, r in audit_w[audit_cols].head(MAX_AUDIT_ROWS_PER_CONTEXT).iterrows():
            fields = [safe_str(r.get(c, "")) for c in audit_cols]
            lines.append("- " + " | ".join(fields))
        if len(audit_w) > MAX_AUDIT_ROWS_PER_CONTEXT:
            lines.append(f"... ({len(audit_w) - MAX_AUDIT_ROWS_PER_CONTEXT} more audit events omitted)")
        lines.append("")

        lines.append("## Unsupported network flows")
        if isinstance(unsupported_flows, str):
            unsupported_flows = []
        for f in shorten_records(unsupported_flows, max_rows=30):
            if isinstance(f, str):
                lines.append(f"- {f}")
            else:
                lines.append(
                    f"- {f.get('flow_id')} | {f.get('src_host')} -> {f.get('dst')} | "
                    f"{f.get('service')} | out={f.get('bytes_out')} in={f.get('bytes_in')} | "
                    f"pattern={f.get('flow_pattern')}"
                )
        lines.append("")

        lines.append("## Unsupported host metric clues")
        if isinstance(unsupported_metrics, str):
            unsupported_metrics = []
        for m in shorten_records(unsupported_metrics, max_rows=30):
            if isinstance(m, str):
                lines.append(f"- {m}")
            else:
                lines.append(
                    f"- {m.get('metric_event_id')} | {m.get('host')} | {m.get('event_type')} | "
                    f"hint={m.get('related_hint')} | z={m.get('z_score')} | {m.get('description')}"
                )
        lines.append("")

        lines.append("## All network flows in this window")
        flow_cols = ["flow_id", "start_time", "src_host", "src_ip", "dst_ip", "dst_port", "service", "bytes_out", "bytes_in", "flow_pattern"]
        flow_cols = [c for c in flow_cols if c in flow_w.columns]
        for _, r in flow_w[flow_cols].head(MAX_FLOW_ROWS_PER_CONTEXT).iterrows():
            lines.append("- " + " | ".join([safe_str(r.get(c, "")) for c in flow_cols]))
        if len(flow_w) > MAX_FLOW_ROWS_PER_CONTEXT:
            lines.append(f"... ({len(flow_w) - MAX_FLOW_ROWS_PER_CONTEXT} more flows omitted)")
        lines.append("")

        lines.append("## Host metric samples")
        metric_cols = ["metric_id", "timestamp", "host", "cpu_percent", "mem_percent", "disk_write_bytes", "net_out_bytes", "proc_count", "file_open_count", "top_process_hint", "metric_note"]
        metric_cols = [c for c in metric_cols if c in metric_w.columns]
        for _, r in metric_w[metric_cols].head(MAX_METRIC_ROWS_PER_CONTEXT).iterrows():
            lines.append("- " + " | ".join([safe_str(r.get(c, "")) for c in metric_cols]))
        lines.append("")

        lines.append("## Host metric events")
        me_cols = ["metric_event_id", "start_time", "host", "event_type", "related_hint", "z_score", "description"]
        me_cols = [c for c in me_cols if c in metric_event_w.columns]
        for _, r in metric_event_w[me_cols].head(MAX_METRIC_EVENT_ROWS_PER_CONTEXT).iterrows():
            lines.append("- " + " | ".join([safe_str(r.get(c, "")) for c in me_cols]))

        write_text(context_dir / f"{w}_context.txt", "\n".join(lines))


# =========================
# Plotting
# =========================

def plot_scores(score_df: pd.DataFrame, out_path: Path, title: str):
    df = score_df.sort_values("window_id").copy()

    x = np.arange(len(df))
    y = df["corruption_score"].values

    plt.figure(figsize=(16, 6))
    plt.bar(x, y)

    gt_idx = [i for i, v in enumerate(df["is_corrupted"].astype(int).tolist()) if v == 1]
    gt_y = [y[i] for i in gt_idx]
    plt.scatter(gt_idx, gt_y, marker="x", s=100)

    plt.axhline(GAMMA, linestyle="--", linewidth=1)
    plt.xticks(x, df["window_id"].tolist(), rotation=45)
    plt.xlabel("Window ID")
    plt.ylabel("Corruption score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


# =========================
# Main
# =========================

def compact_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["clue_unsupported_flows", "clue_unsupported_metric_clues"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else safe_str(x))
    return out


def main():
    args = parse_args()

    global DATA_ROOT, OUT_DIR, GAMMA, SEED, EPOCHS, DEVICE
    global SCORE_MODE, HYBRID_TRANSFORMER_WEIGHT, HYBRID_CLUE_WEIGHT

    DATA_ROOT = args.data_root
    OUT_DIR = args.out_dir
    GAMMA = args.gamma
    SEED = args.seed
    EPOCHS = args.epochs
    DEVICE = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    SCORE_MODE = args.score_mode
    HYBRID_TRANSFORMER_WEIGHT = bounded01(args.hybrid_transformer_weight)
    HYBRID_CLUE_WEIGHT = 1.0 - HYBRID_TRANSFORMER_WEIGHT

    set_seed(SEED)
    ensure_dir(OUT_DIR)

    print("=== Rinnegan Stage-1: Cross-modal corrupted-window discovery ===")
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"OUT_DIR   = {OUT_DIR}")
    print(f"DEVICE    = {DEVICE}")
    print(f"GAMMA     = {GAMMA}")
    print(f"SCORE_MODE = {SCORE_MODE}")
    if SCORE_MODE == "hybrid":
        print(f"HYBRID    = {HYBRID_TRANSFORMER_WEIGHT:.2f} * transformer + {HYBRID_CLUE_WEIGHT:.2f} * clue")
    print("")

    data = load_dataset()

    samples = build_samples(data)
    train_samples, val_samples, train_windows, val_windows = split_samples_by_window(samples)

    normalizer = FeatureNormalizer()
    normalizer.fit([s["features"] for s in train_samples])

    train_x, train_y, train_groups = samples_to_tensors(train_samples, normalizer)
    val_x, val_y, val_groups = samples_to_tensors(val_samples, normalizer)

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Train windows: {', '.join(train_windows)}")
    print(f"Val windows: {', '.join(val_windows)}")
    print("")

    model, history_df = train_model(train_x, train_y, train_groups, val_x, val_y, val_groups)
    history_df.to_csv(OUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    torch.save({
        "model_state_dict": model.state_dict(),
        "normalizer": normalizer.to_dict(),
        "all_features": ALL_FEATURES,
        "feature_modalities": FEATURE_MODALITIES.tolist(),
        "config": {
            "version": "stage1_artifact_transformer_discovery",
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "ff_dim": FF_DIM,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "lr": LR,
            "lambda_cl": LAMBDA_CL,
            "tau": TAU,
            "gamma": GAMMA,
            "k_corruptions": K_CORRUPTIONS,
            "score_mode": SCORE_MODE,
            "hybrid_transformer_weight": HYBRID_TRANSFORMER_WEIGHT,
            "hybrid_clue_weight": HYBRID_CLUE_WEIGHT,
            "device": DEVICE,
        }
    }, OUT_DIR / "cross_modal_transformer_verifier.pt")

    clean_scores = score_windows(model, data["clean_audit"], data, normalizer)
    corrupted_scores = score_windows(model, data["corrupted_audit"], data, normalizer)

    clean_scores = attach_labels(clean_scores, data["window_labels"])
    corrupted_scores = attach_labels(corrupted_scores, data["window_labels"])

    clean_scores = clean_scores.sort_values("corruption_score", ascending=False).reset_index(drop=True)
    corrupted_scores = corrupted_scores.sort_values("corruption_score", ascending=False).reset_index(drop=True)

    compact_for_csv(corrupted_scores).to_csv(OUT_DIR / "corrupted_window_scores.csv", index=False, encoding="utf-8-sig")
    compact_for_csv(clean_scores).to_csv(OUT_DIR / "clean_reference_window_scores.csv", index=False, encoding="utf-8-sig")

    eval_main = evaluate_discovery(corrupted_scores, gamma=GAMMA, score_col="corruption_score")
    eval_transformer = evaluate_discovery(corrupted_scores, gamma=GAMMA, score_col="transformer_corruption_score")
    eval_clue = evaluate_discovery(corrupted_scores, gamma=GAMMA, score_col="clue_score")

    clean_eval_main = evaluate_discovery(clean_scores, gamma=GAMMA, score_col="corruption_score")

    threshold_sweep(corrupted_scores, score_col="corruption_score").to_csv(
        OUT_DIR / "threshold_sweep_corrupted.csv", index=False, encoding="utf-8-sig"
    )
    threshold_sweep(corrupted_scores, score_col="transformer_corruption_score").to_csv(
        OUT_DIR / "threshold_sweep_corrupted_transformer.csv", index=False, encoding="utf-8-sig"
    )
    threshold_sweep(corrupted_scores, score_col="clue_score").to_csv(
        OUT_DIR / "threshold_sweep_corrupted_clue.csv", index=False, encoding="utf-8-sig"
    )
    threshold_sweep(clean_scores, score_col="corruption_score").to_csv(
        OUT_DIR / "threshold_sweep_clean_reference.csv", index=False, encoding="utf-8-sig"
    )

    metrics = {
        "model": "cross_modal_transformer_discovery",
        "score_mode": SCORE_MODE,
        "gamma": GAMMA,
        "hybrid_transformer_weight": HYBRID_TRANSFORMER_WEIGHT,
        "hybrid_clue_weight": HYBRID_CLUE_WEIGHT,
        "train_windows": train_windows,
        "val_windows": val_windows,
        "training_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "corrupted_eval_main": eval_main,
        "corrupted_eval_transformer_only": eval_transformer,
        "corrupted_eval_clue_only": eval_clue,
        "clean_reference_eval_main": clean_eval_main,
    }
    write_json(OUT_DIR / "discovery_metrics.json", metrics)

    export_window_contexts(
        corrupted_scores,
        data["corrupted_audit"],
        data,
        OUT_DIR,
        fallback_top_n=args.context_fallback_top_n,
    )

    plot_scores(
        corrupted_scores,
        OUT_DIR / "corrupted_window_scores.png",
        "Rinnegan Stage-1: corruption scores on corrupted logs",
    )
    plot_scores(
        clean_scores,
        OUT_DIR / "clean_reference_window_scores.png",
        "Rinnegan Stage-1: corruption scores on clean-reference logs",
    )

    lines = []
    lines.append("Rinnegan Stage-1: Cross-modal corrupted-window discovery")
    lines.append("")
    lines.append(f"DATA_ROOT: {DATA_ROOT}")
    lines.append(f"OUT_DIR: {OUT_DIR}")
    lines.append(f"DEVICE: {DEVICE}")
    lines.append(f"GAMMA: {GAMMA}")
    lines.append(f"K_CORRUPTIONS: {K_CORRUPTIONS}")
    lines.append(f"SCORE_MODE: {SCORE_MODE}")
    if SCORE_MODE == "hybrid":
        lines.append(f"HYBRID_SCORE: {HYBRID_TRANSFORMER_WEIGHT:.2f} * transformer_corruption_score + {HYBRID_CLUE_WEIGHT:.2f} * clue_score")
    lines.append("")
    lines.append("Training setup")
    lines.append(f"  train_windows: {', '.join(train_windows)}")
    lines.append(f"  val_windows: {', '.join(val_windows)}")
    lines.append(f"  training_samples: {len(train_samples)}")
    lines.append(f"  validation_samples: {len(val_samples)}")
    if len(history_df):
        last = history_df.iloc[-1].to_dict()
        lines.append(f"  best_epoch: {safe_int(last.get('best_epoch'))}")
        lines.append(f"  best_val_f1: {safe_float(last.get('best_val_f1')):.4f}")
    lines.append("")

    lines.append("Gamma-threshold protocol on corrupted logs")
    for k, v in eval_main["gamma_metrics"].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  predicted windows: {', '.join(eval_main['gamma_windows'])}")
    lines.append("")

    lines.append("Diagnostic Top-K protocol on corrupted logs")
    lines.append(f"  K: {eval_main['topk_k']}")
    for k, v in eval_main["topk_metrics"].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  predicted windows: {', '.join(eval_main['topk_windows'])}")
    lines.append("")

    lines.append("Transformer-only Top-K protocol on corrupted logs")
    for k, v in eval_transformer["topk_metrics"].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  predicted windows: {', '.join(eval_transformer['topk_windows'])}")
    lines.append("")

    lines.append("Clue-only Top-K protocol on corrupted logs")
    for k, v in eval_clue["topk_metrics"].items():
        lines.append(f"  {k}: {v}")
    lines.append(f"  predicted windows: {', '.join(eval_clue['topk_windows'])}")
    lines.append("")

    lines.append("Diagnostic threshold sweep on corrupted logs")
    best = eval_main["diagnostic_best_threshold_from_eval_sweep"]
    lines.append(
        f"  threshold={best.get('threshold')} | "
        f"P={best.get('precision')} R={best.get('recall')} F1={best.get('f1')}"
    )
    lines.append("")

    lines.append("Top suspicious windows")
    top_cols = [
        "window_id",
        "consistency_probability",
        "transformer_corruption_score",
        "clue_score",
        "hybrid_corruption_score",
        "corruption_score",
        "score_mode",
        "log_side_similarity",
        "is_corrupted",
        "removed_event_count",
        "attack_stages",
        "unsupported_flow_count",
        "unsupported_external_flow_count",
        "unsupported_metric_clue_count",
        "unsupported_metric_zsum",
        "clue_external_flow_log_gap",
    ]
    top_cols = [c for c in top_cols if c in corrupted_scores.columns]

    for _, r in corrupted_scores[top_cols].head(12).iterrows():
        lines.append(
            f"  {safe_str(r.get('window_id'))} | "
            f"p={safe_float(r.get('consistency_probability')):.4f} | "
            f"tr_score={safe_float(r.get('transformer_corruption_score')):.4f} | "
            f"clue={safe_float(r.get('clue_score')):.4f} | "
            f"score={safe_float(r.get('corruption_score')):.4f} | "
            f"sim={safe_float(r.get('log_side_similarity')):.4f} | "
            f"gt={safe_int(r.get('is_corrupted'))} | "
            f"removed={safe_int(r.get('removed_event_count'))} | "
            f"stage={safe_str(r.get('attack_stages'))} | "
            f"flow_gap={safe_int(r.get('unsupported_flow_count'))} | "
            f"ext_gap={safe_int(r.get('unsupported_external_flow_count'))} | "
            f"metric_gap={safe_int(r.get('unsupported_metric_clue_count'))} | "
            f"metric_zsum={safe_float(r.get('unsupported_metric_zsum')):.2f} | "
            f"ext_flow_log_gap={safe_float(r.get('clue_external_flow_log_gap')):.2f}"
        )

    lines.append("")
    lines.append("Outputs")
    lines.append(f"  {OUT_DIR / 'cross_modal_transformer_verifier.pt'}")
    lines.append(f"  {OUT_DIR / 'training_history.csv'}")
    lines.append(f"  {OUT_DIR / 'corrupted_window_scores.csv'}")
    lines.append(f"  {OUT_DIR / 'clean_reference_window_scores.csv'}")
    lines.append(f"  {OUT_DIR / 'threshold_sweep_corrupted.csv'}")
    lines.append(f"  {OUT_DIR / 'threshold_sweep_corrupted_transformer.csv'}")
    lines.append(f"  {OUT_DIR / 'threshold_sweep_corrupted_clue.csv'}")
    lines.append(f"  {OUT_DIR / 'window_contexts'}")

    summary_text = "\n".join(lines)
    write_text(OUT_DIR / "discovery_summary.txt", summary_text)

    print("")
    print(summary_text)
    print("")
    print("[DONE] Stage-1 corrupted-window discovery finished.")


if __name__ == "__main__":
    main()