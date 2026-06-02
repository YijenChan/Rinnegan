# -*- coding: utf-8 -*-
"""
Rinnegan Stage-3: Reliability-aware graph initialization.

This script builds three provenance graphs:
1. clean graph
2. corrupted graph
3. remediated graph

For the remediated graph:
- observed audit events are converted into canonical provenance edges with reliability=1.0;
- recovered event hypotheses are instantiated as hypothesis-specific edges;
- recovered endpoint copies inherit R(hat e);
- copy-to-canonical mapping is exported for later node-level score merging.

Default usage from the repository root:
    python scripts/run_stage3_graph_initialization.py \
        --data-root OfficeFog_desensitized_version/full-dataset \
        --stage2-out outputs/stage2_remediation \
        --out-dir outputs/stage3_graph_init

Dependencies:
    pip install pandas networkx
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List

import pandas as pd
import networkx as nx


# =========================
# Config
# =========================

DATA_ROOT = Path("OfficeFog_desensitized_version/full-dataset")
STAGE2_OUT = Path("outputs/stage2_remediation")
OUT_DIR = Path("outputs/stage3_graph_init")

ALLOWED_RELATIONS = [
    "read",
    "write",
    "execute",
    "spawn",
    "connect",
    "accept",
    "chmod",
    "unlink",
    "rename",
]

NODE_TYPES = ["process", "file", "socket", "unknown"]

NODE_COLUMNS = [
    "node_id",
    "canonical_id",
    "raw_entity",
    "entity_type",
    "host",
    "source",
    "is_recovered_copy",
    "hypothesis_id",
    "reliability",
    "semantic_text",
    "is_malicious",
    "label_stage",
    "label_description",
    "type_process",
    "type_file",
    "type_socket",
    "type_unknown",
    "src_observed",
    "src_recovered_copy",
    "in_degree",
    "out_degree",
    "total_degree",
]

EDGE_COLUMNS = [
    "edge_id",
    "src_node",
    "dst_node",
    "src_canonical",
    "dst_canonical",
    "src_entity",
    "src_type",
    "dst_entity",
    "dst_type",
    "event_id",
    "timestamp",
    "window_id",
    "host",
    "user",
    "relation",
    "command",
    "args",
    "source",
    "hypothesis_id",
    "reliability",
    "omega_e",
    "R_cross",
    "R_self",
    "is_recovered_edge",
    "is_attack_edge",
    "matched_removed_event",
    "matched_removed_event_id",
    "label",
    "stage",
]

COPY_MAPPING_COLUMNS = [
    "copy_node_id",
    "canonical_node_id",
    "canonical_entity",
    "endpoint_role",
    "hypothesis_id",
    "event_id",
    "reliability",
]


# =========================
# Arguments
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rinnegan Stage-3 reliability-aware provenance graph initialization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--stage2-out", type=Path, default=STAGE2_OUT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)

    parser.add_argument("--clean-audit", type=Path, default=None)
    parser.add_argument("--corrupted-audit", type=Path, default=None)
    parser.add_argument("--remediated-audit", type=Path, default=None)
    parser.add_argument("--recovered-scored", type=Path, default=None)

    parser.add_argument(
        "--export-graph-formats",
        action="store_true",
        help="Export GraphML and GEXF files in addition to CSV tables.",
    )
    parser.add_argument(
        "--no-diagnostic-labels",
        action="store_true",
        help="Do not attach diagnostic attack labels to exported nodes/edges.",
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


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return read_csv(path)
    return pd.DataFrame()


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


def safe_float(x, default: float = 1.0) -> float:
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


def normalize_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        df["timestamp"] = pd.NaT

    if "event_id" not in df.columns:
        df["event_id"] = [f"event_{i:06d}" for i in range(len(df))]

    df = df.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    return df


# =========================
# Canonicalization
# =========================

def infer_entity_type(entity: str, default: str = "") -> str:
    entity = safe_str(entity).strip()

    if re.match(r"^\d+\.\d+\.\d+\.\d+:\d+$", entity):
        return "socket"

    if entity.startswith("/") or entity.startswith("./"):
        return "file"

    if default in {"process", "file", "socket"}:
        return default

    return "unknown"


def canonical_process_name(entity: str) -> str:
    """
    Examples:
        firefox:2412@host-1 -> firefox
        firefox@host-1      -> firefox
        firefox             -> firefox
    """
    e = safe_str(entity).strip().lower()

    if "@" in e:
        e = e.split("@", 1)[0]

    if ":" in e:
        e = e.split(":", 1)[0]

    return e


def canonical_entity(entity: str, entity_type: str, host: str) -> str:
    entity = safe_str(entity).strip()
    entity_type = infer_entity_type(entity, entity_type)
    host = safe_str(host).strip()

    if entity_type == "socket":
        return entity.lower()

    if entity_type == "file":
        return entity

    if entity_type == "process":
        proc_name = canonical_process_name(entity)
        if host:
            return f"{proc_name}@{host}"
        return proc_name

    return entity


def semantic_text(entity: str, entity_type: str, host: str) -> str:
    entity = safe_str(entity)
    entity_type = safe_str(entity_type)
    host = safe_str(host)

    if entity_type == "process":
        return canonical_process_name(entity)

    if entity_type == "file":
        parts = [p for p in entity.split("/") if p]
        return " ".join(parts[-4:]) if parts else entity

    if entity_type == "socket":
        return entity

    return f"{entity_type} {entity} {host}".strip()


def node_type_onehot(entity_type: str) -> Dict[str, int]:
    entity_type = entity_type if entity_type in NODE_TYPES else "unknown"
    return {f"type_{t}": int(t == entity_type) for t in NODE_TYPES}


def source_onehot(source: str) -> Dict[str, int]:
    source = safe_str(source)
    return {
        "src_observed": int(source == "observed"),
        "src_recovered_copy": int(source == "recovered_copy"),
    }


# =========================
# Label maps for diagnostics
# =========================

def load_entity_label_map(entity_labels_path: Path, include_labels: bool) -> Dict[str, Dict[str, Any]]:
    if not include_labels or not entity_labels_path.exists():
        return {}

    df = read_csv(entity_labels_path)
    label_map = {}

    for _, r in df.iterrows():
        raw_name = safe_str(r.get("canonical_name"))
        raw_type = safe_str(r.get("entity_type"))
        host = safe_str(r.get("host"))

        keys = set()
        if raw_name:
            keys.add(raw_name)
            keys.add(canonical_entity(raw_name, raw_type, host))

        for k in keys:
            label_map[k] = {
                "is_malicious": safe_int(r.get("is_malicious", 0)),
                "stage": safe_str(r.get("stage")),
                "description": safe_str(r.get("description")),
            }

    return label_map


def load_event_label_map(event_labels_path: Path, include_labels: bool) -> Dict[str, Dict[str, Any]]:
    if not include_labels or not event_labels_path.exists():
        return {}

    df = read_csv(event_labels_path)
    m = {}

    for _, r in df.iterrows():
        eid = safe_str(r.get("event_id"))
        if not eid:
            continue

        m[eid] = {
            "is_attack_event": safe_int(r.get("is_attack_event", 0)),
            "stage": safe_str(r.get("stage")),
            "is_removed_in_corrupted": safe_int(r.get("is_removed_in_corrupted", 0)),
            "attack_step_id": safe_str(r.get("attack_step_id")),
            "description": safe_str(r.get("description")),
        }

    return m


def load_recovered_match_map(recovered_scored_path: Path, include_labels: bool) -> Dict[str, Dict[str, Any]]:
    """
    recovered_events_scored.csv may provide hypothesis-level ground-truth matching.
    It is attached only as a diagnostic label and is not used to construct graph topology.
    """
    if not include_labels or not recovered_scored_path.exists():
        return {}

    df = read_csv(recovered_scored_path)
    m = {}

    for _, r in df.iterrows():
        hid = safe_str(r.get("hypothesis_id"))
        if not hid:
            continue

        m[hid] = {
            "candidate_id": safe_str(r.get("candidate_id")),
            "matched_removed_event": safe_int(r.get("matched_removed_event", 0)),
            "matched_removed_event_id": safe_str(r.get("matched_removed_event_id")),
            "matched_original_event_id": safe_str(r.get("matched_original_event_id")),
            "matched_removed_reason": safe_str(r.get("matched_removed_reason")),
            "reliability": safe_float(r.get("reliability"), 1.0),
            "R_cross": safe_float(r.get("R_cross"), 1.0),
            "R_self": safe_float(r.get("R_self"), 1.0),
        }

    return m


# =========================
# Graph construction
# =========================

def make_observed_node_id(entity: str, entity_type: str, host: str) -> str:
    c = canonical_entity(entity, entity_type, host)
    return f"obs::{c}"


def make_recovered_copy_node_id(canonical: str, hypothesis_id: str, endpoint: str) -> str:
    return f"rec::{hypothesis_id}::{endpoint}::{canonical}"


def add_node_if_needed(
    G: nx.MultiDiGraph,
    node_id: str,
    canonical_id: str,
    raw_entity: str,
    entity_type: str,
    host: str,
    source: str,
    reliability: float,
    hypothesis_id: str,
    entity_label_map: Dict[str, Dict[str, Any]],
):
    if node_id in G:
        G.nodes[node_id]["reliability"] = max(float(G.nodes[node_id].get("reliability", 0.0)), reliability)
        return

    entity_type = entity_type if entity_type in {"process", "file", "socket"} else infer_entity_type(raw_entity, entity_type)
    label = entity_label_map.get(canonical_id, entity_label_map.get(raw_entity, {}))

    attrs = {
        "node_id": node_id,
        "canonical_id": canonical_id,
        "raw_entity": raw_entity,
        "entity_type": entity_type,
        "host": host,
        "source": source,
        "is_recovered_copy": int(source == "recovered_copy"),
        "hypothesis_id": hypothesis_id,
        "reliability": float(reliability),
        "semantic_text": semantic_text(raw_entity, entity_type, host),
        "is_malicious": safe_int(label.get("is_malicious", 0)),
        "label_stage": safe_str(label.get("stage", "")),
        "label_description": safe_str(label.get("description", "")),
    }

    attrs.update(node_type_onehot(entity_type))
    attrs.update(source_onehot(source))

    G.add_node(node_id, **attrs)


def parse_row_source(row: pd.Series) -> str:
    src = safe_str(row.get("source")).lower()
    label = safe_str(row.get("label")).lower()

    recovered_sources = {
        "llm_recovered",
        "mock_recovered",
        "cached_recovered",
        "openai_recovered",
        "recovered",
    }

    if src in recovered_sources or src.endswith("_recovered") or label == "recovered":
        return "recovered"

    return "observed"


def build_graph_from_audit(
    audit_df: pd.DataFrame,
    graph_name: str,
    entity_label_map: Dict[str, Dict[str, Any]],
    event_label_map: Dict[str, Dict[str, Any]],
    recovered_match_map: Dict[str, Dict[str, Any]],
    use_recovered_copies: bool,
) -> Tuple[nx.MultiDiGraph, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_df = normalize_timestamp_col(audit_df)

    G = nx.MultiDiGraph(name=graph_name)
    edge_rows = []
    copy_mapping_rows = []

    for row_idx, r in audit_df.iterrows():
        event_id = safe_str(r.get("event_id")) or f"event_{row_idx:06d}"
        timestamp = safe_str(r.get("timestamp"))
        window_id = safe_str(r.get("window_id"))
        host = safe_str(r.get("host"))
        user = safe_str(r.get("user"))

        src_entity_raw = safe_str(r.get("src_entity"))
        dst_entity_raw = safe_str(r.get("dst_entity"))
        src_type = infer_entity_type(src_entity_raw, safe_str(r.get("src_type")))
        dst_type = infer_entity_type(dst_entity_raw, safe_str(r.get("dst_type")))
        relation = safe_str(r.get("relation"))

        if relation not in ALLOWED_RELATIONS:
            continue

        source_type = parse_row_source(r)

        hypothesis_id = safe_str(r.get("hypothesis_id"))
        candidate_id = safe_str(r.get("candidate_id"))

        reliability = safe_float(r.get("reliability"), 1.0)
        R_cross = safe_float(r.get("R_cross"), reliability)
        R_self = safe_float(r.get("R_self"), 1.0)

        canonical_src = canonical_entity(src_entity_raw, src_type, host)
        canonical_dst = canonical_entity(dst_entity_raw, dst_type, host)

        if source_type == "recovered" and use_recovered_copies:
            if not hypothesis_id:
                hypothesis_id = event_id

            src_node = make_recovered_copy_node_id(canonical_src, hypothesis_id, "src")
            dst_node = make_recovered_copy_node_id(canonical_dst, hypothesis_id, "dst")

            node_source = "recovered_copy"
            edge_source = "recovered"

            copy_mapping_rows.append({
                "copy_node_id": src_node,
                "canonical_node_id": f"obs::{canonical_src}",
                "canonical_entity": canonical_src,
                "endpoint_role": "src",
                "hypothesis_id": hypothesis_id,
                "event_id": event_id,
                "reliability": reliability,
            })
            copy_mapping_rows.append({
                "copy_node_id": dst_node,
                "canonical_node_id": f"obs::{canonical_dst}",
                "canonical_entity": canonical_dst,
                "endpoint_role": "dst",
                "hypothesis_id": hypothesis_id,
                "event_id": event_id,
                "reliability": reliability,
            })

        else:
            src_node = make_observed_node_id(src_entity_raw, src_type, host)
            dst_node = make_observed_node_id(dst_entity_raw, dst_type, host)

            node_source = "observed"
            edge_source = "observed"
            reliability = 1.0
            R_cross = 1.0
            R_self = 1.0
            hypothesis_id = ""

        add_node_if_needed(
            G=G,
            node_id=src_node,
            canonical_id=canonical_src,
            raw_entity=src_entity_raw,
            entity_type=src_type,
            host=host,
            source=node_source,
            reliability=reliability,
            hypothesis_id=hypothesis_id,
            entity_label_map=entity_label_map,
        )

        add_node_if_needed(
            G=G,
            node_id=dst_node,
            canonical_id=canonical_dst,
            raw_entity=dst_entity_raw,
            entity_type=dst_type,
            host=host,
            source=node_source,
            reliability=reliability,
            hypothesis_id=hypothesis_id,
            entity_label_map=entity_label_map,
        )

        event_label = event_label_map.get(event_id, {})
        matched_info = recovered_match_map.get(hypothesis_id, {}) if hypothesis_id else {}

        if edge_source == "observed":
            is_attack_edge = safe_int(event_label.get("is_attack_event", 0))
            matched_removed_event = 0
            matched_removed_event_id = ""
        else:
            is_attack_edge = safe_int(matched_info.get("matched_removed_event", 0))
            matched_removed_event = safe_int(matched_info.get("matched_removed_event", 0))
            matched_removed_event_id = safe_str(matched_info.get("matched_removed_event_id", ""))

        edge_key = event_id if event_id else f"edge_{row_idx}"

        edge_attrs = {
            "event_id": event_id,
            "timestamp": timestamp,
            "window_id": window_id,
            "host": host,
            "user": user,
            "relation": relation,
            "command": safe_str(r.get("command")),
            "args": safe_str(r.get("args")),
            "source": edge_source,
            "hypothesis_id": hypothesis_id,
            "candidate_id": candidate_id,
            "reliability": float(reliability),
            "omega_e": float(reliability),
            "R_cross": float(R_cross),
            "R_self": float(R_self),
            "is_recovered_edge": int(edge_source == "recovered"),
            "is_attack_edge": int(is_attack_edge),
            "matched_removed_event": int(matched_removed_event),
            "matched_removed_event_id": matched_removed_event_id,
            "label": safe_str(r.get("label")),
            "stage": safe_str(r.get("stage")),
        }

        G.add_edge(src_node, dst_node, key=edge_key, **edge_attrs)

        edge_row = {
            "edge_id": edge_key,
            "src_node": src_node,
            "dst_node": dst_node,
            "src_canonical": canonical_src,
            "dst_canonical": canonical_dst,
            "src_entity": src_entity_raw,
            "src_type": src_type,
            "dst_entity": dst_entity_raw,
            "dst_type": dst_type,
        }
        edge_row.update(edge_attrs)
        edge_rows.append(edge_row)

    nodes_df = graph_nodes_to_df(G)
    edges_df = pd.DataFrame(edge_rows)
    copy_mapping_df = pd.DataFrame(copy_mapping_rows)

    if len(nodes_df) == 0:
        nodes_df = pd.DataFrame(columns=NODE_COLUMNS)
    if len(edges_df) == 0:
        edges_df = pd.DataFrame(columns=EDGE_COLUMNS)
    if len(copy_mapping_df) == 0:
        copy_mapping_df = pd.DataFrame(columns=COPY_MAPPING_COLUMNS)

    return G, nodes_df, edges_df, copy_mapping_df


def graph_nodes_to_df(G: nx.MultiDiGraph) -> pd.DataFrame:
    rows = []
    for node_id, d in G.nodes(data=True):
        row = dict(d)
        row["node_id"] = node_id
        row["in_degree"] = G.in_degree(node_id)
        row["out_degree"] = G.out_degree(node_id)
        row["total_degree"] = G.degree(node_id)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=NODE_COLUMNS)

    df = pd.DataFrame(rows)

    sort_cols = [c for c in ["is_recovered_copy", "is_malicious", "total_degree"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    return df


# =========================
# Feature table and export
# =========================

def export_node_features(nodes_df: pd.DataFrame, out_path: Path):
    """
    This table is the graph-initialization feature seed.
    Stage-4/5 can replace semantic_text with learned embeddings.
    Diagnostic labels are exported separately in nodes.csv and should not be used
    as input features during training.
    """
    if len(nodes_df) == 0:
        pd.DataFrame(columns=[
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
        ]).to_csv(out_path, index=False, encoding="utf-8-sig")
        return

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

    cols = [c for c in keep_cols if c in nodes_df.columns]
    nodes_df[cols].to_csv(out_path, index=False, encoding="utf-8-sig")


def sanitize_attrs(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)):
            out[k] = v
        elif v is None:
            out[k] = ""
        else:
            out[k] = safe_str(v)
    return out


def export_graph_formats(G: nx.MultiDiGraph, out_dir: Path):
    H = nx.MultiDiGraph()

    for n, d in G.nodes(data=True):
        H.add_node(n, **sanitize_attrs(d))

    for u, v, k, d in G.edges(keys=True, data=True):
        H.add_edge(u, v, key=k, **sanitize_attrs(d))

    nx.write_graphml(H, out_dir / "provenance_graph.graphml")
    nx.write_gexf(H, out_dir / "provenance_graph.gexf")


def compute_graph_stats(G: nx.MultiDiGraph, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, graph_name: str) -> Dict[str, Any]:
    node_type_counts = nodes_df["entity_type"].value_counts().to_dict() if len(nodes_df) and "entity_type" in nodes_df.columns else {}
    node_source_counts = nodes_df["source"].value_counts().to_dict() if len(nodes_df) and "source" in nodes_df.columns else {}

    relation_counts = edges_df["relation"].value_counts().to_dict() if len(edges_df) and "relation" in edges_df.columns else {}
    edge_source_counts = edges_df["source"].value_counts().to_dict() if len(edges_df) and "source" in edges_df.columns else {}

    UG = nx.Graph()
    UG.add_nodes_from(G.nodes())
    UG.add_edges_from([(u, v) for u, v in G.edges()])
    comp_sizes = sorted([len(c) for c in nx.connected_components(UG)], reverse=True) if G.number_of_nodes() else []

    stats = {
        "graph_name": graph_name,
        "num_nodes": int(G.number_of_nodes()),
        "num_edges": int(G.number_of_edges()),
        "num_observed_nodes": int((nodes_df["source"] == "observed").sum()) if len(nodes_df) and "source" in nodes_df.columns else 0,
        "num_recovered_copy_nodes": int((nodes_df["source"] == "recovered_copy").sum()) if len(nodes_df) and "source" in nodes_df.columns else 0,
        "num_observed_edges": int((edges_df["source"] == "observed").sum()) if len(edges_df) and "source" in edges_df.columns else 0,
        "num_recovered_edges": int((edges_df["source"] == "recovered").sum()) if len(edges_df) and "source" in edges_df.columns else 0,
        "num_attack_edges_diagnostic": int(edges_df["is_attack_edge"].sum()) if len(edges_df) and "is_attack_edge" in edges_df.columns else 0,
        "num_malicious_nodes_diagnostic": int(nodes_df["is_malicious"].sum()) if len(nodes_df) and "is_malicious" in nodes_df.columns else 0,
        "node_type_counts": {str(k): int(v) for k, v in node_type_counts.items()},
        "node_source_counts": {str(k): int(v) for k, v in node_source_counts.items()},
        "relation_counts": {str(k): int(v) for k, v in relation_counts.items()},
        "edge_source_counts": {str(k): int(v) for k, v in edge_source_counts.items()},
        "connected_components_undirected": int(len(comp_sizes)),
        "largest_component_size": int(comp_sizes[0]) if comp_sizes else 0,
    }

    if "reliability" in edges_df.columns and len(edges_df):
        rec = edges_df[edges_df["source"] == "recovered"]
        if len(rec):
            stats["recovered_edge_reliability_min"] = float(rec["reliability"].min())
            stats["recovered_edge_reliability_mean"] = float(rec["reliability"].mean())
            stats["recovered_edge_reliability_max"] = float(rec["reliability"].max())
        else:
            stats["recovered_edge_reliability_min"] = None
            stats["recovered_edge_reliability_mean"] = None
            stats["recovered_edge_reliability_max"] = None

    return stats


def export_one_graph(
    graph_name: str,
    audit_path: Path,
    out_dir: Path,
    entity_label_map: Dict[str, Dict[str, Any]],
    event_label_map: Dict[str, Dict[str, Any]],
    recovered_match_map: Dict[str, Dict[str, Any]],
    use_recovered_copies: bool,
    export_graph_files: bool,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    audit_df = read_csv(audit_path)

    G, nodes_df, edges_df, copy_mapping_df = build_graph_from_audit(
        audit_df=audit_df,
        graph_name=graph_name,
        entity_label_map=entity_label_map,
        event_label_map=event_label_map,
        recovered_match_map=recovered_match_map,
        use_recovered_copies=use_recovered_copies,
    )

    nodes_df.to_csv(out_dir / "nodes.csv", index=False, encoding="utf-8-sig")
    edges_df.to_csv(out_dir / "edges.csv", index=False, encoding="utf-8-sig")
    copy_mapping_df.to_csv(out_dir / "copy_mapping.csv", index=False, encoding="utf-8-sig")

    export_node_features(nodes_df, out_dir / "node_features.csv")

    if export_graph_files:
        export_graph_formats(G, out_dir)

    stats = compute_graph_stats(G, nodes_df, edges_df, graph_name)
    write_json(out_dir / "graph_stats.json", stats)

    stats_lines = []
    stats_lines.append(f"Graph: {graph_name}")
    stats_lines.append(f"Nodes: {stats['num_nodes']}")
    stats_lines.append(f"Edges: {stats['num_edges']}")
    stats_lines.append(f"Observed nodes: {stats['num_observed_nodes']}")
    stats_lines.append(f"Recovered copy nodes: {stats['num_recovered_copy_nodes']}")
    stats_lines.append(f"Observed edges: {stats['num_observed_edges']}")
    stats_lines.append(f"Recovered edges: {stats['num_recovered_edges']}")
    stats_lines.append(f"Attack edges diagnostic: {stats['num_attack_edges_diagnostic']}")
    stats_lines.append(f"Malicious nodes diagnostic: {stats['num_malicious_nodes_diagnostic']}")
    stats_lines.append(f"Connected components: {stats['connected_components_undirected']}")
    stats_lines.append(f"Largest component size: {stats['largest_component_size']}")
    stats_lines.append("")
    stats_lines.append("Node type counts:")
    for k, v in stats["node_type_counts"].items():
        stats_lines.append(f"  {k}: {v}")
    stats_lines.append("")
    stats_lines.append("Edge source counts:")
    for k, v in stats["edge_source_counts"].items():
        stats_lines.append(f"  {k}: {v}")
    stats_lines.append("")
    stats_lines.append("Relation counts:")
    for k, v in stats["relation_counts"].items():
        stats_lines.append(f"  {k}: {v}")

    write_text(out_dir / "graph_stats.txt", "\n".join(stats_lines))

    return stats


# =========================
# Cross-graph comparison
# =========================

def compare_graphs(
    stats_clean: Dict[str, Any],
    stats_corrupted: Dict[str, Any],
    stats_remediated: Dict[str, Any],
    out_dir: Path,
):
    summary = {
        "clean": stats_clean,
        "corrupted": stats_corrupted,
        "remediated": stats_remediated,
        "delta": {
            "clean_minus_corrupted_edges": stats_clean["num_edges"] - stats_corrupted["num_edges"],
            "remediated_minus_corrupted_edges": stats_remediated["num_edges"] - stats_corrupted["num_edges"],
            "remediated_minus_clean_edges": stats_remediated["num_edges"] - stats_clean["num_edges"],
            "clean_minus_corrupted_attack_edges_diagnostic": stats_clean["num_attack_edges_diagnostic"] - stats_corrupted["num_attack_edges_diagnostic"],
            "remediated_minus_corrupted_attack_edges_diagnostic": stats_remediated["num_attack_edges_diagnostic"] - stats_corrupted["num_attack_edges_diagnostic"],
            "remediated_minus_clean_attack_edges_diagnostic": stats_remediated["num_attack_edges_diagnostic"] - stats_clean["num_attack_edges_diagnostic"],
            "remediated_recovered_edges": stats_remediated["num_recovered_edges"],
            "remediated_recovered_copy_nodes": stats_remediated["num_recovered_copy_nodes"],
        }
    }

    write_json(out_dir / "graph_init_summary.json", summary)

    lines = []
    lines.append("Rinnegan Stage-3: Graph Initialization Summary")
    lines.append("")
    lines.append("Graph sizes")
    lines.append(f"  clean nodes/edges:      {stats_clean['num_nodes']} / {stats_clean['num_edges']}")
    lines.append(f"  corrupted nodes/edges:  {stats_corrupted['num_nodes']} / {stats_corrupted['num_edges']}")
    lines.append(f"  remediated nodes/edges: {stats_remediated['num_nodes']} / {stats_remediated['num_edges']}")
    lines.append("")
    lines.append("Observed/recovered edges")
    lines.append(f"  clean observed edges:      {stats_clean['num_observed_edges']}")
    lines.append(f"  corrupted observed edges:  {stats_corrupted['num_observed_edges']}")
    lines.append(f"  remediated observed edges: {stats_remediated['num_observed_edges']}")
    lines.append(f"  remediated recovered edges: {stats_remediated['num_recovered_edges']}")
    lines.append("")
    lines.append("Diagnostic attack labels")
    lines.append(f"  clean attack edges:      {stats_clean['num_attack_edges_diagnostic']}")
    lines.append(f"  corrupted attack edges:  {stats_corrupted['num_attack_edges_diagnostic']}")
    lines.append(f"  remediated attack edges: {stats_remediated['num_attack_edges_diagnostic']}")
    lines.append("")
    lines.append("Deltas")
    for k, v in summary["delta"].items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Output folders")
    lines.append(f"  {out_dir / 'clean_graph'}")
    lines.append(f"  {out_dir / 'corrupted_graph'}")
    lines.append(f"  {out_dir / 'remediated_graph'}")

    write_text(out_dir / "graph_init_summary.txt", "\n".join(lines))
    print("\n".join(lines))


# =========================
# Main
# =========================

def main():
    args = parse_args()

    data_root = args.data_root
    stage2_out = args.stage2_out
    out_dir = args.out_dir

    clean_audit = args.clean_audit or (data_root / "clean" / "audit_log.csv")
    corrupted_audit = args.corrupted_audit or (data_root / "corrupted" / "audit_log.csv")
    remediated_audit = args.remediated_audit or (stage2_out / "remediated_audit_log.csv")
    recovered_scored = args.recovered_scored or (stage2_out / "recovered_events_scored.csv")

    entity_labels = data_root / "labels" / "entity_labels.csv"
    event_labels = data_root / "labels" / "event_labels.csv"

    include_labels = not args.no_diagnostic_labels

    ensure_dir(out_dir)

    print("=== Rinnegan Stage-3: Reliability-aware graph initialization ===")
    print(f"DATA_ROOT = {data_root}")
    print(f"STAGE2_OUT = {stage2_out}")
    print(f"OUT_DIR = {out_dir}")
    print(f"CLEAN_AUDIT = {clean_audit}")
    print(f"CORRUPTED_AUDIT = {corrupted_audit}")
    print(f"REMEDIATED_AUDIT = {remediated_audit}")
    print(f"RECOVERED_SCORED = {recovered_scored}")
    print(f"DIAGNOSTIC_LABELS = {include_labels}")
    print(f"EXPORT_GRAPH_FORMATS = {args.export_graph_formats}")
    print("")

    if not clean_audit.exists():
        raise FileNotFoundError(f"Missing clean audit log: {clean_audit}")
    if not corrupted_audit.exists():
        raise FileNotFoundError(f"Missing corrupted audit log: {corrupted_audit}")
    if not remediated_audit.exists():
        raise FileNotFoundError(
            f"Missing remediated audit log: {remediated_audit}. "
            "Run Stage-2 first or pass --remediated-audit."
        )

    entity_label_map = load_entity_label_map(entity_labels, include_labels=include_labels)
    event_label_map = load_event_label_map(event_labels, include_labels=include_labels)
    recovered_match_map = load_recovered_match_map(recovered_scored, include_labels=include_labels)

    write_json(out_dir / "relation_vocab.json", {r: i for i, r in enumerate(ALLOWED_RELATIONS)})
    write_json(out_dir / "node_type_vocab.json", {t: i for i, t in enumerate(NODE_TYPES)})

    stats_clean = export_one_graph(
        graph_name="clean_graph",
        audit_path=clean_audit,
        out_dir=out_dir / "clean_graph",
        entity_label_map=entity_label_map,
        event_label_map=event_label_map,
        recovered_match_map=recovered_match_map,
        use_recovered_copies=False,
        export_graph_files=args.export_graph_formats,
    )

    stats_corrupted = export_one_graph(
        graph_name="corrupted_graph",
        audit_path=corrupted_audit,
        out_dir=out_dir / "corrupted_graph",
        entity_label_map=entity_label_map,
        event_label_map=event_label_map,
        recovered_match_map=recovered_match_map,
        use_recovered_copies=False,
        export_graph_files=args.export_graph_formats,
    )

    stats_remediated = export_one_graph(
        graph_name="remediated_graph",
        audit_path=remediated_audit,
        out_dir=out_dir / "remediated_graph",
        entity_label_map=entity_label_map,
        event_label_map=event_label_map,
        recovered_match_map=recovered_match_map,
        use_recovered_copies=True,
        export_graph_files=args.export_graph_formats,
    )

    compare_graphs(stats_clean, stats_corrupted, stats_remediated, out_dir=out_dir)

    print("")
    print("[DONE] Stage-3 graph initialization completed.")


if __name__ == "__main__":
    main()