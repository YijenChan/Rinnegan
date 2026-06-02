# -*- coding: utf-8 -*-
"""
Rinnegan Stage-5: Train and evaluate Reliability-aware Graph Autoencoder (R-GAE).

This script trains R-GAE on benign-only provenance graphs, calibrates the
anomaly threshold on validation benign graphs, and evaluates node-level
detection on clean, corrupted, and remediated test graphs.

Default usage from the repository root:
    python scripts/run_stage5_train_rgae.py \
        --stage4-root outputs/stage4_rgae_data \
        --out-dir outputs/stage5_rgae_detection \
        --device cpu

Dependencies:
    pip install pandas numpy torch
"""

import argparse
import json
import random
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Config
# =========================

STAGE4_ROOT = Path("outputs/stage4_rgae_data")
OUT_DIR = Path("outputs/stage5_rgae_detection")

SEED = 42
SEM_DIM = 64
HIDDEN_DIM = 64
LATENT_DIM = 32

EPOCHS = 300
LR = 0.005
WEIGHT_DECAY = 1e-4

NEG_RATIO = 1.0
REL_AUG_RATE = 0.15
REL_AUG_MIN = 0.40
REL_AUG_MAX = 0.90

VALIDATE_EVERY = 10
EARLY_STOPPING = True
PATIENCE_EPOCHS = 80
MIN_DELTA = 1e-5

CHECKPOINT_POLICY = "best_val"  # choices: best_val, last
EPS = 1e-3

DEVICE = "cpu"


# =========================
# Arguments
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rinnegan Stage-5 R-GAE training and detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage4-root", type=Path, default=STAGE4_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)

    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])

    parser.add_argument("--sem-dim", type=int, default=SEM_DIM)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)

    parser.add_argument("--neg-ratio", type=float, default=NEG_RATIO)
    parser.add_argument("--rel-aug-rate", type=float, default=REL_AUG_RATE)
    parser.add_argument("--rel-aug-min", type=float, default=REL_AUG_MIN)
    parser.add_argument("--rel-aug-max", type=float, default=REL_AUG_MAX)

    parser.add_argument("--validate-every", type=int, default=VALIDATE_EVERY)
    parser.add_argument("--patience-epochs", type=int, default=PATIENCE_EPOCHS)
    parser.add_argument("--min-delta", type=float, default=MIN_DELTA)
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Disable early stopping.",
    )
    parser.add_argument(
        "--checkpoint-policy",
        type=str,
        default=CHECKPOINT_POLICY,
        choices=["best_val", "last"],
        help="Checkpoint used for calibration and test-time scoring.",
    )
    return parser.parse_args()


# =========================
# Reproducibility
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# IO helpers
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


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


def state_dict_to_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict_to_device(model: nn.Module, state_dict_cpu: Dict[str, torch.Tensor]):
    model.load_state_dict({k: v.to(DEVICE) for k, v in state_dict_cpu.items()})


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Feature construction
# =========================

def tokenize_text(text: str) -> List[str]:
    text = safe_str(text).lower()
    tokens = re.split(r"[^a-zA-Z0-9_.:/-]+", text)
    out = []

    for t in tokens:
        t = t.strip()
        if not t:
            continue
        out.append(t)
        if "/" in t:
            out.extend([x for x in t.split("/") if x])
        if "." in t:
            out.extend([x for x in t.split(".") if x])
        if "-" in t:
            out.extend([x for x in t.split("-") if x])
        if ":" in t:
            out.extend([x for x in t.split(":") if x])

    return [x for x in out if x]


def stable_hash_int(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def build_semantic_feature(text: str, dim: int) -> np.ndarray:
    """
    Lightweight semantic encoder for artifact execution.

    The paper-level implementation can replace this hashing encoder with
    Word2Vec or another tokenizer-based semantic embedding.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = tokenize_text(text)

    if not tokens:
        return vec

    for tok in tokens:
        h = stable_hash_int(tok)
        idx = h % dim
        sign = 1.0 if ((h // dim) % 2 == 0) else -1.0
        vec[idx] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec.astype(np.float32)


def to_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    if col not in row:
        return default
    try:
        if pd.isna(row[col]):
            return default
        return float(row[col])
    except Exception:
        return default


def build_node_features(nodes_df: pd.DataFrame, sem_dim: int) -> np.ndarray:
    """
    Build node features h_v^(0) = semantic || type || reliability.

    Diagnostic label columns, such as is_malicious, are intentionally not used.
    """
    feats = []

    for _, r in nodes_df.iterrows():
        semantic_text = safe_str(r.get("semantic_text"))
        if not semantic_text:
            semantic_text = " ".join([
                safe_str(r.get("raw_entity")),
                safe_str(r.get("canonical_id")),
                safe_str(r.get("entity_type")),
                safe_str(r.get("host")),
            ])

        sem = build_semantic_feature(semantic_text, dim=sem_dim)

        type_feat = np.array([
            to_float(r, "type_process", 0.0),
            to_float(r, "type_file", 0.0),
            to_float(r, "type_socket", 0.0),
            to_float(r, "type_unknown", 0.0),
        ], dtype=np.float32)

        rel_feat = np.array([
            to_float(r, "reliability", 1.0),
            to_float(r, "src_observed", 1.0),
            to_float(r, "src_recovered_copy", 0.0),
            to_float(r, "is_recovered_copy", 0.0),
        ], dtype=np.float32)

        x = np.concatenate([sem, type_feat, rel_feat], axis=0)
        feats.append(x)

    if not feats:
        return np.zeros((0, sem_dim + 8), dtype=np.float32)

    return np.stack(feats, axis=0).astype(np.float32)


# =========================
# Graph loading
# =========================

class GraphData:
    def __init__(
        self,
        name: str,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        copy_mapping_df: pd.DataFrame,
        relation_vocab: Dict[str, int],
        sem_dim: int,
    ):
        self.name = name
        self.nodes_df = nodes_df.reset_index(drop=True).copy()
        self.edges_df = edges_df.reset_index(drop=True).copy()
        self.copy_mapping_df = copy_mapping_df.copy()

        if "node_id" not in self.nodes_df.columns:
            raise ValueError(f"{name}: nodes.csv must contain node_id column.")

        self.node_ids = self.nodes_df["node_id"].astype(str).tolist()
        self.node_to_idx = {n: i for i, n in enumerate(self.node_ids)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}

        self.relation_vocab = relation_vocab
        self.num_relations = len(relation_vocab)

        self.x_np = build_node_features(self.nodes_df, sem_dim=sem_dim)
        self.x = torch.tensor(self.x_np, dtype=torch.float32, device=DEVICE)

        self.edge_index, self.edge_rel, self.edge_weight, self.valid_edges_df = self._build_edges()

    def _build_edges(self):
        src_idx = []
        dst_idx = []
        rel_ids = []
        weights = []
        rows = []

        if len(self.edges_df) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)
            edge_rel = torch.empty((0,), dtype=torch.long, device=DEVICE)
            edge_weight = torch.empty((0,), dtype=torch.float32, device=DEVICE)
            return edge_index, edge_rel, edge_weight, pd.DataFrame()

        for _, r in self.edges_df.iterrows():
            src = safe_str(r.get("src_node"))
            dst = safe_str(r.get("dst_node"))
            rel = safe_str(r.get("relation"))

            if src not in self.node_to_idx or dst not in self.node_to_idx:
                continue
            if rel not in self.relation_vocab:
                continue

            w = None
            for col in ["omega_e", "reliability"]:
                if col in r and not pd.isna(r[col]):
                    try:
                        w = float(r[col])
                        break
                    except Exception:
                        pass
            if w is None:
                w = 1.0

            src_idx.append(self.node_to_idx[src])
            dst_idx.append(self.node_to_idx[dst])
            rel_ids.append(self.relation_vocab[rel])
            weights.append(float(w))
            rows.append(r.to_dict())

        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long, device=DEVICE)
        edge_rel = torch.tensor(rel_ids, dtype=torch.long, device=DEVICE)
        edge_weight = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

        valid_edges_df = pd.DataFrame(rows)
        return edge_index, edge_rel, edge_weight, valid_edges_df

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


def load_relation_vocab(stage4_root: Path) -> Dict[str, int]:
    relation_vocab_path = stage4_root / "relation_vocab.json"
    if relation_vocab_path.exists():
        vocab = read_json(relation_vocab_path)
        return {str(k): int(v) for k, v in vocab.items()}

    default = ["read", "write", "execute", "spawn", "connect", "accept", "chmod", "unlink", "rename"]
    return {r: i for i, r in enumerate(default)}


def load_graph(name: str, graph_dir: Path, relation_vocab: Dict[str, int], sem_dim: int) -> GraphData:
    nodes = read_csv(graph_dir / "nodes.csv")
    edges = read_csv(graph_dir / "edges.csv")
    copy_mapping_path = graph_dir / "copy_mapping.csv"
    copy_mapping = read_csv(copy_mapping_path) if copy_mapping_path.exists() else pd.DataFrame()

    if "src_node" not in edges.columns or "dst_node" not in edges.columns:
        raise ValueError(f"{graph_dir}/edges.csv missing src_node/dst_node columns.")

    graph = GraphData(name, nodes, edges, copy_mapping, relation_vocab, sem_dim=sem_dim)

    if graph.num_nodes == 0:
        raise ValueError(f"{name}: graph contains no nodes.")
    if graph.num_edges == 0:
        raise ValueError(f"{name}: graph contains no valid edges.")

    return graph


def validate_stage4_root(stage4_root: Path):
    required_dirs = [
        stage4_root / "train_benign_graph",
        stage4_root / "val_benign_graph",
        stage4_root / "test_clean_graph",
        stage4_root / "test_corrupted_graph",
        stage4_root / "test_remediated_graph",
    ]
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing Stage-4 graph folder: {d}")
        for fname in ["nodes.csv", "edges.csv"]:
            if not (d / fname).exists():
                raise FileNotFoundError(f"Missing graph file: {d / fname}")


# =========================
# Sparse adjacency
# =========================

def build_norm_adj(num_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
    """
    Build symmetric reliability-weighted normalized adjacency.

    Propagation uses a symmetrized adjacency, while the decoder still scores
    typed directed edges.
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    if edge_index.numel() == 0:
        idx = torch.arange(num_nodes, device=DEVICE)
        indices = torch.stack([idx, idx], dim=0)
        values = torch.ones(num_nodes, dtype=torch.float32, device=DEVICE)
        return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=DEVICE).coalesce()

    src, dst = edge_index[0], edge_index[1]
    w = edge_weight.clamp(min=0.0)

    rows = torch.cat([src, dst], dim=0)
    cols = torch.cat([dst, src], dim=0)
    vals = torch.cat([w, w], dim=0)

    loop = torch.arange(num_nodes, device=DEVICE)
    rows = torch.cat([rows, loop], dim=0)
    cols = torch.cat([cols, loop], dim=0)
    vals = torch.cat([vals, torch.ones(num_nodes, dtype=torch.float32, device=DEVICE)], dim=0)

    deg = torch.zeros(num_nodes, dtype=torch.float32, device=DEVICE)
    deg.index_add_(0, rows, vals)

    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
    norm_vals = vals * deg_inv_sqrt[rows] * deg_inv_sqrt[cols]

    indices = torch.stack([rows, cols], dim=0)
    adj = torch.sparse_coo_tensor(indices, norm_vals, (num_nodes, num_nodes), device=DEVICE)
    return adj.coalesce()


# =========================
# Negative sampling
# =========================

def make_positive_set(edge_index: torch.Tensor, edge_rel: torch.Tensor) -> Set[Tuple[int, int, int]]:
    src = edge_index[0].detach().cpu().numpy().tolist()
    dst = edge_index[1].detach().cpu().numpy().tolist()
    rel = edge_rel.detach().cpu().numpy().tolist()
    return set(zip(src, dst, rel))


def sample_negative_edges(
    num_nodes: int,
    num_samples: int,
    relation_ids: List[int],
    positive_set: Set[Tuple[int, int, int]],
    rng: Optional[random.Random] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if rng is None:
        rng = random

    if num_nodes < 2:
        raise ValueError("Negative sampling requires at least two nodes.")

    if not relation_ids:
        raise ValueError("Negative sampling requires at least one relation id.")

    neg_src = []
    neg_dst = []
    neg_rel = []

    max_try = max(num_samples * 100, 1000)
    tries = 0
    seen = set()

    while len(neg_src) < num_samples and tries < max_try:
        tries += 1
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)

        if u == v:
            continue

        r = rng.choice(relation_ids)
        key = (u, v, r)

        if key in positive_set or key in seen:
            continue

        seen.add(key)
        neg_src.append(u)
        neg_dst.append(v)
        neg_rel.append(r)

    if len(neg_src) == 0:
        raise RuntimeError("Failed to sample negative edges.")

    edge_index = torch.tensor([neg_src, neg_dst], dtype=torch.long, device=DEVICE)
    edge_rel = torch.tensor(neg_rel, dtype=torch.long, device=DEVICE)
    return edge_index, edge_rel


# =========================
# Model
# =========================

class RGAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int, num_relations: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.lin2 = nn.Linear(hidden_dim, latent_dim, bias=False)

        self.rel_diag = nn.Parameter(torch.empty(num_relations, latent_dim))
        nn.init.xavier_uniform_(self.rel_diag)

    def encode(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        h = torch.sparse.mm(norm_adj, x)
        h = F.relu(self.lin1(h))
        h = torch.sparse.mm(norm_adj, h)
        z = self.lin2(h)
        return z

    def decode_logits(self, z: torch.Tensor, edge_index: torch.Tensor, edge_rel: torch.Tensor) -> torch.Tensor:
        u = edge_index[0]
        v = edge_index[1]
        rel_diag = self.rel_diag[edge_rel]
        logits = torch.sum(z[u] * rel_diag * z[v], dim=1)
        return logits


# =========================
# Training / validation
# =========================

def link_prediction_loss(
    model: RGAE,
    graph: GraphData,
    neg_edge_index: torch.Tensor,
    neg_edge_rel: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if pos_weight is None:
        pos_weight = graph.edge_weight

    norm_adj = build_norm_adj(graph.num_nodes, graph.edge_index, pos_weight)
    z = model.encode(graph.x, norm_adj)

    pos_logits = model.decode_logits(z, graph.edge_index, graph.edge_rel)
    neg_logits = model.decode_logits(z, neg_edge_index, neg_edge_rel)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits,
        torch.ones_like(pos_logits),
        weight=pos_weight,
        reduction="mean",
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits,
        torch.zeros_like(neg_logits),
        reduction="mean",
    )
    loss = pos_loss + neg_loss

    return {
        "loss_tensor": loss,
        "loss": float(loss.detach().cpu().item()),
        "pos_loss": float(pos_loss.detach().cpu().item()),
        "neg_loss": float(neg_loss.detach().cpu().item()),
        "pos_prob_mean": float(torch.sigmoid(pos_logits).mean().detach().cpu().item()),
        "neg_prob_mean": float(torch.sigmoid(neg_logits).mean().detach().cpu().item()),
    }


def make_fixed_validation_negatives(
    graph: GraphData,
    relation_ids: List[int],
    seed: int,
    neg_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    val_pos_set = make_positive_set(graph.edge_index, graph.edge_rel)
    num_val_neg = max(1, int(graph.num_edges * neg_ratio))
    val_rng = random.Random(seed)
    return sample_negative_edges(
        num_nodes=graph.num_nodes,
        num_samples=num_val_neg,
        relation_ids=relation_ids,
        positive_set=val_pos_set,
        rng=val_rng,
    )


def train_rgae(
    model: RGAE,
    train_graph: GraphData,
    val_graph: GraphData,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, torch.Tensor]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    train_pos_set = make_positive_set(train_graph.edge_index, train_graph.edge_rel)
    train_relation_ids = sorted(set(train_graph.edge_rel.detach().cpu().numpy().tolist()))
    if not train_relation_ids:
        train_relation_ids = list(range(train_graph.num_relations))

    val_neg_edge_index, val_neg_edge_rel = make_fixed_validation_negatives(
        val_graph,
        relation_ids=train_relation_ids,
        seed=cfg["seed"] + 999,
        neg_ratio=cfg["neg_ratio"],
    )

    history = []
    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    best_row = {}
    stop_reason = "max_epochs_reached"

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()

        pos_weight = train_graph.edge_weight.clone()

        if cfg["rel_aug_rate"] > 0:
            mask = torch.rand_like(pos_weight) < cfg["rel_aug_rate"]
            sampled = cfg["rel_aug_min"] + (cfg["rel_aug_max"] - cfg["rel_aug_min"]) * torch.rand_like(pos_weight)
            pos_weight = torch.where(mask, sampled, pos_weight)

        num_neg = max(1, int(train_graph.num_edges * cfg["neg_ratio"]))
        neg_edge_index, neg_edge_rel = sample_negative_edges(
            num_nodes=train_graph.num_nodes,
            num_samples=num_neg,
            relation_ids=train_relation_ids,
            positive_set=train_pos_set,
        )

        train_stats = link_prediction_loss(
            model=model,
            graph=train_graph,
            neg_edge_index=neg_edge_index,
            neg_edge_rel=neg_edge_rel,
            pos_weight=pos_weight,
        )

        loss = train_stats["loss_tensor"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_stats = None
        improved = False

        should_validate = (epoch == 1) or (epoch % cfg["validate_every"] == 0) or (epoch == cfg["epochs"])
        if should_validate:
            model.eval()
            with torch.no_grad():
                val_stats = link_prediction_loss(
                    model=model,
                    graph=val_graph,
                    neg_edge_index=val_neg_edge_index,
                    neg_edge_rel=val_neg_edge_rel,
                    pos_weight=val_graph.edge_weight,
                )

            val_loss = val_stats["loss"]
            if val_loss < best_val_loss - cfg["min_delta"]:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = state_dict_to_cpu(model)
                improved = True
                best_row = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_pos_loss": val_stats["pos_loss"],
                    "val_neg_loss": val_stats["neg_loss"],
                    "val_pos_prob_mean": val_stats["pos_prob_mean"],
                    "val_neg_prob_mean": val_stats["neg_prob_mean"],
                }

        if should_validate:
            row = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_pos_loss": train_stats["pos_loss"],
                "train_neg_loss": train_stats["neg_loss"],
                "train_pos_prob_mean": train_stats["pos_prob_mean"],
                "train_neg_prob_mean": train_stats["neg_prob_mean"],
                "val_loss": val_stats["loss"] if val_stats else np.nan,
                "val_pos_loss": val_stats["pos_loss"] if val_stats else np.nan,
                "val_neg_loss": val_stats["neg_loss"] if val_stats else np.nan,
                "val_pos_prob_mean": val_stats["pos_prob_mean"] if val_stats else np.nan,
                "val_neg_prob_mean": val_stats["neg_prob_mean"] if val_stats else np.nan,
                "is_best": int(improved),
                "best_epoch_so_far": best_epoch,
                "best_val_loss_so_far": best_val_loss,
            }
            history.append(row)

            msg = (
                f"Epoch {epoch:04d} | "
                f"train={train_stats['loss']:.4f} "
                f"(pos={train_stats['pos_loss']:.4f}, neg={train_stats['neg_loss']:.4f})"
            )

            if val_stats:
                msg += (
                    f" | val={val_stats['loss']:.4f} "
                    f"(pos={val_stats['pos_loss']:.4f}, neg={val_stats['neg_loss']:.4f})"
                    f" | best={best_val_loss:.4f}@{best_epoch}"
                )
                if improved:
                    msg += " *"

            print(msg)

        if cfg["early_stopping"] and best_epoch > 0:
            if epoch - best_epoch >= cfg["patience_epochs"]:
                stop_reason = f"early_stopping_patience_{cfg['patience_epochs']}_epochs"
                print(
                    f"[EarlyStop] Stop at epoch {epoch}; "
                    f"best epoch = {best_epoch}, best val loss = {best_val_loss:.6f}"
                )
                break

    if best_state is None:
        best_state = state_dict_to_cpu(model)
        best_epoch = cfg["epochs"]
        best_val_loss = float("nan")

    checkpoint_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "stop_reason": stop_reason,
        "max_epochs": cfg["epochs"],
        "early_stopping": cfg["early_stopping"],
        "patience_epochs": cfg["patience_epochs"],
        "validate_every": cfg["validate_every"],
        "min_delta": cfg["min_delta"],
        "best_validation_row": best_row,
    }

    return pd.DataFrame(history), checkpoint_info, best_state


# =========================
# Scoring and evaluation
# =========================

def get_node_reliability(nodes_df: pd.DataFrame) -> np.ndarray:
    vals = []
    for _, r in nodes_df.iterrows():
        vals.append(safe_float(r.get("reliability", 1.0), default=1.0))
    return np.array(vals, dtype=np.float32)


def build_copy_mapping(copy_mapping_df: pd.DataFrame) -> Dict[str, str]:
    if len(copy_mapping_df) == 0:
        return {}

    if "copy_node_id" not in copy_mapping_df.columns or "canonical_node_id" not in copy_mapping_df.columns:
        return {}

    return dict(zip(copy_mapping_df["copy_node_id"].astype(str), copy_mapping_df["canonical_node_id"].astype(str)))


def node_to_canonical_key(node_row: pd.Series, copy_map: Dict[str, str]) -> str:
    node_id = safe_str(node_row.get("node_id"))

    if node_id in copy_map:
        return copy_map[node_id]

    if node_id.startswith("obs::"):
        return node_id

    canonical_id = safe_str(node_row.get("canonical_id"))
    if canonical_id:
        return f"obs::{canonical_id}"

    return node_id


def compute_graph_scores(model: RGAE, graph: GraphData) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model.eval()

    with torch.no_grad():
        norm_adj = build_norm_adj(graph.num_nodes, graph.edge_index, graph.edge_weight)
        z = model.encode(graph.x, norm_adj)
        logits = model.decode_logits(z, graph.edge_index, graph.edge_rel)
        prob = torch.sigmoid(logits).clamp(min=1e-8, max=1.0)
        edge_error = -torch.log(prob)

    edge_error_np = edge_error.detach().cpu().numpy()
    edge_prob_np = prob.detach().cpu().numpy()

    edge_scores_df = graph.valid_edges_df.copy()
    edge_scores_df["recon_prob"] = edge_prob_np
    edge_scores_df["recon_error"] = edge_error_np

    src = graph.edge_index[0].detach().cpu().numpy()
    dst = graph.edge_index[1].detach().cpu().numpy()
    w = graph.edge_weight.detach().cpu().numpy()

    numerator = np.zeros(graph.num_nodes, dtype=np.float64)
    denominator = np.zeros(graph.num_nodes, dtype=np.float64)

    for i in range(len(edge_error_np)):
        u = src[i]
        v = dst[i]
        weight = float(w[i])
        err = float(edge_error_np[i])

        numerator[u] += weight * err
        denominator[u] += weight

        numerator[v] += weight * err
        denominator[v] += weight

    score = numerator / (denominator + EPS)
    confidence = get_node_reliability(graph.nodes_df)

    node_scores_df = graph.nodes_df.copy()
    node_scores_df["node_score"] = score
    node_scores_df["confidence"] = confidence

    copy_map = build_copy_mapping(graph.copy_mapping_df)

    canonical_rows = {}
    for _, r in node_scores_df.iterrows():
        ckey = node_to_canonical_key(r, copy_map)
        s = float(r["node_score"])
        conf = float(r["confidence"])
        is_mal = safe_int(r.get("is_malicious", 0), default=0)

        if ckey not in canonical_rows or s > canonical_rows[ckey]["score"]:
            canonical_rows[ckey] = {
                "canonical_node_id": ckey,
                "canonical_entity": ckey.replace("obs::", "", 1),
                "score": s,
                "confidence": conf,
                "best_node_id": safe_str(r.get("node_id")),
                "best_node_source": safe_str(r.get("source")),
                "entity_type": safe_str(r.get("entity_type")),
                "host": safe_str(r.get("host")),
                "is_malicious": is_mal,
            }
        else:
            canonical_rows[ckey]["is_malicious"] = max(canonical_rows[ckey]["is_malicious"], is_mal)

    canonical_df = pd.DataFrame(list(canonical_rows.values()))
    if len(canonical_df):
        canonical_df = canonical_df.sort_values("score", ascending=False).reset_index(drop=True)

    return node_scores_df, edge_scores_df, canonical_df


def compute_metrics(canonical_df: pd.DataFrame, delta: float) -> Dict[str, Any]:
    if len(canonical_df) == 0:
        return {
            "delta": float(delta),
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
            "num_nodes": 0,
            "num_malicious": 0,
            "num_predicted": 0,
        }

    if "is_malicious" not in canonical_df.columns:
        raise ValueError("canonical_df must contain is_malicious for diagnostic evaluation.")

    y_true = canonical_df["is_malicious"].astype(int).values
    y_pred = (canonical_df["score"].values > delta).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0

    return {
        "delta": float(delta),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "num_nodes": int(len(canonical_df)),
        "num_malicious": int(y_true.sum()),
        "num_predicted": int(y_pred.sum()),
    }


def compute_topk_metrics(canonical_df: pd.DataFrame) -> Dict[str, Any]:
    if len(canonical_df) == 0:
        return {"k": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    y_true = canonical_df["is_malicious"].astype(int).values
    k = int(y_true.sum())

    if k <= 0:
        return {"k": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    df = canonical_df.sort_values("score", ascending=False).reset_index(drop=True)
    pred = np.zeros(len(df), dtype=int)
    pred[:k] = 1

    y = df["is_malicious"].astype(int).values

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


def add_predictions(canonical_df: pd.DataFrame, delta: float) -> pd.DataFrame:
    out = canonical_df.copy()
    if len(out):
        out["pred_anomalous"] = (out["score"].values > delta).astype(int)
    else:
        out["pred_anomalous"] = []
    return out


# =========================
# Main
# =========================

def main():
    args = parse_args()

    global STAGE4_ROOT, OUT_DIR, DEVICE

    STAGE4_ROOT = args.stage4_root
    OUT_DIR = args.out_dir
    DEVICE = resolve_device(args.device)

    cfg = {
        "seed": args.seed,
        "sem_dim": args.sem_dim,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "neg_ratio": args.neg_ratio,
        "rel_aug_rate": args.rel_aug_rate,
        "rel_aug_min": args.rel_aug_min,
        "rel_aug_max": args.rel_aug_max,
        "validate_every": args.validate_every,
        "early_stopping": not args.disable_early_stopping,
        "patience_epochs": args.patience_epochs,
        "min_delta": args.min_delta,
        "checkpoint_policy": args.checkpoint_policy,
        "device": DEVICE,
    }

    set_seed(cfg["seed"])
    ensure_dir(OUT_DIR)
    validate_stage4_root(STAGE4_ROOT)

    train_graph_dir = STAGE4_ROOT / "train_benign_graph"
    val_graph_dir = STAGE4_ROOT / "val_benign_graph"
    test_graph_dirs = {
        "clean": STAGE4_ROOT / "test_clean_graph",
        "corrupted": STAGE4_ROOT / "test_corrupted_graph",
        "remediated": STAGE4_ROOT / "test_remediated_graph",
    }

    print("=== Rinnegan Stage-5: R-GAE Training and Detection ===")
    print(f"STAGE4_ROOT = {STAGE4_ROOT}")
    print(f"OUT_DIR     = {OUT_DIR}")
    print(f"DEVICE      = {DEVICE}")
    print(f"CUDA avail. = {torch.cuda.is_available()}")
    print("")

    relation_vocab = load_relation_vocab(STAGE4_ROOT)

    train_graph = load_graph("train_benign", train_graph_dir, relation_vocab, sem_dim=cfg["sem_dim"])
    val_graph = load_graph("val_benign", val_graph_dir, relation_vocab, sem_dim=cfg["sem_dim"])

    print(f"Train graph: nodes={train_graph.num_nodes}, edges={train_graph.num_edges}")
    print(f"Val graph:   nodes={val_graph.num_nodes}, edges={val_graph.num_edges}")
    print("")

    in_dim = train_graph.x.shape[1]
    model = RGAE(
        in_dim=in_dim,
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        num_relations=len(relation_vocab),
    ).to(DEVICE)

    history_df, checkpoint_info, best_state = train_rgae(model, train_graph, val_graph, cfg=cfg)
    history_df.to_csv(OUT_DIR / "training_history.csv", index=False, encoding="utf-8-sig")

    last_checkpoint = {
        "model_state_dict": state_dict_to_cpu(model),
        "relation_vocab": relation_vocab,
        "config": cfg,
        "checkpoint_info": checkpoint_info,
    }
    torch.save(last_checkpoint, OUT_DIR / "rgae_model_last.pt")

    best_checkpoint = {
        "model_state_dict": best_state,
        "relation_vocab": relation_vocab,
        "config": cfg,
        "checkpoint_info": checkpoint_info,
    }
    torch.save(best_checkpoint, OUT_DIR / "rgae_model_best.pt")

    if cfg["checkpoint_policy"] == "best_val":
        load_state_dict_to_device(model, best_state)
        checkpoint_used = "best_val"
    elif cfg["checkpoint_policy"] == "last":
        checkpoint_used = "last"
    else:
        raise ValueError(f"Unknown checkpoint policy: {cfg['checkpoint_policy']}")

    final_checkpoint = {
        "model_state_dict": state_dict_to_cpu(model),
        "relation_vocab": relation_vocab,
        "config": cfg,
        "checkpoint_info": checkpoint_info,
        "checkpoint_used_for_detection": checkpoint_used,
    }
    torch.save(final_checkpoint, OUT_DIR / "rgae_model.pt")

    checkpoint_info["checkpoint_policy"] = cfg["checkpoint_policy"]
    checkpoint_info["checkpoint_used_for_detection"] = checkpoint_used
    write_json(OUT_DIR / "checkpoint_info.json", checkpoint_info)

    # Calibration on benign validation graph.
    val_node_scores, val_edge_scores, val_canonical_scores = compute_graph_scores(model, val_graph)
    val_node_scores.to_csv(OUT_DIR / "val_node_instance_scores.csv", index=False, encoding="utf-8-sig")
    val_edge_scores.to_csv(OUT_DIR / "val_edge_errors.csv", index=False, encoding="utf-8-sig")
    val_canonical_scores.to_csv(OUT_DIR / "val_canonical_scores.csv", index=False, encoding="utf-8-sig")

    mu_b = float(val_canonical_scores["score"].mean()) if len(val_canonical_scores) else 0.0
    sigma_b = float(val_canonical_scores["score"].std(ddof=0)) if len(val_canonical_scores) else 0.0
    delta = mu_b + 2.0 * sigma_b

    calibration = {
        "mu_b": mu_b,
        "sigma_b": sigma_b,
        "delta": delta,
        "num_val_canonical_entities": int(len(val_canonical_scores)),
        "checkpoint_used": checkpoint_used,
        "checkpoint_policy": cfg["checkpoint_policy"],
        "best_epoch": checkpoint_info["best_epoch"],
        "best_val_loss": checkpoint_info["best_val_loss"],
    }
    write_json(OUT_DIR / "calibration.json", calibration)

    print("")
    print(
        f"Calibration: mu_b={mu_b:.6f}, sigma_b={sigma_b:.6f}, delta={delta:.6f} "
        f"| checkpoint_used={checkpoint_used} "
        f"| best_epoch={checkpoint_info['best_epoch']} "
        f"| best_val_loss={checkpoint_info['best_val_loss']:.6f}"
    )
    print("")

    all_metrics = {
        "calibration": calibration,
        "checkpoint_info": checkpoint_info,
        "threshold_metrics": {},
        "topk_metrics": {},
    }

    for test_name, graph_dir in test_graph_dirs.items():
        print(f"[Test] {test_name}")

        graph = load_graph(test_name, graph_dir, relation_vocab, sem_dim=cfg["sem_dim"])
        node_scores, edge_scores, canonical_scores = compute_graph_scores(model, graph)

        node_scores.to_csv(OUT_DIR / f"{test_name}_node_instance_scores.csv", index=False, encoding="utf-8-sig")
        edge_scores.to_csv(OUT_DIR / f"{test_name}_edge_errors.csv", index=False, encoding="utf-8-sig")

        canonical_scores = add_predictions(canonical_scores, delta)
        canonical_scores.to_csv(OUT_DIR / f"{test_name}_canonical_scores.csv", index=False, encoding="utf-8-sig")

        threshold_metrics = compute_metrics(canonical_scores, delta)
        topk_metrics = compute_topk_metrics(canonical_scores)

        all_metrics["threshold_metrics"][test_name] = threshold_metrics
        all_metrics["topk_metrics"][test_name] = topk_metrics

        print(
            f"  threshold: P={threshold_metrics['precision']} "
            f"R={threshold_metrics['recall']} F1={threshold_metrics['f1']} "
            f"pred={threshold_metrics['num_predicted']} malicious={threshold_metrics['num_malicious']}"
        )
        print(
            f"  top-k:     P={topk_metrics['precision']} "
            f"R={topk_metrics['recall']} F1={topk_metrics['f1']} k={topk_metrics['k']}"
        )

    write_json(OUT_DIR / "detection_metrics.json", all_metrics)

    lines = []
    lines.append("Rinnegan Stage-5: R-GAE Detection Summary")
    lines.append("")
    lines.append("Configuration")
    lines.append(f"  device: {DEVICE}")
    lines.append(f"  cuda_available: {torch.cuda.is_available()}")
    lines.append(f"  seed: {cfg['seed']}")
    lines.append(f"  epochs: {cfg['epochs']}")
    lines.append(f"  hidden_dim: {cfg['hidden_dim']}")
    lines.append(f"  latent_dim: {cfg['latent_dim']}")
    lines.append(f"  semantic_dim: {cfg['sem_dim']}")
    lines.append(f"  neg_ratio: {cfg['neg_ratio']}")
    lines.append(f"  rel_aug_rate: {cfg['rel_aug_rate']}")
    lines.append(f"  validate_every: {cfg['validate_every']}")
    lines.append(f"  early_stopping: {cfg['early_stopping']}")
    lines.append(f"  patience_epochs: {cfg['patience_epochs']}")
    lines.append("")
    lines.append("Checkpoint")
    lines.append(f"  checkpoint_used: {checkpoint_used}")
    lines.append(f"  checkpoint_policy: {cfg['checkpoint_policy']}")
    lines.append(f"  best_epoch: {checkpoint_info['best_epoch']}")
    lines.append(f"  best_val_loss: {checkpoint_info['best_val_loss']:.6f}")
    lines.append(f"  stop_reason: {checkpoint_info['stop_reason']}")
    lines.append("")
    lines.append("Calibration")
    lines.append(f"  mu_b: {mu_b:.6f}")
    lines.append(f"  sigma_b: {sigma_b:.6f}")
    lines.append(f"  delta: {delta:.6f}")
    lines.append("")
    lines.append("Threshold-based metrics")
    for name, m in all_metrics["threshold_metrics"].items():
        lines.append(
            f"  {name}: P={m['precision']}, R={m['recall']}, F1={m['f1']}, "
            f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, pred={m['num_predicted']}, malicious={m['num_malicious']}"
        )
    lines.append("")
    lines.append("Top-K ranking metrics")
    for name, m in all_metrics["topk_metrics"].items():
        lines.append(
            f"  {name}: K={m['k']}, P={m['precision']}, R={m['recall']}, F1={m['f1']}, "
            f"TP={m['tp']}, FP={m['fp']}, FN={m['fn']}"
        )
    lines.append("")
    lines.append("Feature safety")
    lines.append("  Diagnostic labels are used only for metrics and are not included in node features.")
    lines.append("")
    lines.append("Expected interpretation")
    lines.append("  clean is the complete-evidence reference graph.")
    lines.append("  corrupted has missing attack-related edges.")
    lines.append("  remediated adds reliability-weighted recovered hypotheses.")
    lines.append("")
    lines.append("Outputs")
    lines.append(f"  {OUT_DIR / 'training_history.csv'}")
    lines.append(f"  {OUT_DIR / 'checkpoint_info.json'}")
    lines.append(f"  {OUT_DIR / 'rgae_model.pt'}")
    lines.append(f"  {OUT_DIR / 'rgae_model_best.pt'}")
    lines.append(f"  {OUT_DIR / 'rgae_model_last.pt'}")
    lines.append(f"  {OUT_DIR / 'calibration.json'}")
    lines.append(f"  {OUT_DIR / 'detection_metrics.json'}")
    lines.append(f"  {OUT_DIR / '*_canonical_scores.csv'}")
    lines.append(f"  {OUT_DIR / '*_edge_errors.csv'}")

    summary = "\n".join(lines)
    write_text(OUT_DIR / "detection_summary.txt", summary)

    print("")
    print(summary)
    print("")
    print("[DONE] Stage-5 R-GAE training and detection completed.")


if __name__ == "__main__":
    main()