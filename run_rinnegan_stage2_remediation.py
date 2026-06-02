# -*- coding: utf-8 -*-
"""
Rinnegan Stage-2: Corrupted-window remediation.

This script implements the Stage-2 remediation module for the anonymized
Rinnegan artifact package. It takes candidate corrupted windows from Stage-1,
constructs three-window context, invokes an LLM or a local mock backend, validates
schema-compatible recovered event hypotheses, and assigns reliability scores.

Default usage from the repository root:
    python scripts/run_stage2_remediation.py \
        --data-root OfficeFog_desensitized_version/full-dataset \
        --stage1-out outputs/stage1_discovery \
        --out-dir outputs/stage2_remediation \
        --llm-backend mock

For OpenAI-compatible LLM inference:
    set RINNEGAN_OPENAI_API_KEY=your_api_key
    python scripts/run_stage2_remediation.py --llm-backend openai --model gpt-4o-mini

Backends:
    mock    : deterministic local backend for artifact smoke tests.
    cached  : reuse cached JSON outputs from --cache-dir.
    openai  : call an OpenAI-compatible chat completion endpoint.
"""

import argparse
import os
import re
import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# =========================
# Config
# =========================

DATA_ROOT = Path("OfficeFog_desensitized_version/full-dataset")
STAGE1_OUT = Path("outputs/stage1_discovery")
OUT_DIR = Path("outputs/stage2_remediation")
CACHE_DIR = Path("cached_llm_runs")

TIME_START = pd.Timestamp("2026-01-20 09:00:00")
WINDOW_SIZE_MINUTES = 10

CANDIDATE_MODE = "stage1_threshold"
MANUAL_WINDOWS = ""

N_RUNS = 5
TEMPERATURE = 0.45
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 5
GAMMA = 0.60
FALLBACK_TOP_N = 6

LLM_BACKEND = "mock"
MODEL_NAME = "gpt-4o-mini"
API_KEY = os.environ.get("RINNEGAN_OPENAI_API_KEY", "").strip()
BASE_URL = os.environ.get("RINNEGAN_OPENAI_BASE_URL", "").strip()

MAX_AUDIT_ROWS_PER_WINDOW = 90
MAX_FLOW_ROWS_PER_WINDOW = 80
MAX_METRIC_ROWS_PER_WINDOW = 30
MAX_METRIC_EVENT_ROWS_PER_WINDOW = 40

ALLOWED_RELATIONS = {
    "read": ["process", "file"],
    "write": ["process", "file"],
    "execute": ["process", "file"],
    "spawn": ["process", "process"],
    "connect": ["process", "socket"],
    "accept": ["process", "socket"],
    "chmod": ["process", "file"],
    "unlink": ["process", "file"],
    "rename": ["process", "file"],
}


# =========================
# Arguments
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rinnegan Stage-2 corrupted-window remediation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--stage1-out", type=Path, default=STAGE1_OUT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)

    parser.add_argument(
        "--candidate-mode",
        type=str,
        default=CANDIDATE_MODE,
        choices=["stage1_threshold", "stage1_topn", "manual", "gt_corrupted_windows_diagnostic"],
        help="How candidate corrupted windows are selected. The diagnostic GT mode should not be used for final evaluation.",
    )
    parser.add_argument(
        "--manual-windows",
        type=str,
        default=MANUAL_WINDOWS,
        help="Comma-separated windows used only when --candidate-mode manual.",
    )
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--fallback-top-n", type=int, default=FALLBACK_TOP_N)

    parser.add_argument(
        "--llm-backend",
        type=str,
        default=LLM_BACKEND,
        choices=["mock", "cached", "openai"],
        help="Remediation backend.",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--base-url", type=str, default=BASE_URL)
    parser.add_argument("--api-key", type=str, default=API_KEY)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--retry-sleep", type=float, default=RETRY_SLEEP_SECONDS)
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


def text_col(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=str)
    return df[name].astype(str)


def numeric_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)


def infer_window_start(window_id: str) -> pd.Timestamp:
    idx = int(safe_str(window_id).replace("W", ""))
    return TIME_START + pd.Timedelta(minutes=idx * WINDOW_SIZE_MINUTES)


def get_available_windows(data: Dict[str, pd.DataFrame]) -> List[str]:
    wins = set()
    for key in ["corrupted_audit", "network_flow", "host_metrics", "host_metric_events"]:
        df = data.get(key)
        if df is not None and "window_id" in df.columns:
            wins.update(df["window_id"].astype(str).tolist())
    return sorted(wins, key=lambda x: int(x.replace("W", "")) if x.startswith("W") else 999999)


def neighbor_windows(window_id: str, available_windows: List[str]) -> List[str]:
    available = set(available_windows)
    idx = int(window_id.replace("W", ""))
    out = []
    for j in [idx - 1, idx, idx + 1]:
        w = f"W{j:03d}"
        if w in available:
            out.append(w)
    return out


def is_external_ip(ip: str) -> bool:
    ip = safe_str(ip)
    if ip == "":
        return False
    return not ip.startswith("10.20.")


# =========================
# Load dataset
# =========================

def maybe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return read_csv(path)
    return pd.DataFrame()


def load_dataset() -> Dict[str, pd.DataFrame]:
    data = {
        "corrupted_audit": read_csv(DATA_ROOT / "corrupted" / "audit_log.csv"),
        "clean_audit": maybe_read_csv(DATA_ROOT / "clean" / "audit_log.csv"),
        "network_flow": read_csv(DATA_ROOT / "corrupted" / "network_flow.csv"),
        "host_metrics": read_csv(DATA_ROOT / "corrupted" / "host_metrics.csv"),
        "host_metric_events": read_csv(DATA_ROOT / "corrupted" / "host_metric_events.csv"),
        "window_labels": maybe_read_csv(DATA_ROOT / "labels" / "window_labels.csv"),
        "removed_events": maybe_read_csv(DATA_ROOT / "labels" / "removed_events.csv"),
        "entity_labels": maybe_read_csv(DATA_ROOT / "labels" / "entity_labels.csv"),
    }

    for key in ["corrupted_audit", "clean_audit"]:
        if len(data[key]) and "timestamp" in data[key].columns:
            data[key]["timestamp"] = pd.to_datetime(data[key]["timestamp"], errors="coerce")

    if "start_time" in data["network_flow"].columns:
        data["network_flow"]["start_time"] = pd.to_datetime(data["network_flow"]["start_time"], errors="coerce")
    if "end_time" in data["network_flow"].columns:
        data["network_flow"]["end_time"] = pd.to_datetime(data["network_flow"]["end_time"], errors="coerce")

    if "timestamp" in data["host_metrics"].columns:
        data["host_metrics"]["timestamp"] = pd.to_datetime(data["host_metrics"]["timestamp"], errors="coerce")

    if "start_time" in data["host_metric_events"].columns:
        data["host_metric_events"]["start_time"] = pd.to_datetime(data["host_metric_events"]["start_time"], errors="coerce")
    if "end_time" in data["host_metric_events"].columns:
        data["host_metric_events"]["end_time"] = pd.to_datetime(data["host_metric_events"]["end_time"], errors="coerce")

    return data


# =========================
# Candidate selection
# =========================

def parse_manual_windows(s: str) -> List[str]:
    return [x.strip() for x in safe_str(s).split(",") if x.strip()]


def select_candidate_windows(data: Dict[str, pd.DataFrame]) -> List[str]:
    if CANDIDATE_MODE == "manual":
        windows = parse_manual_windows(MANUAL_WINDOWS)
        if not windows:
            raise ValueError("No manual windows provided. Use --manual-windows W001,W002.")
        return windows

    if CANDIDATE_MODE == "gt_corrupted_windows_diagnostic":
        if len(data["window_labels"]) == 0:
            raise FileNotFoundError("window_labels.csv is required for diagnostic GT candidate mode.")
        wdf = data["window_labels"].copy()
        return wdf[wdf["is_corrupted"].astype(int) == 1]["window_id"].astype(str).tolist()

    score_path = STAGE1_OUT / "corrupted_window_scores.csv"
    if not score_path.exists():
        raise FileNotFoundError(
            f"Missing Stage-1 scores: {score_path}. Run Stage-1 first or use --candidate-mode manual."
        )

    scores = read_csv(score_path)
    if "corruption_score" not in scores.columns:
        raise ValueError(f"Stage-1 score file lacks corruption_score column: {score_path}")

    scores = scores.sort_values("corruption_score", ascending=False)

    if CANDIDATE_MODE == "stage1_topn":
        return scores.head(max(1, FALLBACK_TOP_N))["window_id"].astype(str).tolist()

    selected = scores[scores["corruption_score"].astype(float) > GAMMA]["window_id"].astype(str).tolist()
    if not selected:
        selected = scores.head(max(1, FALLBACK_TOP_N))["window_id"].astype(str).tolist()
    return selected


# =========================
# Context construction
# =========================

def compact_audit_rows(df: pd.DataFrame, window_id: str) -> List[str]:
    if len(df) == 0:
        return []

    tmp = df.copy()
    rel = text_col(tmp, "relation", "")
    rel_priority = {"connect": 4, "spawn": 4, "execute": 4, "write": 3, "chmod": 3, "unlink": 3, "rename": 3, "read": 1}
    tmp["priority"] = rel.map(lambda x: rel_priority.get(x, 0))
    if "timestamp" in tmp.columns:
        tmp = tmp.sort_values(["priority", "timestamp"], ascending=[False, True])
    else:
        tmp = tmp.sort_values(["priority"], ascending=[False])
    tmp = tmp.head(MAX_AUDIT_ROWS_PER_WINDOW)
    if "timestamp" in tmp.columns:
        tmp = tmp.sort_values("timestamp")

    rows = []
    for _, r in tmp.iterrows():
        rows.append(
            f"{safe_str(r.get('event_id'))} | {safe_str(r.get('timestamp'))} | {safe_str(r.get('window_id', window_id))} | "
            f"host={safe_str(r.get('host'))} user={safe_str(r.get('user'))} process={safe_str(r.get('process'))} | "
            f"{safe_str(r.get('src_entity'))} --{safe_str(r.get('relation'))}--> {safe_str(r.get('dst_entity'))} "
            f"({safe_str(r.get('dst_type'))}) | cmd={safe_str(r.get('command', ''))} | args={safe_str(r.get('args', ''))}"
        )
    if len(df) > MAX_AUDIT_ROWS_PER_WINDOW:
        rows.append(f"... {len(df) - MAX_AUDIT_ROWS_PER_WINDOW} additional audit rows omitted for {window_id}")
    return rows


def compact_flow_rows(df: pd.DataFrame, window_id: str) -> List[str]:
    if len(df) == 0:
        return []

    tmp = df.copy()
    tmp["total_bytes"] = numeric_col(tmp, "bytes_out", 0.0) + numeric_col(tmp, "bytes_in", 0.0)
    tmp["external"] = text_col(tmp, "dst_ip", "").map(lambda x: 0 if x.startswith("10.20.") else 1)
    sort_cols = ["external", "total_bytes"]
    asc = [False, False]
    if "start_time" in tmp.columns:
        sort_cols.append("start_time")
        asc.append(True)
    tmp = tmp.sort_values(sort_cols, ascending=asc).head(MAX_FLOW_ROWS_PER_WINDOW)
    if "start_time" in tmp.columns:
        tmp = tmp.sort_values("start_time")

    rows = []
    for _, r in tmp.iterrows():
        rows.append(
            f"{safe_str(r.get('flow_id'))} | {safe_str(r.get('start_time'))}~{safe_str(r.get('end_time'))} | {safe_str(r.get('window_id', window_id))} | "
            f"{safe_str(r.get('src_host'))} {safe_str(r.get('src_ip'))}:{safe_str(r.get('src_port'))} -> "
            f"{safe_str(r.get('dst_ip'))}:{safe_str(r.get('dst_port'))} | "
            f"service={safe_str(r.get('service'))} pattern={safe_str(r.get('flow_pattern'))} | "
            f"bytes_out={safe_str(r.get('bytes_out'))} bytes_in={safe_str(r.get('bytes_in'))} | "
            f"related_audit={safe_str(r.get('related_audit_events', ''))}"
        )
    if len(df) > MAX_FLOW_ROWS_PER_WINDOW:
        rows.append(f"... {len(df) - MAX_FLOW_ROWS_PER_WINDOW} additional flow rows omitted for {window_id}")
    return rows


def compact_metric_rows(df: pd.DataFrame, window_id: str) -> List[str]:
    if len(df) == 0:
        return []

    tmp = df.copy()
    tmp["activity"] = (
        numeric_col(tmp, "cpu_percent", 0.0)
        + numeric_col(tmp, "disk_write_bytes", 0.0) / 100000.0
        + numeric_col(tmp, "net_out_bytes", 0.0) / 100000.0
        + numeric_col(tmp, "file_open_count", 0.0) / 100.0
    )
    tmp = tmp.sort_values(["activity"], ascending=[False]).head(MAX_METRIC_ROWS_PER_WINDOW)
    sort_cols = [c for c in ["timestamp", "host"] if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols)

    rows = []
    for _, r in tmp.iterrows():
        rows.append(
            f"{safe_str(r.get('metric_id'))} | {safe_str(r.get('timestamp'))} | {safe_str(r.get('window_id', window_id))} | host={safe_str(r.get('host'))} | "
            f"cpu={safe_str(r.get('cpu_percent'))} mem={safe_str(r.get('mem_percent'))} "
            f"disk_read={safe_str(r.get('disk_read_bytes'))} disk_write={safe_str(r.get('disk_write_bytes'))} "
            f"net_in={safe_str(r.get('net_in_bytes'))} net_out={safe_str(r.get('net_out_bytes'))} "
            f"proc_count={safe_str(r.get('proc_count'))} open_files={safe_str(r.get('file_open_count'))} | "
            f"hint={safe_str(r.get('top_process_hint'))} | note={safe_str(r.get('metric_note'))}"
        )
    return rows


def compact_metric_event_rows(df: pd.DataFrame, window_id: str) -> List[str]:
    if len(df) == 0:
        return []

    tmp = df.copy()
    tmp["z_score_num"] = numeric_col(tmp, "z_score", 0.0)
    sort_cols = ["z_score_num"]
    asc = [False]
    if "start_time" in tmp.columns:
        sort_cols.append("start_time")
        asc.append(True)
    tmp = tmp.sort_values(sort_cols, ascending=asc).head(MAX_METRIC_EVENT_ROWS_PER_WINDOW)
    if "start_time" in tmp.columns:
        tmp = tmp.sort_values("start_time")

    rows = []
    for _, r in tmp.iterrows():
        rows.append(
            f"{safe_str(r.get('metric_event_id'))} | {safe_str(r.get('start_time'))}~{safe_str(r.get('end_time'))} | {safe_str(r.get('window_id', window_id))} | "
            f"host={safe_str(r.get('host'))} event_type={safe_str(r.get('event_type'))} "
            f"metric={safe_str(r.get('affected_metric'))} z={safe_str(r.get('z_score'))} | "
            f"hint={safe_str(r.get('related_hint'))} | desc={safe_str(r.get('description'))}"
        )
    return rows


def get_stage1_score(window_id: str) -> str:
    score_path = STAGE1_OUT / "corrupted_window_scores.csv"
    if not score_path.exists():
        return "Stage-1 score not available."
    scores = read_csv(score_path)
    row = scores[scores["window_id"].astype(str) == window_id]
    if len(row) == 0:
        return "Stage-1 score not available."
    r = row.iloc[0]
    fields = [
        "corruption_score",
        "transformer_corruption_score",
        "clue_score",
        "unsupported_flow_count",
        "unsupported_external_flow_count",
        "unsupported_metric_clue_count",
        "unsupported_metric_zsum",
    ]
    parts = []
    for f in fields:
        if f in r:
            parts.append(f"{f}={r[f]}")
    return ", ".join(parts)


def build_window_context(window_id: str, data: Dict[str, pd.DataFrame]) -> str:
    windows = neighbor_windows(window_id, get_available_windows(data))
    lines = []

    lines.append(f"# Target candidate corrupted window: {window_id}")
    lines.append(f"Neighbor context C_t = {windows}")
    lines.append(f"Stage-1 inconsistency clues for target: {get_stage1_score(window_id)}")
    lines.append("")
    lines.append("Recover missing audit events only for the TARGET window, not for neighboring windows.")
    lines.append("Neighboring windows are provided only for temporal continuity.")
    lines.append("")

    for w in windows:
        lines.append(f"## Window {w}")

        audit_w = data["corrupted_audit"][data["corrupted_audit"]["window_id"].astype(str) == w].copy()
        flow_w = data["network_flow"][data["network_flow"]["window_id"].astype(str) == w].copy()
        metric_w = data["host_metrics"][data["host_metrics"]["window_id"].astype(str) == w].copy()
        metric_event_w = data["host_metric_events"][data["host_metric_events"]["window_id"].astype(str) == w].copy()

        lines.append("")
        lines.append("### Observed corrupted audit log snippets")
        audit_rows = compact_audit_rows(audit_w, w)
        lines.extend([f"- {x}" for x in audit_rows] if audit_rows else ["- none"])

        lines.append("")
        lines.append("### Network flow side-view evidence")
        flow_rows = compact_flow_rows(flow_w, w)
        lines.extend([f"- {x}" for x in flow_rows] if flow_rows else ["- none"])

        lines.append("")
        lines.append("### Host metric samples")
        metric_rows = compact_metric_rows(metric_w, w)
        lines.extend([f"- {x}" for x in metric_rows] if metric_rows else ["- none"])

        lines.append("")
        lines.append("### Host metric clue events")
        metric_event_rows = compact_metric_event_rows(metric_event_w, w)
        lines.extend([f"- {x}" for x in metric_event_rows] if metric_event_rows else ["- none"])

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# =========================
# Prompt
# =========================

def build_system_prompt() -> str:
    relation_schema = json.dumps(ALLOWED_RELATIONS, indent=2)

    return f"""
You are Rinnegan's corrupted-window remediation module for APT detection.

Your task:
Recover missing AUDIT LOG events for one target candidate corrupted window, using:
1. observed corrupted audit logs,
2. network traffic side-view evidence,
3. host-metric side-view evidence,
4. local temporal context from W_(t-1), W_t, W_(t+1).

Methodology constraints:
- Side views are NOT complete event sources.
- Use side views to constrain and ground candidate audit events.
- Socket-related events can be directly supported by network flows.
- File/process events may be indirectly constrained by host-metric bursts and local log continuity.
- Do not take a hard intersection across modalities.
- Only output events that are temporally aligned, schema-compatible, and evidence-grounded.

Allowed provenance relation schema:
{relation_schema}

Entity types:
- process: process instance or process name, e.g., shell@host-1, updater@host-2
- file: absolute file path, e.g., /etc/cron.d/syswatch
- socket: ip:port, e.g., 198.51.100.77:443

Output requirements:
- Return JSON only.
- Do not include markdown.
- Do not include explanations outside JSON.
- Recover only missing attack-related audit events in the TARGET window.
- Do not output benign background events.
- Do not output events for neighboring windows.
- Prefer entities already present in the context; infer only when necessary.
- If evidence is insufficient, return an empty recovered_events list.

Each recovered event must contain:
{{
  "window_id": "Wxxx",
  "approx_timestamp": "YYYY-MM-DD HH:MM:SS",
  "host": "...",
  "user": "...",
  "src_entity": "...",
  "src_type": "process",
  "relation": "read|write|execute|spawn|connect|accept|chmod|unlink|rename",
  "dst_entity": "...",
  "dst_type": "process|file|socket",
  "command": "...",
  "args": "...",
  "supporting_side_views": ["traffic", "metrics", "log_context"],
  "supporting_observations": ["specific flow IDs, metric_event IDs, audit event IDs, or short evidence descriptions"],
  "confidence_rationale": "brief reason"
}}

Return exactly:
{{
  "recovered_events": [...]
}}
""".strip()


def build_user_prompt(window_id: str, context: str) -> str:
    return f"""
Recover missing audit events for target window {window_id}.

Remember:
- Only recover missing events in {window_id}.
- Use neighboring windows only as temporal context.
- Each event must be compatible with the provenance schema.
- Each event must be grounded in traffic, metrics, or log continuity evidence.
- Be conservative. Prefer 1 to 8 high-confidence events over many speculative events.

Window context:
{context}
""".strip()


# =========================
# LLM / cached / mock backends
# =========================

def get_client() -> Any:
    if OpenAI is None:
        raise ImportError("OpenAI SDK is not installed. Install it with: pip install openai")
    if not API_KEY:
        raise RuntimeError("Missing API key. Set RINNEGAN_OPENAI_API_KEY or pass --api-key.")
    if BASE_URL:
        return OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return OpenAI(api_key=API_KEY)


def parse_llm_json(text: str) -> Dict[str, Any]:
    text = safe_str(text).strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))

    if "recovered_events" not in obj:
        obj = {"recovered_events": []}

    if not isinstance(obj["recovered_events"], list):
        obj["recovered_events"] = []

    return obj


def call_openai_llm(client: Any, system_prompt: str, user_prompt: str, run_id: int) -> Dict[str, Any]:
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    response_format={"type": "json_object"},
                )
            except Exception:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                )

            content = resp.choices[0].message.content
            return parse_llm_json(content)

        except Exception as e:
            last_err = e
            print(f"[WARN] LLM call failed, run={run_id}, attempt={attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_err}")


def load_cached_response(window_id: str, run_id: int) -> Dict[str, Any]:
    candidates = [
        CACHE_DIR / f"{window_id}_run{run_id}.json",
        CACHE_DIR / window_id / f"run{run_id}.json",
        CACHE_DIR / f"{window_id}.json",
    ]

    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if "recovered_events" not in obj:
                obj = {"recovered_events": []}
            return obj

    print(f"[WARN] Cached response not found for {window_id} run {run_id}; returning empty list.")
    return {"recovered_events": []}


def mock_llm_response(window_id: str, data: Dict[str, pd.DataFrame], run_id: int) -> Dict[str, Any]:
    """
    Deterministic local backend for artifact smoke tests.
    It does not use ground-truth removed events. It only proposes conservative
    events from side-view evidence in the target window.
    """
    events = []
    win_start = infer_window_start(window_id)
    approx_ts = str(win_start + pd.Timedelta(minutes=5))

    flow_w = data["network_flow"][data["network_flow"]["window_id"].astype(str) == window_id].copy()
    if len(flow_w):
        flow_w["total_bytes"] = numeric_col(flow_w, "bytes_out", 0.0) + numeric_col(flow_w, "bytes_in", 0.0)
        flow_w["external"] = text_col(flow_w, "dst_ip", "").map(is_external_ip)
        flow_w = flow_w.sort_values(["external", "total_bytes"], ascending=[False, False])

        for _, r in flow_w.head(2).iterrows():
            dst_ip = safe_str(r.get("dst_ip"))
            dst_port = safe_str(r.get("dst_port"))
            if not dst_ip or not dst_port:
                continue
            host = safe_str(r.get("src_host"))
            src_proc = "network_client"
            pattern = safe_str(r.get("flow_pattern")).lower()
            if "beacon" in pattern:
                src_proc = "beacon"
            elif "payload" in pattern or "download" in pattern:
                src_proc = "downloader"
            events.append({
                "window_id": window_id,
                "approx_timestamp": approx_ts,
                "host": host,
                "user": "",
                "src_entity": f"{src_proc}@{host}" if host else src_proc,
                "src_type": "process",
                "relation": "connect",
                "dst_entity": f"{dst_ip}:{dst_port}",
                "dst_type": "socket",
                "command": "",
                "args": "",
                "supporting_side_views": ["traffic"],
                "supporting_observations": [f"flow_id={safe_str(r.get('flow_id'))}, pattern={safe_str(r.get('flow_pattern'))}"],
                "confidence_rationale": "Mock backend: external or high-volume network flow without relying on ground truth.",
            })

    metric_w = data["host_metric_events"][data["host_metric_events"]["window_id"].astype(str) == window_id].copy()
    if len(metric_w):
        metric_w["z_score_num"] = numeric_col(metric_w, "z_score", 0.0)
        metric_w = metric_w.sort_values("z_score_num", ascending=False)

        for _, r in metric_w.head(2).iterrows():
            hint = safe_str(r.get("related_hint")).lower()
            host = safe_str(r.get("host"))
            relation = "spawn"
            src = f"parent_proc@{host}" if host else "parent_proc"
            dst = f"child_proc@{host}" if host else "child_proc"
            dst_type = "process"

            if "cron" in hint:
                relation = "write"
                src = f"shell@{host}" if host else "shell"
                dst = "/etc/cron.d/syswatch"
                dst_type = "file"
            elif "sudo" in hint:
                relation = "execute"
                src = f"shell@{host}" if host else "shell"
                dst = "/usr/bin/sudo"
                dst_type = "file"

            events.append({
                "window_id": window_id,
                "approx_timestamp": approx_ts,
                "host": host,
                "user": "",
                "src_entity": src,
                "src_type": "process",
                "relation": relation,
                "dst_entity": dst,
                "dst_type": dst_type,
                "command": "",
                "args": "",
                "supporting_side_views": ["metrics"],
                "supporting_observations": [f"metric_event_id={safe_str(r.get('metric_event_id'))}, hint={safe_str(r.get('related_hint'))}, z={safe_str(r.get('z_score'))}"],
                "confidence_rationale": "Mock backend: host-metric anomaly suggests missing local audit activity.",
            })

    return {"recovered_events": events[:8]}


def call_backend(
    client: Any,
    window_id: str,
    data: Dict[str, pd.DataFrame],
    system_prompt: str,
    user_prompt: str,
    run_id: int,
) -> Dict[str, Any]:
    if LLM_BACKEND == "openai":
        return call_openai_llm(client, system_prompt, user_prompt, run_id)
    if LLM_BACKEND == "cached":
        return load_cached_response(window_id, run_id)
    return mock_llm_response(window_id, data, run_id)


# =========================
# Canonicalization and validation
# =========================

def infer_entity_type(entity: str, default: str = "") -> str:
    e = safe_str(entity)
    if re.match(r"^\d+\.\d+\.\d+\.\d+:\d+$", e):
        return "socket"
    if e.startswith("/") or e.startswith("./"):
        return "file"
    if default in {"process", "file", "socket"}:
        return default
    return "process"


def canonical_entity(entity: str, entity_type: str, host: str) -> str:
    entity = safe_str(entity).strip()
    entity_type = infer_entity_type(entity, entity_type)
    host = safe_str(host).strip()

    if entity_type == "socket":
        return entity.lower()

    if entity_type == "file":
        return entity.strip()

    e = entity.lower()
    if "@" in e:
        proc_part, host_part = e.split("@", 1)
        proc_name = proc_part.split(":")[0]
        return f"{proc_name}@{host_part}"

    proc_name = e.split(":")[0]
    if host:
        return f"{proc_name}@{host}"
    return proc_name


def normalize_side_views(x: Any) -> List[str]:
    if isinstance(x, list):
        vals = x
    elif isinstance(x, str):
        vals = re.split(r"[,;/\s]+", x)
    else:
        vals = []

    out = []
    for v in vals:
        v = safe_str(v).strip().lower()
        if v in {"traffic", "network", "network_flow", "flow"}:
            out.append("traffic")
        elif v in {"metric", "metrics", "host_metric", "host_metrics"}:
            out.append("metrics")
        elif v in {"log", "logs", "audit", "audit_log", "context", "log_context"}:
            out.append("log_context")

    return sorted(set(out))


def validate_schema(ev: Dict[str, Any]) -> Tuple[bool, str]:
    relation = safe_str(ev.get("relation")).strip()
    src_type = infer_entity_type(ev.get("src_entity", ""), safe_str(ev.get("src_type")))
    dst_type = infer_entity_type(ev.get("dst_entity", ""), safe_str(ev.get("dst_type")))

    if relation not in ALLOWED_RELATIONS:
        return False, f"invalid relation: {relation}"

    expected = ALLOWED_RELATIONS[relation]
    if [src_type, dst_type] != expected:
        return False, f"schema mismatch: {src_type} --{relation}--> {dst_type}, expected {expected}"

    return True, "ok"


def event_key(ev: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    window_id = safe_str(ev.get("window_id"))
    host = safe_str(ev.get("host"))
    relation = safe_str(ev.get("relation"))
    src_type = infer_entity_type(ev.get("src_entity", ""), safe_str(ev.get("src_type")))
    dst_type = infer_entity_type(ev.get("dst_entity", ""), safe_str(ev.get("dst_type")))
    src = canonical_entity(ev.get("src_entity", ""), src_type, host)
    dst = canonical_entity(ev.get("dst_entity", ""), dst_type, host)
    return window_id, host, src, relation, dst


def applicable_side_views(ev: Dict[str, Any]) -> List[str]:
    relation = safe_str(ev.get("relation"))
    dst_type = infer_entity_type(ev.get("dst_entity", ""), safe_str(ev.get("dst_type")))

    if relation in {"connect", "accept"} or dst_type == "socket":
        return ["traffic"]

    if relation in {"write", "read", "execute", "spawn", "chmod", "unlink", "rename"}:
        return ["metrics"]

    return ["traffic", "metrics"]


def compute_r_cross(ev: Dict[str, Any]) -> float:
    supported = set(normalize_side_views(ev.get("supporting_side_views", [])))
    app = set(applicable_side_views(ev))

    side_supported = supported.intersection(app)
    if app:
        if side_supported:
            return round(len(side_supported) / len(app), 4)
        if "log_context" in supported:
            return 0.3
        return 0.0

    return 0.0


def clean_recovered_event(ev: Dict[str, Any], target_window: str) -> Optional[Dict[str, Any]]:
    out = dict(ev)

    out["window_id"] = safe_str(out.get("window_id", target_window)) or target_window
    if out["window_id"] != target_window:
        return None

    out["host"] = safe_str(out.get("host"))
    out["user"] = safe_str(out.get("user"))
    out["src_entity"] = safe_str(out.get("src_entity"))
    out["src_type"] = infer_entity_type(out["src_entity"], safe_str(out.get("src_type")))
    out["relation"] = safe_str(out.get("relation")).strip()
    out["dst_entity"] = safe_str(out.get("dst_entity"))
    out["dst_type"] = infer_entity_type(out["dst_entity"], safe_str(out.get("dst_type")))
    out["command"] = safe_str(out.get("command"))
    out["args"] = safe_str(out.get("args"))
    out["confidence_rationale"] = safe_str(out.get("confidence_rationale"))

    approx_ts = safe_str(out.get("approx_timestamp"))
    if not approx_ts:
        out["approx_timestamp"] = str(infer_window_start(target_window) + pd.Timedelta(minutes=5))
    else:
        out["approx_timestamp"] = approx_ts

    out["supporting_side_views"] = normalize_side_views(out.get("supporting_side_views", []))

    obs = out.get("supporting_observations", [])
    if isinstance(obs, list):
        out["supporting_observations"] = [safe_str(x) for x in obs]
    elif isinstance(obs, str):
        out["supporting_observations"] = [obs]
    else:
        out["supporting_observations"] = []

    ok, reason = validate_schema(out)
    if not ok:
        out["validation_error"] = reason
        return None

    if not out["supporting_side_views"] and not out["supporting_observations"]:
        return None

    return out


# =========================
# Ground-truth matching for diagnostics
# =========================

def removed_event_key(row: pd.Series) -> Tuple[str, str, str, str, str]:
    window_id = safe_str(row.get("window_id"))
    host = safe_str(row.get("host"))
    relation = safe_str(row.get("relation"))
    src_type = infer_entity_type(row.get("src_entity", ""), safe_str(row.get("src_type")))
    dst_type = infer_entity_type(row.get("dst_entity", ""), safe_str(row.get("dst_type")))
    src = canonical_entity(row.get("src_entity", ""), src_type, host)
    dst = canonical_entity(row.get("dst_entity", ""), dst_type, host)
    return window_id, host, src, relation, dst


def build_removed_gt(data: Dict[str, pd.DataFrame]) -> Dict[Tuple[str, str, str, str, str], Dict[str, Any]]:
    gt = {}
    if len(data["removed_events"]) == 0:
        return gt

    for _, r in data["removed_events"].iterrows():
        k = removed_event_key(r)
        gt[k] = {
            "removed_event_id": safe_str(r.get("removed_event_id")),
            "original_event_id": safe_str(r.get("original_event_id")),
            "window_id": safe_str(r.get("window_id")),
            "host": safe_str(r.get("host")),
            "src_entity": safe_str(r.get("src_entity")),
            "relation": safe_str(r.get("relation")),
            "dst_entity": safe_str(r.get("dst_entity")),
            "stage": safe_str(r.get("stage")),
            "removed_reason": safe_str(r.get("removed_reason")),
        }
    return gt


# =========================
# Merge N runs and score
# =========================

def merge_and_score(all_events: List[Dict[str, Any]], data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)

    for ev in all_events:
        grouped[event_key(ev)].append(ev)

    gt = build_removed_gt(data)
    rows = []

    for idx, (k, evs) in enumerate(grouped.items(), start=1):
        base = evs[0]
        runs = sorted(set(int(e.get("_run_id", -1)) for e in evs if "_run_id" in e))
        n_self = len(runs)

        support_union = sorted(set(sum([normalize_side_views(e.get("supporting_side_views", [])) for e in evs], [])))

        obs_union = []
        for e in evs:
            obs_union.extend(e.get("supporting_observations", []))
        obs_union = sorted(set([safe_str(x) for x in obs_union if safe_str(x)]))

        r_cross = compute_r_cross({**base, "supporting_side_views": support_union})
        r_self = round(n_self / N_RUNS, 4) if N_RUNS else 0.0
        reliability = round(r_cross * r_self, 4)

        matched = gt.get(k)

        row = {
            "hypothesis_id": f"REC{idx:04d}",
            "window_id": k[0],
            "host": k[1],
            "src_entity": base["src_entity"],
            "src_type": base["src_type"],
            "relation": base["relation"],
            "dst_entity": base["dst_entity"],
            "dst_type": base["dst_type"],
            "canonical_src": k[2],
            "canonical_dst": k[4],
            "approx_timestamp": base["approx_timestamp"],
            "user": base.get("user", ""),
            "command": base.get("command", ""),
            "args": base.get("args", ""),
            "supporting_side_views": ";".join(support_union),
            "supporting_observations": " | ".join(obs_union[:20]),
            "n_self": n_self,
            "runs": ",".join(map(str, runs)),
            "R_cross": r_cross,
            "R_self": r_self,
            "reliability": reliability,
            "matched_removed_event": 1 if matched else 0,
            "matched_removed_event_id": matched["removed_event_id"] if matched else "",
            "matched_original_event_id": matched["original_event_id"] if matched else "",
            "matched_removed_reason": matched["removed_reason"] if matched else "",
            "confidence_rationale": base.get("confidence_rationale", ""),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda x: (x["window_id"], -x["reliability"], x["hypothesis_id"]))
    return rows


def evaluate_recovery(scored_rows: List[Dict[str, Any]], data: Dict[str, pd.DataFrame], candidate_windows: List[str]) -> Dict[str, Any]:
    if len(data["removed_events"]) == 0:
        return {
            "diagnostic_gt_available": False,
            "candidate_mode": CANDIDATE_MODE,
            "candidate_windows": candidate_windows,
            "N_runs": N_RUNS,
            "model": MODEL_NAME if LLM_BACKEND == "openai" else LLM_BACKEND,
            "predicted_hypotheses": len(scored_rows),
        }

    gt_df = data["removed_events"].copy()
    gt_df = gt_df[gt_df["window_id"].astype(str).isin(candidate_windows)]

    gt_ids = set(gt_df["removed_event_id"].astype(str).tolist())
    matched_ids = set(
        r["matched_removed_event_id"]
        for r in scored_rows
        if int(r.get("matched_removed_event", 0)) == 1 and r.get("matched_removed_event_id")
    )

    tp = len(matched_ids)
    fp = sum(1 for r in scored_rows if int(r.get("matched_removed_event", 0)) == 0)
    fn = len(gt_ids - matched_ids)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    pred_nodes = set()
    for r in scored_rows:
        pred_nodes.add(r["canonical_src"])
        pred_nodes.add(r["canonical_dst"])

    gt_nodes = set()
    for _, r in gt_df.iterrows():
        host = safe_str(r.get("host"))
        src_type = infer_entity_type(r.get("src_entity", ""), safe_str(r.get("src_type")))
        dst_type = infer_entity_type(r.get("dst_entity", ""), safe_str(r.get("dst_type")))
        gt_nodes.add(canonical_entity(r.get("src_entity", ""), src_type, host))
        gt_nodes.add(canonical_entity(r.get("dst_entity", ""), dst_type, host))

    node_jaccard = len(pred_nodes & gt_nodes) / len(pred_nodes | gt_nodes) if (pred_nodes | gt_nodes) else 0.0

    by_window = {}
    for w in candidate_windows:
        gt_w = set(gt_df[gt_df["window_id"].astype(str) == w]["removed_event_id"].astype(str).tolist())
        pred_w = [r for r in scored_rows if r["window_id"] == w]
        matched_w = set(r["matched_removed_event_id"] for r in pred_w if r.get("matched_removed_event_id"))
        tp_w = len(gt_w & matched_w)
        fp_w = sum(1 for r in pred_w if int(r.get("matched_removed_event", 0)) == 0)
        fn_w = len(gt_w - matched_w)
        p_w = tp_w / (tp_w + fp_w) if (tp_w + fp_w) else 0.0
        r_w = tp_w / (tp_w + fn_w) if (tp_w + fn_w) else 0.0
        f1_w = 2 * p_w * r_w / (p_w + r_w) if (p_w + r_w) else 0.0
        by_window[w] = {
            "gt_removed_events": len(gt_w),
            "predicted_hypotheses": len(pred_w),
            "tp": tp_w,
            "fp": fp_w,
            "fn": fn_w,
            "precision": round(p_w, 4),
            "recall": round(r_w, 4),
            "f1": round(f1_w, 4),
        }

    return {
        "diagnostic_gt_available": True,
        "candidate_mode": CANDIDATE_MODE,
        "candidate_windows": candidate_windows,
        "N_runs": N_RUNS,
        "backend": LLM_BACKEND,
        "model": MODEL_NAME if LLM_BACKEND == "openai" else LLM_BACKEND,
        "temperature": TEMPERATURE,
        "gt_removed_events": len(gt_ids),
        "predicted_hypotheses": len(scored_rows),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "node_jaccard": round(node_jaccard, 4),
        "matched_removed_event_ids": sorted(matched_ids),
        "missed_removed_event_ids": sorted(gt_ids - matched_ids),
        "by_window": by_window,
    }


# =========================
# Export recovered audit log
# =========================

def export_remediated_audit(scored_rows: List[Dict[str, Any]], data: Dict[str, pd.DataFrame], out_path: Path):
    audit = data["corrupted_audit"].copy()

    extra_cols = [
        "source",
        "hypothesis_id",
        "R_cross",
        "R_self",
        "reliability",
        "supporting_side_views",
        "supporting_observations",
    ]

    for c in extra_cols:
        if c not in audit.columns:
            audit[c] = ""

    audit["source"] = "observed"
    audit["hypothesis_id"] = ""
    audit["R_cross"] = 1.0
    audit["R_self"] = 1.0
    audit["reliability"] = 1.0

    recovered_rows = []
    for i, r in enumerate(scored_rows, start=1):
        recovered_rows.append({
            "event_id": f"REC_{r['window_id']}_{i:04d}",
            "timestamp": r["approx_timestamp"],
            "window_id": r["window_id"],
            "host": r["host"],
            "user": r.get("user", ""),
            "process": r["canonical_src"].split("@")[0] if "@" in r["canonical_src"] else r["canonical_src"],
            "pid": "",
            "ppid": "",
            "src_entity": r["src_entity"],
            "src_type": r["src_type"],
            "relation": r["relation"],
            "dst_entity": r["dst_entity"],
            "dst_type": r["dst_type"],
            "command": r.get("command", ""),
            "args": r.get("args", ""),
            "return_code": 0,
            "label": "recovered",
            "stage": "recovered",
            "attack_step": "recovered",
            "source": "llm_recovered" if LLM_BACKEND == "openai" else f"{LLM_BACKEND}_recovered",
            "hypothesis_id": r["hypothesis_id"],
            "R_cross": r["R_cross"],
            "R_self": r["R_self"],
            "reliability": r["reliability"],
            "supporting_side_views": r["supporting_side_views"],
            "supporting_observations": r["supporting_observations"],
        })

    remediated = pd.concat([audit, pd.DataFrame(recovered_rows)], ignore_index=True)
    if "timestamp" in remediated.columns:
        remediated["timestamp"] = pd.to_datetime(remediated["timestamp"], errors="coerce")
        remediated = remediated.sort_values(["timestamp", "event_id"])
    remediated.to_csv(out_path, index=False, encoding="utf-8-sig")


# =========================
# Main
# =========================

def main():
    args = parse_args()

    global DATA_ROOT, STAGE1_OUT, OUT_DIR, CACHE_DIR
    global CANDIDATE_MODE, MANUAL_WINDOWS, GAMMA, FALLBACK_TOP_N
    global LLM_BACKEND, MODEL_NAME, API_KEY, BASE_URL
    global N_RUNS, TEMPERATURE, MAX_RETRIES, RETRY_SLEEP_SECONDS

    DATA_ROOT = args.data_root
    STAGE1_OUT = args.stage1_out
    OUT_DIR = args.out_dir
    CACHE_DIR = args.cache_dir

    CANDIDATE_MODE = args.candidate_mode
    MANUAL_WINDOWS = args.manual_windows
    GAMMA = args.gamma
    FALLBACK_TOP_N = args.fallback_top_n

    LLM_BACKEND = args.llm_backend
    MODEL_NAME = args.model
    API_KEY = args.api_key
    BASE_URL = args.base_url
    N_RUNS = args.n_runs
    TEMPERATURE = args.temperature
    MAX_RETRIES = args.max_retries
    RETRY_SLEEP_SECONDS = args.retry_sleep

    ensure_dir(OUT_DIR)
    ensure_dir(OUT_DIR / "raw_runs")
    ensure_dir(OUT_DIR / "prompt_contexts")

    print("=== Rinnegan Stage-2: Corrupted-window remediation ===")
    print(f"DATA_ROOT = {DATA_ROOT}")
    print(f"STAGE1_OUT = {STAGE1_OUT}")
    print(f"OUT_DIR   = {OUT_DIR}")
    print(f"BACKEND   = {LLM_BACKEND}")
    print(f"MODEL     = {MODEL_NAME if LLM_BACKEND == 'openai' else LLM_BACKEND}")
    print(f"MODE      = {CANDIDATE_MODE}")
    print("")

    data = load_dataset()
    candidate_windows = select_candidate_windows(data)
    print(f"Candidate windows: {candidate_windows}")

    client = get_client() if LLM_BACKEND == "openai" else None
    system_prompt = build_system_prompt()

    all_cleaned_events = []

    for w in candidate_windows:
        print(f"\n[Window] {w}")
        context = build_window_context(w, data)
        user_prompt = build_user_prompt(w, context)

        write_text(OUT_DIR / "prompt_contexts" / f"{w}_context.txt", context)
        write_text(OUT_DIR / "prompt_contexts" / f"{w}_prompt.txt", user_prompt)

        for run_id in range(1, N_RUNS + 1):
            print(f"  remediation run {run_id}/{N_RUNS} ...")
            obj = call_backend(client, w, data, system_prompt, user_prompt, run_id)

            raw_path = OUT_DIR / "raw_runs" / f"{w}_run{run_id}.json"
            write_json(raw_path, obj)

            recovered = obj.get("recovered_events", [])
            kept = 0
            for ev in recovered:
                cleaned = clean_recovered_event(ev, target_window=w)
                if cleaned is None:
                    continue
                cleaned["_run_id"] = run_id
                cleaned["_target_window"] = w
                all_cleaned_events.append(cleaned)
                kept += 1

            print(f"    recovered={len(recovered)}, kept_after_schema={kept}")

    raw_events_path = OUT_DIR / "recovered_events_raw_validated.json"
    write_json(raw_events_path, all_cleaned_events)

    scored_rows = merge_and_score(all_cleaned_events, data)
    metrics = evaluate_recovery(scored_rows, data, candidate_windows)

    scored_df = pd.DataFrame(scored_rows)
    scored_csv = OUT_DIR / "recovered_events_scored.csv"
    scored_df.to_csv(scored_csv, index=False, encoding="utf-8-sig")

    metrics_path = OUT_DIR / "remediation_metrics.json"
    write_json(metrics_path, metrics)

    export_remediated_audit(scored_rows, data, OUT_DIR / "remediated_audit_log.csv")

    lines = []
    lines.append("Rinnegan Stage-2: Corrupted-window remediation")
    lines.append("")
    lines.append(f"candidate_mode: {CANDIDATE_MODE}")
    lines.append(f"candidate_windows: {', '.join(candidate_windows)}")
    lines.append(f"backend: {LLM_BACKEND}")
    lines.append(f"model: {MODEL_NAME if LLM_BACKEND == 'openai' else LLM_BACKEND}")
    lines.append(f"N_runs: {N_RUNS}")
    lines.append(f"temperature: {TEMPERATURE}")
    lines.append("")
    lines.append("Remediation diagnostics")
    for k in [
        "diagnostic_gt_available",
        "gt_removed_events",
        "predicted_hypotheses",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "node_jaccard",
    ]:
        if k in metrics:
            lines.append(f"  {k}: {metrics[k]}")
    lines.append("")

    if "by_window" in metrics:
        lines.append("By-window diagnostics")
        for w, m in metrics["by_window"].items():
            lines.append(
                f"  {w}: gt={m['gt_removed_events']}, pred={m['predicted_hypotheses']}, "
                f"tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, "
                f"P={m['precision']}, R={m['recall']}, F1={m['f1']}"
            )
        lines.append("")

    if "matched_removed_event_ids" in metrics:
        lines.append("Matched removed events")
        lines.append("  " + ", ".join(metrics["matched_removed_event_ids"]))
        lines.append("")
        lines.append("Missed removed events")
        lines.append("  " + ", ".join(metrics["missed_removed_event_ids"]))
        lines.append("")

    lines.append("Outputs")
    lines.append(f"  {scored_csv}")
    lines.append(f"  {metrics_path}")
    lines.append(f"  {OUT_DIR / 'remediated_audit_log.csv'}")
    lines.append(f"  {OUT_DIR / 'prompt_contexts'}")
    lines.append(f"  {OUT_DIR / 'raw_runs'}")

    summary = "\n".join(lines)
    write_text(OUT_DIR / "remediation_summary.txt", summary)

    print("")
    print(summary)
    print("")
    print("[DONE] Stage-2 remediation finished.")


if __name__ == "__main__":
    main()