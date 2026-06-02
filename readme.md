# Rinnegan Artifact

This repository provides an anonymized implementation of **Rinnegan**, a multi-stage pipeline for robust APT detection under attack-related log removal.

Rinnegan addresses the case where critical audit-log events are selectively removed during an APT campaign. The pipeline uses side-view evidence from network traffic and host metrics to discover suspicious corrupted windows, recover missing audit-event hypotheses, initialize reliability-aware provenance graphs, and perform node-level APT detection with a reliability-aware graph autoencoder.

The artifact is designed for inspection and smoke-test execution on the provided desensitized OfficeFog dataset subset.

---

## Pipeline Overview

Rinnegan consists of six stages:

1. **Corrupted-Window Discovery**  
   Detect candidate windows where audit logs are inconsistent with network and host-metric side views.

2. **Corrupted-Window Remediation**  
   Recover missing audit-event hypotheses from local temporal context and side-view evidence.

3. **Reliability-Aware Graph Initialization**  
   Build clean, corrupted, and remediated provenance graphs with reliability annotations.

4. **R-GAE Data Preparation**  
   Prepare benign train/validation graphs and test graphs for reliability-aware detection.

5. **R-GAE Training and Detection**  
   Train a reliability-aware graph autoencoder on benign graphs and evaluate node-level anomaly scores.

6. **Detection Analysis**  
   Analyze clean, corrupted, and remediated detection results.

---

## Repository Structure

```text
.
├── OfficeFog_desensitized_version/
│   └── full-dataset/
│       ├── clean/
│       ├── corrupted/
│       └── labels/
│
├── requirements.txt
├── readme.md
│
├── run_rinnegan_stage1_discovery.py
├── run_rinnegan_stage2_remediation.py
├── run_rinnegan_stage3_graph_initialization.py
├── run_rinnegan_stage4_prepare_rgae_data.py
├── run_rinnegan_stage5_train_rgae.py
└── run_rinnegan_stage6_detection_analysis.py
```

The repository contains only anonymized scripts and desensitized data for artifact evaluation.

---

## Environment Setup

Python 3.10 or newer is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

Run the following commands from the repository root.

### Stage 1: Corrupted-Window Discovery

```bash
python run_rinnegan_stage1_discovery.py ^
  --data-root OfficeFog_desensitized_version/full-dataset ^
  --out-dir outputs/stage1_discovery ^
  --device cpu
```

Linux/macOS:

```bash
python run_rinnegan_stage1_discovery.py \
  --data-root OfficeFog_desensitized_version/full-dataset \
  --out-dir outputs/stage1_discovery \
  --device cpu
```

---

### Stage 2: Corrupted-Window Remediation

The default artifact setting uses a local mock backend, so no API key is required.

```bash
python run_rinnegan_stage2_remediation.py ^
  --data-root OfficeFog_desensitized_version/full-dataset ^
  --stage1-out outputs/stage1_discovery ^
  --out-dir outputs/stage2_remediation ^
  --llm-backend mock
```

Linux/macOS:

```bash
python run_rinnegan_stage2_remediation.py \
  --data-root OfficeFog_desensitized_version/full-dataset \
  --stage1-out outputs/stage1_discovery \
  --out-dir outputs/stage2_remediation \
  --llm-backend mock
```

To use an OpenAI-compatible backend:

```bash
set RINNEGAN_OPENAI_API_KEY=your_api_key

python run_rinnegan_stage2_remediation.py ^
  --data-root OfficeFog_desensitized_version/full-dataset ^
  --stage1-out outputs/stage1_discovery ^
  --out-dir outputs/stage2_remediation ^
  --llm-backend openai ^
  --model gpt-4o-mini
```

Linux/macOS:

```bash
export RINNEGAN_OPENAI_API_KEY=your_api_key

python run_rinnegan_stage2_remediation.py \
  --data-root OfficeFog_desensitized_version/full-dataset \
  --stage1-out outputs/stage1_discovery \
  --out-dir outputs/stage2_remediation \
  --llm-backend openai \
  --model gpt-4o-mini
```

---

### Stage 3: Reliability-Aware Graph Initialization

```bash
python run_rinnegan_stage3_graph_initialization.py ^
  --data-root OfficeFog_desensitized_version/full-dataset ^
  --stage2-out outputs/stage2_remediation ^
  --out-dir outputs/stage3_graph_init
```

Linux/macOS:

```bash
python run_rinnegan_stage3_graph_initialization.py \
  --data-root OfficeFog_desensitized_version/full-dataset \
  --stage2-out outputs/stage2_remediation \
  --out-dir outputs/stage3_graph_init
```

---

### Stage 4: Prepare R-GAE Data

```bash
python run_rinnegan_stage4_prepare_rgae_data.py ^
  --data-root OfficeFog_desensitized_version/full-dataset ^
  --stage3-root outputs/stage3_graph_init ^
  --out-dir outputs/stage4_rgae_data
```

Linux/macOS:

```bash
python run_rinnegan_stage4_prepare_rgae_data.py \
  --data-root OfficeFog_desensitized_version/full-dataset \
  --stage3-root outputs/stage3_graph_init \
  --out-dir outputs/stage4_rgae_data
```

---

### Stage 5: Train and Evaluate R-GAE

```bash
python run_rinnegan_stage5_train_rgae.py ^
  --stage4-root outputs/stage4_rgae_data ^
  --out-dir outputs/stage5_rgae_detection ^
  --device cpu
```

Linux/macOS:

```bash
python run_rinnegan_stage5_train_rgae.py \
  --stage4-root outputs/stage4_rgae_data \
  --out-dir outputs/stage5_rgae_detection \
  --device cpu
```

GPU execution can be enabled with:

```bash
python run_rinnegan_stage5_train_rgae.py --device cuda
```

or automatic device selection:

```bash
python run_rinnegan_stage5_train_rgae.py --device auto
```

---

### Stage 6: Detection Analysis

```bash
python run_rinnegan_stage6_detection_analysis.py ^
  --stage5-root outputs/stage5_rgae_detection ^
  --stage4-root outputs/stage4_rgae_data ^
  --out-dir outputs/stage6_detection_analysis
```

Linux/macOS:

```bash
python run_rinnegan_stage6_detection_analysis.py \
  --stage5-root outputs/stage5_rgae_detection \
  --stage4-root outputs/stage4_rgae_data \
  --out-dir outputs/stage6_detection_analysis
```

---

## Data Notes

The included OfficeFog data is a desensitized subset for artifact review. It contains aligned audit logs, network-flow records, host-metric records, and diagnostic labels.

The artifact does not include non-anonymized raw logs, private identifiers, local machine paths, API keys, or operational credentials.

---

## Reproducibility Notes

The default Stage-2 setting uses a deterministic local mock backend to make the pipeline runnable without external LLM access.

When using an external LLM backend, results may vary depending on the model, temperature, API implementation, and decoding behavior.

Random seeds are fixed by default in the main training and scoring stages, but exact numerical results may still vary slightly across hardware and library versions.

---

## Expected Qualitative Behavior

The expected trend is:

```text
complete logs    > corrupted logs
remediated logs  > corrupted logs
```

In other words, attack-related log removal should degrade detection, while reliability-weighted remediation should partially recover the missing-evidence effect.

---

## License and Anonymity

This repository is provided as an anonymized research artifact for review. Identifying metadata, private paths, raw operational logs, and credentials have been removed.