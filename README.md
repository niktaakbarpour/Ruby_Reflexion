# RAMP: A Lightweight Multi-Agent Program Repair for Ruby with Feedback-Driven LLM

> Automated Program Repair (APR) has advanced rapidly with Large Language Models (LLMs), but most existing methods remain computationally expensive, and focused on a small set of languages. Ruby, despite its widespread use in web development and the persistent challenges faced by its developers, has received little attention in APR research.
In this paper, we introduce RAMP, a novel lightweight framework that formulates program repair as a feedback-driven, iterative process for Ruby. RAMP employs a team of collaborative agents that generate targeted tests, reflect on errors, and refine candidate fixes until a correct solution is found. Unlike prior approaches, RAMP is designed to avoid reliance on large multilingual repair databases or costly fine-tuning, instead operating directly on Ruby through lightweight prompting and test-driven feedback.
Evaluation on the \textsc{XCodeEval} benchmark shows that RAMP achieves a Pass@1 of 67\% on Ruby, outperforming prior approaches. RAMP converges quickly within five iterations, and ablation studies confirm that test generation and self-reflection are key drivers of its performance. Further analysis shows that RAMP is particularly effective at repairing wrong answers, compilation errors, and runtime errors.
Our approach provides new insights into multi-agent repair strategies, and establishes a foundation for extending LLM-based debugging tools to under-studied languages.

---

## Table of Contents

* [Overview](#overview)
* [Quick Start](#quick-start)
* [Configuration](#configuration)
* [Repository Layout](#repository-layout)
* [Running Experiments](#running-experiments)
* [Results & Logs](#results--logs)
* [Reproducing Figures](#reproducing-figures)

---

## Overview

This repository implements a **reflection-augmented multi-agent pipeline** for **Automated Program Repair (APR)** on Ruby tasks. It includes:

* Core framework with modular executors, generators, strategies, and prompts.
* Baseline methods for comparison (e.g., ChatRepair, LANTERN, Self-Collaboration, Self-Planning).
* Benchmarks for Ruby APR evaluation.
* Analysis scripts to evaluate results and generate publication-quality plots.

## Quick Start

### Setup

```bash
# create & activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install project dependencies
pip install -r requirements.txt
```

---

## Configuration

* **RAMP** is launched via shell scripts in `config/`:

  * `config/run_ramp.sh` (Ruby)
  * `config/run_cpp.sh` (C++)
* **Baselines** each have their own entry points and **do not** use `main.py`:

  * See the [Baselines](#baselines) section for exact commands.
* `config/` also contains YAML files used by specific components (e.g., **ChatRepair** uses `config/tr_chatreapir.yaml`).
* Dataset/model settings (e.g., `base_dir`, `dataset_path`, decoding params) are specified inside each method’s config or CLI flags as applicable.

## Repository Layout

```
analysis/     # Scripts for analyzing results and generating plots (heatmaps, bar charts, venn diagrams, etc.)

baselines/    # Implementations of baselines (ChatRepair, LANTERN, Self-Collaboration, Self-Planning, etc.)

benchmarks/   # Benchmark datasets and utilities for APR (Ruby-only sets, 10% sampled subsets, validation set, etc.)

config/       # Configurations for training/experiments (YAML + helper shell scripts)

executors/    # Language-specific executors

generators/   # Code/test generators and prompt builders

strategies/   # High-level repair strategies and ablations

prompts/      # Prompt templates for Reflexion variations

images/       # Figures for paper/README
logs/         # Logs from experiment runs
results/      # Saved experiment outputs and aggregated results
main.py       # Entry point
utils.py      # Utility functions
requirements.txt  # Python dependencies
```

---

## Running Experiments

### RAMP (Ruby/C++)

You can run RAMP directly with `main.py` and the appropriate arguments:

````bash
python main.py \
  --run_name \
  --root_dir \
  --dataset_path \
  --strategy: repair strategy. Options include:
  - `simple`
  - `few_shot`
  - `reflexion`
  - `first_refl_omission`
  - `self_refl_omission`
  - `refl_omission`
  - `test_gen_omission`
  - `infer_spec`
  - `IO_remove`
  - `add_buggy`
  - `time_memory_remove`

- `--language`: target language (`rb`, `cpp`)
- `--model`: model name (e.g., `gpt-4`, `deepseek-coder`)
- `--model_path`: local model path
- `--pass_at_k`: "1"
- `--max_iters`: maximum iterations for repair loop
- `--verbose`: enable detailed logs

You may also use the helper shell scripts:
```bash
bash config/run_ramp.sh   # Ruby RAMP
bash config/run_cpp.sh    # C++ RAMP
````

### Baselines

**Docker Engine**

* Install Docker Engine (Docker CE) on your machine.

**Python environment** (example for LANTERN)

```bash
conda create -n lantern python=3.9.2 -y
conda activate lantern
cd baselines/LANTERN
```

**ExecEval (xCodeEval execution engine)**

```bash
git clone https://github.com/ntunlp/ExecEval
cd ExecEval
docker build . -t exec-eval:1.0
# expose the service on port 5000 (adjust NUM_WORKERS as needed)
docker run -it -p 5000:5000 -e NUM_WORKERS=37 exec-eval:1.0
```

> Ensure your baseline configs point to the running ExecEval server.

**Running baselines**

* **ChatRepair**

```bash
python ramp/baselines/python main.py --config config/tr_reasoning.yaml
```

* **ChatRepair**

```bash
python ramp/baselines/chat_repair/main.py --config ramp/config/tr_chatrepair.yaml
```

* **Self-Planning**

```bash
python ramp/baselines/self-planning/planning.py --base-dir <result directory> --dataset-path <xcodeeval_dataset>
python ramp/baselines/self-planning/implementation.py --base-dir <result directory>
```

* **Self-Collaboration**

```bash
bash ramp/baselines/self-collaboration/run.sh
bash ramp/baselines/self-collaboration/evaluate.sh
```

### Notes on Baselines

* For baselines, we relied on the replication package of LANTERN but made two modifications:

  * Changed the code to compute Pass@1 instead of the default metric.
  * Configured it to use a local model rather than external API keys.

* For Pass@1 calculation, we used the field ```bug_code_uid``` to identify unique buggy programs.


### Dataset & Model Configuration

* **Choose a dataset** from `benchmarks/` and reference it in your config (e.g., `ramp_ruby_10percent.jsonl`, `ruby_only.jsonl`).
* Set the following in your config files:

  * `base_dir`: working directory for artifacts/results
  * `dataset_path`: path to selected benchmark JSONL/CSV
  * **LLM settings**: model name, decoding params (temperature, top\_p), API keys/endpoints if applicable

## Results & Logs

* Results: JSON/CSV summaries in `results/`
* Logs: Structured logs in `logs/`
* Benchmarks: located in `benchmarks/`

## Reproducing Figures

Scripts in `analysis/` generate plots for experiments. Examples:

```bash
python analysis/plot_pass1_over_iterations.py \
  --input results/summaries.csv --out images/pass1_vs_iter.png

python analysis/heatmap_difficulty.py \
  --input results/difficulty.csv --out images/difficulty_heatmap.png
```

## Acknowledgments

Thanks to prior works including ChatRepair, LANTERN, Self-Collaboration, Self-Planning, and ExecEval.