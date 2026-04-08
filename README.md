---
title: CSVAnalystEnv
sdk: docker
app_port: 8000
tags:
  - openenv
  - tabular-reasoning
  - benchmark
  - reinforcement-learning
---

# CSVAnalystEnv

An **OpenEnv-compatible** reinforcement-learning environment for evaluating data-analysis agents on structured CSV reasoning tasks.

In each episode, an agent receives a question about a CSV dataset and must use a **constrained action set** to inspect columns, filter data, compute aggregates, and submit a final answer. The environment rewards correctness, penalizes invalid actions, and encourages efficient tool use.

---

## 💎 Why This is a Great Environment

- **Reproducible Benchmark**: Fixed dataset + 13 verified tasks for systematic agent evaluation.
- **Constrained Tool Use**: Agents use discrete data tools (`filter`, `groupby`, `aggregate`) rather than arbitrary code or SQL.
- **Exact Programmatic Grading**: Normalized `0.0–1.0` scores with numeric tolerance.
- **OpenEnv Standards**: Fully compliant `reset/step/state` HTTP interface with typed Pydantic models.

---

## 🏛️ OpenEnv Integration

| Feature | Implementation |
|---------|---------------|
| **Interface** | `reset(task_id?)`, `step(CSVAction)`, `state()` |
| **Transport** | JSON over HTTP (FastAPI) |
| **Deployment** | Docker-ready (Hugging Face Spaces compatible) |
| **Typed models** | `CSVAction`, `CSVObservation`, `CSVState` (Pydantic v2) |
| **Reward signal** | Scalar per-step: base cost, success, invalid penalty |
| **Episode semantics** | One question per episode, 8-step budget, deterministic grader |
| **Score range** | `0.0` (incorrect) – `1.0` (correct) |

---

## ⚖️ Judging Notes

| Criterion | How we satisfy it |
|-----------|------------------|
| Task Clarity | 13 tasks targeting specific reasoning types: count, aggregate, groupby, lookup |
| Tool-use Dynamics | 7 discrete actions, each with typed parameters and reward feedback |
| Reproducibility | Single bundled `orders.csv`, fixed task bank, deterministic grader |
| Benchmarking | `run_eval.py` for aggregate reports, `baseline_inference.py` for LLM evaluation |
| OpenEnv Compliance | `openenv.yaml` manifest, `reset/step/state` HTTP API, Docker deployment |

---

## 🚀 Quick Start for Judges

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run a single demo episode
```bash
python demo_run.py
```

### 3. Run the full benchmark over all 13 tasks
```bash
python run_eval.py
```

### 4. Start the HTTP server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 5. Run LLM baseline evaluation
```bash
export HF_TOKEN=hf_your_token_here
python baseline_inference.py
```
> If `HF_TOKEN` is not set, the script runs in **heuristic mode** (no LLM calls) to demonstrate the evaluation loop.

---

## 📊 Baseline Performance

All results produced by `baseline_inference.py` against a live local server.

| Model | Accuracy | Avg Score | Avg Steps | Invalid Actions |
|-------|----------|-----------|-----------|-----------------|
| Heuristic (no LLM) | 0% (0/13) | 0.000 | 3.0 | 0 |
| LLM (Llama-3-8B-Instruct) | 7.7% (1/13) | 0.077 | 6.1 | 0 |

> The heuristic baseline (`list_columns → preview_rows → submit "unknown"`) scores 0% by design — it confirms the grading loop is live and working. Set `HF_TOKEN` to run the LLM baseline.

---

## 📂 Project Structure

```
csv-analyst-env/
├── openenv.yaml            # OpenEnv packaging manifest ✅
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── orders.csv          # 30-row sample dataset
├── tasks/
│   └── tasks.json          # 13 tasks (easy/medium/hard) with ground-truth
├── models.py               # Typed Pydantic models (Action, Observation, State)
├── environment.py          # Core environment — reset / step / state
├── grader.py               # Programmatic grading + normalized 0.0–1.0 scores
├── demo_run.py             # Single episode walkthrough
├── run_eval.py             # Full benchmark script (all 13 tasks)
├── baseline_inference.py   # LLM baseline via HF Inference API
└── server/
    ├── app.py              # FastAPI HTTP wrapper
    └── Dockerfile          # Container deployment
```

---

## 🗂️ Task Difficulty Breakdown

| Difficulty | Count | Example Question |
|------------|-------|-----------------|
| **Easy** | 6 | "How many total orders are delivered?" |
| **Medium** | 5 | "Which category has the highest total quantity sold?" |
| **Hard** | 2 | "Which customer placed the highest unit_price order?" |

---

## 📊 Reward Design

| Event | Reward |
|-------|--------|
| Base step cost | `−0.05` |
| `groupby_aggregate` step cost | `−0.08` |
| Invalid action penalty | `−0.20` (additional) |
| Correct answer | `+1.00` |
| Incorrect answer | `−1.00` |
| Efficiency bonus (≤ 4 steps) | `+0.20` |
| Forced termination (max steps) | `−0.50` |

---

## 🔗 HTTP API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome + API map |
| `/health` | GET | Health check |
| `/tasks` | GET | Task list (questions only, no answers) |
| `/reset` | POST | Start episode `{"task_id": "q1"}` |
| `/step` | POST | Submit a `CSVAction` |
| `/state` | GET | Current episode metadata |

### Example cURL

```bash
# Reset to task q1
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "q1"}'

# List columns
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "list_columns"}'

# Submit answer
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "submit_answer", "answer": "8"}'
```

---

## 🐳 Docker

```bash
docker build -t csv-analyst-env -f server/Dockerfile .
docker run -p 8000:8000 csv-analyst-env
```

---

## 🛠️ Tech Stack

- **Python 3.11+**
- **Pydantic v2** — Typed models
- **Pandas** — Data operations
- **FastAPI + Uvicorn** — HTTP serving
- **Hugging Face Inference API** — LLM baseline

---

## 📜 License

MIT
