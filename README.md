# CSVAnalystEnv

An **OpenEnv-compatible** reinforcement-learning environment for evaluating data-analysis agents on structured CSV reasoning tasks.

In each episode, an agent receives a question about a CSV dataset and must use a **constrained action set** to inspect columns, filter data, compute aggregates, and submit a final answer. The environment rewards correctness, penalizes invalid actions, and encourages efficient tool use.

---

## 💎 Why This is a Great Environment
- **Reproducible Benchmark**: Unlike open-ended chatbot UIs, this provides a fixed dataset and 13 verified tasks for systematic agent evaluation.
- **Constrained Tool Use**: Agents must learn to use discrete data tools (`filter`, `groupby`, `aggregate`) rather than generating arbitrary code/SQL.
- **Exact Programmatic Grading**: Success metrics are computed automatically with numeric tolerance, ideal for training Reward Models.
- **OpenEnv Standards**: Built from the ground up to follow the `reset/step/state` pattern with typed Pydantic models.

---

## 🏛️ OpenEnv Integration
This environment follows the standardized **OpenEnv pattern** for agentic execution environments:

| Feature | implementation |
|---------|----------------|
| **Interface** | `reset(task_id?)`, `step(action)`, `state()` |
| **Transport** | JSON over HTTP (FastAPI) |
| **Deployment** | Docker-ready for isolated evaluation |
| **Typed models** | `CSVAction`, `CSVObservation`, `CSVState` |
| **Rewards** | base step cost, success reward, invalid action penalty |

---

## ⚖️ Judging Notes
Explicitly mapped to the hackathon criteria:
- **Task Clarity**: 13 predefined tasks targeting specific reasoning types (counting, sums, group-by).
- **Tool-use Dynamics**: A compact set of 7 high-level data operations.
- **Reproduction**: A single bundled `orders.csv` ensures all testers see the same results.
- **Benchmarking**: Includes `run_eval.py` to generate aggregate accuracy/efficiency reports.

---

## 🚀 Quick Start for Judges

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run an end-to-end Demo episode**:
   ```bash
   python demo_run.py
   ```

3. **Generate a full Benchmark Report over all 13 tasks**:
   ```bash
   python run_eval.py
   ```

4. **Start the HTTP Server (OpenEnv API)**:
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```

---

## 📂 Project Structure
```
csv-analyst-env/
├── README.md
├── requirements.txt
├── data/
│   └── orders.csv          # 30-row sample dataset
├── tasks/
│   └── tasks.json          # 13 predefined tasks with ground-truth answers
├── models.py               # Typed Pydantic models (Action, Observation, State)
├── environment.py          # Core environment logic
├── grader.py               # Programmatic grading & batch evaluation
├── demo_run.py             # Single episode walkthrough
├── run_eval.py             # FULL BENCHMARK script
└── server/
    ├── app.py              # FastAPI HTTP wrapper
    └── Dockerfile          # Container deployment
```

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

## 🔗 HTTP API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/`       | GET    | Welcome message & documentation link |
| `/health` | GET    | Health check |
| `/tasks`  | GET    | List available tasks (questions/IDs only) |
| `/reset`  | POST   | Start episode (`{"task_id": "q1"}`) |
| `/step`   | POST   | Submit action (CSVAction body) |
| `/state`  | GET    | View current episode metadata |

---

## 🛠️ tech Stack
- **Python 3.11+**
- **Pydantic v2** — Typed models
- **Pandas** — Data operations
- **FastAPI + Uvicorn** — HTTP serving

---

## 📜 License
MIT
