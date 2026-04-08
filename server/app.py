"""
FastAPI server wrapping CSVAnalystEnv for HTTP-based agent interaction.

Endpoints
---------
POST /reset          — Start a new episode (optionally specify task_id).
POST /step           — Submit an action and receive an observation.
GET  /state          — Retrieve current episode metadata.
GET  /tasks          — List all available tasks.
GET  /health         — Health check.
"""

import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from models import CSVAction, CSVObservation, CSVState

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "orders.csv")
TASKS_PATH = os.path.join(BASE_DIR, "tasks", "tasks.json")

# Lazy import to avoid circular issues when running from different locations
import sys
sys.path.insert(0, BASE_DIR)
from environment import CSVAnalystEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------

with open(TASKS_PATH) as f:
    TASKS = json.load(f)

env = CSVAnalystEnv(CSV_PATH, TASKS)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CSVAnalystEnv API",
    description=(
        "An OpenEnv-compatible HTTP interface for the CSV Analyst environment. "
        "Agents interact via reset → step → step → … → done."
    ),
    version="0.1.0",
)


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

from fastapi.responses import HTMLResponse

@app.get("/", include_in_schema=False)
def root():
    """Display a premium landing page for judges and users."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>CSVAnalystEnv – OpenEnv Environment</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        body {
          margin: 0;
          padding: 0;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #050816;
          color: #e5e7eb;
          display: flex;
          min-height: 100vh;
          align-items: center;
          justify-content: center;
        }
        .card {
          max-width: 720px;
          width: 90%;
          padding: 32px;
          border-radius: 20px;
          background: radial-gradient(circle at top left, #111827 0, #020617 45%, #000000 100%);
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
          border: 1px solid rgba(148,163,184,0.2);
          backdrop-filter: blur(10px);
        }
        h1 {
          margin: 0 0 8px;
          font-size: 2rem;
          font-weight: 800;
          background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          letter-spacing: -0.02em;
        }
        .tagline {
          margin: 0 0 24px;
          color: #94a3b8;
          font-size: 1.1rem;
          line-height: 1.5;
        }
        .pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 4px 12px;
          border-radius: 999px;
          font-size: 0.8rem;
          font-weight: 600;
          background: rgba(34, 197, 94, 0.1);
          border: 1px solid rgba(34, 197, 94, 0.3);
          color: #4ade80;
          margin-bottom: 16px;
        }
        .pill span {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #22c55e;
          box-shadow: 0 0 10px #22c55e;
        }
        .section-title {
          margin: 24px 0 8px;
          font-size: 0.85rem;
          text-transform: uppercase;
          font-weight: 700;
          letter-spacing: 0.1em;
          color: #64748b;
        }
        ul {
          margin: 0;
          padding-left: 1.2rem;
          font-size: 0.95rem;
          color: #cbd5e1;
          list-style-type: square;
        }
        li { margin-bottom: 8px; }
        code {
          background: rgba(255, 255, 255, 0.1);
          padding: 2px 6px;
          border-radius: 4px;
          font-family: ui-monospace, monospace;
          color: #38bdf8;
        }
        .links {
          display: flex;
          flex-wrap: wrap;
          gap: 12px;
          margin-top: 32px;
        }
        .btn {
          display: inline-flex;
          align-items: center;
          padding: 10px 20px;
          border-radius: 12px;
          font-size: 0.9rem;
          font-weight: 600;
          text-decoration: none;
          transition: all 0.2s ease;
        }
        .btn-primary {
          background: #38bdf8;
          color: #020617;
        }
        .btn-primary:hover {
          background: #7dd3fc;
          transform: translateY(-1px);
        }
        .btn-secondary {
          background: rgba(255, 255, 255, 0.05);
          color: #fff;
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .btn-secondary:hover {
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.2);
        }
        .meta {
          margin-top: 32px;
          padding-top: 24px;
          border-top: 1px solid rgba(255, 255, 255, 0.05);
          font-size: 0.8rem;
          color: #475569;
          text-align: center;
        }
      </style>
    </head>
    <body>
      <main class="card">
        <div class="pill"><span></span> LIVE ENVIRONMENT</div>
        <h1>CSVAnalystEnv</h1>
        <p class="tagline">
          An OpenEnv-compatible benchmark for data-analysis agents. Test your models on tabular reasoning, filtering, and aggregation tasks.
        </p>

        <div class="section">
          <p class="section-title">Available Endpoints</p>
          <ul>
            <li><code>GET /tasks</code> – View the 13 evaluation tasks</li>
            <li><code>POST /reset</code> – Initialize a new episode</li>
            <li><code>POST /step</code> – Perform analytical actions</li>
            <li><code>GET /state</code> – Inspect environment state</li>
          </ul>
        </div>

        <div class="links">
          <a class="btn btn-primary" href="/docs">Interactive API Docs</a>
          <a class="btn btn-secondary" href="https://github.com/CMKarth1kRaj/Meta-Hackathon" target="_blank">View on GitHub</a>
          <a class="btn btn-secondary" href="/tasks">View Tasks</a>
        </div>

        <div class="meta">
          Built for Meta PyTorch OpenEnv Hackathon 2024
        </div>
      </main>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    """Return the full task bank (IDs and questions only, no answers)."""
    return [
        {"id": t["id"], "question": t["question"], "task_type": t.get("task_type")}
        for t in TASKS
    ]


@app.post("/reset", response_model=CSVObservation)
def reset(req: ResetRequest = ResetRequest()):
    """Start a new episode."""
    try:
        obs = env.reset(task_id=req.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=CSVObservation)
def step(action: CSVAction):
    """Execute an action and return the observation."""
    try:
        obs = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.get("/state", response_model=CSVState)
def state():
    """Return current episode metadata."""
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
