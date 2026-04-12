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
          padding: 20px;
          font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #0f172a;
          color: #e2e8f0;
          display: flex;
          min-height: 100vh;
          align-items: center;
          justify-content: center;
        }
        .card {
          max-width: 800px;
          width: 100%;
          padding: 40px;
          border-radius: 24px;
          background: linear-gradient(145deg, #1e293b, #0f172a);
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        h1 {
          margin: 0 0 10px;
          font-size: 2.5rem;
          font-weight: 800;
          background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .tagline {
          font-size: 1.1rem;
          color: #94a3b8;
          margin-bottom: 30px;
        }
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 15px;
          margin-bottom: 30px;
        }
        .stat-box {
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.05);
          border-radius: 12px;
          padding: 15px;
          text-align: center;
        }
        .stat-value {
          font-size: 1.5rem;
          font-weight: 700;
          color: #f8fafc;
          margin-bottom: 5px;
        }
        .stat-label {
          font-size: 0.8rem;
          color: #64748b;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        .grid-2 {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 30px;
          margin-bottom: 30px;
        }
        @media (max-width: 600px) {
          .grid-2 { grid-template-columns: 1fr; }
        }
        h2 {
          font-size: 1.2rem;
          color: #e2e8f0;
          margin-bottom: 15px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          padding-bottom: 10px;
        }
        p, li {
          font-size: 0.95rem;
          line-height: 1.6;
          color: #cbd5e1;
        }
        ul { padding-left: 20px; }
        li { margin-bottom: 8px; }
        code {
          background: rgba(56, 189, 248, 0.1);
          color: #38bdf8;
          padding: 3px 6px;
          border-radius: 4px;
          font-family: monospace;
          font-size: 0.9em;
        }
        .btn-group {
          display: flex;
          flex-wrap: wrap;
          gap: 10px;
          justify-content: center;
          margin-top: 40px;
        }
        .btn {
          padding: 12px 24px;
          border-radius: 99px;
          font-weight: 600;
          font-size: 0.95rem;
          text-decoration: none;
          transition: all 0.2s;
        }
        .btn-primary {
          background: #38bdf8;
          color: #0f172a;
        }
        .btn-primary:hover { background: #0ea5e9; transform: translateY(-2px); }
        .btn-secondary {
          background: rgba(255, 255, 255, 0.1);
          color: #f8fafc;
        }
        .btn-secondary:hover { background: rgba(255, 255, 255, 0.15); transform: translateY(-2px); }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>CSVAnalystEnv</h1>
        <div class="tagline">An OpenEnv-compatible benchmark for tabular reasoning agents.</div>

        <div class="stats-grid">
          <div class="stat-box">
            <div class="stat-value">13</div>
            <div class="stat-label">Evaluation Tasks</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">3</div>
            <div class="stat-label">Difficulty Levels</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">100%</div>
            <div class="stat-label">OpenEnv Compliant</div>
          </div>
          <div class="stat-box">
            <div class="stat-value">Live</div>
            <div class="stat-label">FastAPI HTTP</div>
          </div>
        </div>

        <div class="grid-2">
          <div>
            <h2>How it works</h2>
            <p>Agents interact with a fixed CSV dataset representing e-commerce orders. Instead of writing raw code, agents must use a constrained action space (like <code>filter_rows</code> or <code>groupby_aggregate</code>) to explore the data and find the answer.</p>
            <p>The environment enforces strict programmatic grading, limits episode length, and shapes behavior via normalized rewards (+1 for success, penalties for invalid tool use).</p>
          </div>
          <div>
            <h2>Core Endpoints</h2>
            <ul>
               <li><code>GET /tasks</code> lists the question bank.</li>
               <li><code>POST /reset</code> begins an episode.</li>
               <li><code>POST /step</code> submits an action and returns the next observation.</li>
               <li><code>GET /state</code> returns the full episode transcript.</li>
            </ul>
          </div>
        </div>

        <div class="btn-group">
          <a href="/docs" class="btn btn-primary">Open API Docs</a>
          <a href="/ui" class="btn btn-secondary">Human Interface</a>
          <a href="/health" class="btn btn-secondary">Health Check</a>
          <a href="https://github.com/CMKarth1kRaj/Meta-Hackathon" target="_blank" class="btn btn-secondary">GitHub Repo</a>
        </div>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ui", include_in_schema=False)
def get_ui():
    """Serve the HumanAgent interface."""
    import os
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


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


@app.post("/agent_action")
def agent_action():
    """Ask the LLM to recommend the next action based on current state."""
    from agent_utils import get_llm_action, get_client
    try:
        current_state = env.state()
        # We need the last observation to get the question text
        # Since env doesn't store the full last obs in a simple way, 
        # we reconstruct a minimal one for the prompt.
        obs_dict = {
            "question": next((t["question"] for t in TASKS if t["id"] == current_state.question_id), "N/A"),
            "visible_data": current_state.history[-1]["observation"]["visible_data"] if current_state.history else {},
            "message": "Providing recommendation..."
        }
        
        client = get_client()
        action = get_llm_action(
            client=client,
            question=obs_dict["question"],
            obs=obs_dict,
            step=current_state.step_count + 1,
            history=current_state.history
        )
        return action
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")



@app.get("/state", response_model=CSVState)
def state():
    """Return current episode metadata."""
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


def main():
    import uvicorn
    # Use the port defined in models or common for this project (7860)
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
