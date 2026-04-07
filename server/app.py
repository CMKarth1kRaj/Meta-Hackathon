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

@app.get("/", include_in_schema=False)
def root():
    """Welcome message and redirect to docs."""
    return {
        "message": "Welcome to CSVAnalystEnv API",
        "documentation": "/docs",
        "endpoints": {
            "health": "/health",
            "tasks": "/tasks",
            "reset": "/reset",
            "step": "/step",
            "state": "/state"
        }
    }


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
