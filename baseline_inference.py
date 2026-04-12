"""
baseline_inference.py — Baseline evaluation of CSVAnalystEnv using an LLM.

This script:
  1. Loads all tasks from tasks.json.
  2. For each task, calls the hosted environment via HTTP (reset → step loop).
  3. Uses an LLM (via Hugging Face Inference API) to decide which actions to take.
  4. Grabs the final score from the grader.
  5. Prints an aggregate performance report.

Environment variables required:
  HF_TOKEN          — Your Hugging Face API token.
  ENV_BASE_URL      — Base URL of the running environment (default: http://localhost:8000)
  HF_MODEL_ID       — Model to use (default: mistralai/Mistral-7B-Instruct-v0.2)
"""

from __future__ import annotations

import json
import os
import sys
import requests
from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# Config from env vars
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")

if not HF_TOKEN:
    print("⚠️  HF_TOKEN not set. Set it with: export HF_TOKEN=hf_...")
    print("    Running in HEURISTIC mode (no LLM calls).")

client = InferenceClient(model=HF_MODEL_ID, token=HF_TOKEN) if HF_TOKEN else None

# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action)
    r.raise_for_status()
    return r.json()


def env_state() -> dict:
    r = requests.get(f"{ENV_BASE_URL}/state")
    r.raise_for_status()
    return r.json()


def env_tasks() -> list:
    r = requests.get(f"{ENV_BASE_URL}/tasks")
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

ACTION_TYPES = [
    "list_columns", "preview_rows", "get_unique_values",
    "filter_rows", "aggregate", "groupby_aggregate", "submit_answer"
]

SYSTEM_PROMPT = """You are a data analyst agent operating in a CSV analysis environment.
You have access to these actions: list_columns, preview_rows, get_unique_values,
filter_rows, aggregate, groupby_aggregate, submit_answer.

Given the question and the current observation, respond ONLY with a valid JSON action object.

Examples:
  {"action_type": "list_columns"}
  {"action_type": "aggregate", "column": "quantity", "agg": "sum"}
  {"action_type": "filter_rows", "column": "status", "operator": "==", "value": "Delivered"}
  {"action_type": "groupby_aggregate", "group_column": "category", "value_column": "quantity", "agg": "sum"}
  {"action_type": "submit_answer", "answer": "Electronics"}

Reply with JSON only, no explanation."""


def llm_pick_action(question: str, obs: dict, step: int) -> dict:
    """Call the HF Inference API to pick the next action using modern InferenceClient."""
    if not client:
        return _heuristic_action(step)

    history_str = json.dumps(obs.get("visible_data", {}), indent=2)
    prompt = f"Question: {question}\nStep: {step}\nLast visible data:\n{history_str}"

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        resp = client.chat_completion(
            messages=messages,
            max_tokens=200,
            temperature=0.1,
        )
        
        text = resp.choices[0].message.content
        
        # Extract JSON from the generated text
        start = text.rfind("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        print(f"    LLM error: {e} — falling back to heuristic")

    return _heuristic_action(step)


def _heuristic_action(step: int) -> dict:
    """Simple fallback heuristic (for when no LLM token is available)."""
    if step == 1:
        return {"action_type": "list_columns"}
    elif step == 2:
        return {"action_type": "preview_rows", "n": 3}
    else:
        # Force submit with empty answer to close the episode
        return {"action_type": "submit_answer", "answer": "unknown"}


# ---------------------------------------------------------------------------
# Normalised score from state
# ---------------------------------------------------------------------------

def compute_score(state: dict) -> dict:
    """Return a normalized score dict (0.01-0.99) for this episode."""
    submitted = state.get("submitted_answer")
    truth = state.get("correct_answer", "")
    steps = state.get("step_count", 0)
    invalid = state.get("invalid_actions", 0)

    def _norm(a: str) -> str:
        s = str(a).strip().lower()
        try:
            f = float(s)
            if f == int(f):
                s = str(int(f))
        except (ValueError, TypeError):
            pass
        return s

    correct = _norm(submitted) == _norm(truth) if submitted else False

    # Numeric tolerance fallback
    if not correct and submitted:
        try:
            correct = abs(float(submitted) - float(truth)) < 1e-2
        except (ValueError, TypeError):
            pass

    return {
        "correct": correct,
        "score": 0.99 if correct else 0.01,
        "steps_used": steps,
        "invalid_actions": invalid,
        "submitted": submitted,
        "expected": truth,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_baseline():
    print(f"\n{'='*60}")
    print(f"  CSVAnalystEnv — Baseline Inference")
    print(f"  Model : {HF_MODEL_ID if HF_TOKEN else 'HEURISTIC (no HF_TOKEN)'}")
    print(f"  Server: {ENV_BASE_URL}")
    print(f"{'='*60}\n")

    # Fetch task list from environment (answers are hidden)
    try:
        task_stubs = env_tasks()
    except Exception as e:
        print(f"❌ Cannot reach environment at {ENV_BASE_URL}: {e}")
        print("   Make sure the server is running: uvicorn server.app:app --reload")
        sys.exit(1)

    results = []

    print(f"{'ID':<6} {'Diff':<8} {'Score':<7} {'Steps':<7} {'Status'}")
    print("-" * 55)

    # Load full tasks locally for difficulty display
    with open("tasks/tasks.json") as f:
        full_tasks = {t["id"]: t for t in json.load(f)}

    for stub in task_stubs:
        task_id = stub["id"]
        question = stub["question"]
        difficulty = full_tasks.get(task_id, {}).get("difficulty", "?")

        # Reset episode
        obs = env_reset(task_id)

        # Agent loop
        step = 0
        while not obs.get("done", False):
            step += 1
            action = llm_pick_action(question, obs, step)
            try:
                obs = env_step(action)
            except Exception as e:
                print(f"    Step error: {e}")
                break

        # Grade
        state = env_state()
        score_info = compute_score(state)
        results.append({"task_id": task_id, "difficulty": difficulty, **score_info})

        status = "✅" if score_info["correct"] else "❌"
        print(f"{task_id:<6} {difficulty:<8} {score_info['score']:<7.1f} {score_info['steps_used']:<7} {status}  {question[:35]}...")

    # Aggregate report
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    avg_score = sum(r["score"] for r in results) / n
    avg_steps = sum(r["steps_used"] for r in results) / n
    total_invalid = sum(r["invalid_actions"] for r in results)

    diff_breakdown = {}
    for r in results:
        d = r["difficulty"]
        if d not in diff_breakdown:
            diff_breakdown[d] = {"correct": 0, "total": 0}
        diff_breakdown[d]["total"] += 1
        if r["correct"]:
            diff_breakdown[d]["correct"] += 1

    print(f"\n{'='*55}")
    print(f"  AGGREGATE RESULTS ({HF_MODEL_ID if HF_TOKEN else 'heuristic'})")
    print(f"{'='*55}")
    print(f"  Tasks evaluated : {n}")
    print(f"  Correct         : {correct}/{n}")
    print(f"  Avg Score       : {avg_score:.3f}")
    print(f"  Accuracy        : {correct/n:.1%}")
    print(f"  Avg Steps       : {avg_steps:.1f}")
    print(f"  Invalid Actions : {total_invalid}")
    print()
    print("  By Difficulty:")
    for diff, stats in sorted(diff_breakdown.items()):
        acc = stats["correct"] / stats["total"]
        print(f"    {diff:<8}: {stats['correct']}/{stats['total']}  ({acc:.0%})")
    print(f"{'='*55}\n")

    # Save results to JSON
    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": HF_MODEL_ID if HF_TOKEN else "heuristic",
            "environment": ENV_BASE_URL,
            "summary": {
                "total": n,
                "correct": correct,
                "accuracy": correct / n,
                "avg_score": avg_score,
                "avg_steps": avg_steps,
                "total_invalid_actions": total_invalid,
            },
            "by_difficulty": diff_breakdown,
            "results": results,
        }, f, indent=2)

    print("📄 Full results saved to baseline_results.json")


if __name__ == "__main__":
    run_baseline()
