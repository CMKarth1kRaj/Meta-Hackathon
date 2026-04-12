import os
import json
import sys
import requests
from typing import List, Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from env vars (Hackathon Mandatory)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Meta-Llama-3-8B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy_key"

# Environment config
ENV_BASE_URL = os.getenv("SPACE_URL") or os.getenv("ENV_BASE_URL") or "http://localhost:7860"
ENV_BASE_URL = ENV_BASE_URL.rstrip("/")

BENCHMARK = "csv-analyst-env"
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# Structured Logging Helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Emits the [START] log block to stdout."""
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emits the [STEP] log block to stdout."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emits the [END] log block to stdout."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# LLM Logic
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful, step-by-step data analyst operating in a CSV analysis environment.

You can ONLY act by returning a JSON object with these fields:
- "action_type": one of ["list_columns", "preview_rows", "get_unique_values", "filter_rows", "aggregate", "groupby_aggregate", "submit_answer"].

Optional fields depending on the action:
- preview_rows: { "action_type": "preview_rows", "n": <int> }
- get_unique_values: { "action_type": "get_unique_values", "column": "<col>" }
- filter_rows: {
    "action_type": "filter_rows",
    "column": "<col>",
    "operator": "== or != or > or < or >= or <=",
    "value": "<value>"
  }
- aggregate: {
    "action_type": "aggregate",
    "column": "<numeric col>",
    "agg": "sum or mean or count or max or min"
  }
- groupby_aggregate: {
    "action_type": "groupby_aggregate",
    "group_column": "<col>",
    "value_column": "<numeric col>",
    "agg": "sum or mean or count or max or min"
  }
- submit_answer: { "action_type": "submit_answer", "answer": "<final answer>" }

Rules:
1. Use at most one action per step.
2. Avoid repeating the same action_type with the same arguments.
3. First inspect the data (list_columns, preview_rows, get_unique_values, filter_rows), then compute (aggregate or groupby_aggregate), then finish with submit_answer.
4. By the last step you MUST call submit_answer with your best guess.

Reply with JSON ONLY, no explanation or extra text.
"""

def heuristic_action(step: int) -> dict:
    if step == 1:
        return {"action_type": "list_columns"}
    elif step == 2:
        return {"action_type": "preview_rows", "n": 5}
    else:
        return {"action_type": "submit_answer", "answer": "unknown"}

def coerce_action(raw: dict, step: int, max_steps: int, obs: dict, history: List[dict]) -> dict:
    """Make the action safe & useful without always reverting to list_columns."""
    action_type = raw.get("action_type")

    # Force submit on very last step
    if step >= max_steps and action_type != "submit_answer":
        ans = raw.get("answer") or raw.get("result") or "unknown"
        return {"action_type": "submit_answer", "answer": str(ans)}

    # Unknown or missing type -> fall back but try not to repeat
    if action_type not in [
        "list_columns", "preview_rows", "get_unique_values", 
        "filter_rows", "aggregate", "groupby_aggregate", "submit_answer"
    ]:
        seen_list = any(h.get("action", {}).get("action_type") == "list_columns" for h in history)
        if not seen_list:
            return {"action_type": "list_columns"}
        return {"action_type": "preview_rows", "n": 5}

    if action_type == "preview_rows":
        n = raw.get("n") or 5
        return {"action_type": "preview_rows", "n": int(n)}

    if action_type == "submit_answer":
        ans = raw.get("answer")
        if ans is None:
            vis = obs.get("visible_data", {}) or {}
            ans = vis.get("result") or vis.get("answer") or "unknown"
        return {"action_type": "submit_answer", "answer": str(ans)}

    return raw

def get_llm_action(client: OpenAI, question: str, obs: dict, step: int, history: List[dict]) -> dict:
    """Calls the LLM with action history and explicit step context."""
    history_json = json.dumps([h["action"] for h in history[-5:]], indent=2) if history else "[]"
    message_info = obs.get("message", "No message.")
    visible_data = json.dumps(obs.get("visible_data", {}), indent=2)[:2000] 
    
    user_prompt = (
        f"Question: {question}\n"
        f"Stage: {step} of {MAX_STEPS}\n\n"
        f"Last Observation:\n{visible_data}\n\n"
        f"Action History (recent):\n{history_json}\n"
        "Output ONLY the next JSON action:"
    )
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        text = completion.choices[0].message.content or ""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            raw_action = json.loads(text[start:end])
            return coerce_action(raw_action, step, MAX_STEPS, obs, history)
            
    except Exception as exc:
        print(f"[DEBUG] Model failed at Step {step}: {exc}", flush=True, file=sys.stderr)
    
    return coerce_action(heuristic_action(step), step, MAX_STEPS, obs, history)

# ---------------------------------------------------------------------------
# Environment Interaction
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()

def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action)
    r.raise_for_status()
    return r.json()

def env_tasks() -> list:
    r = requests.get(f"{ENV_BASE_URL}/tasks")
    r.raise_for_status()
    return r.json()

# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    try:
        task_stubs = env_tasks()
    except Exception as e:
        print(f"[DEBUG] Cannot reach environment at {ENV_BASE_URL}: {e}", flush=True, file=sys.stderr)
        sys.exit(1)
    
    # If a specific task is requested via ENV, only run that one
    target_task = os.getenv("TASK_ID")
    if target_task:
        task_stubs = [t for t in task_stubs if t["id"] == target_task]

    for stub in task_stubs:
        task_id = stub["id"]
        question = stub["question"]
        
        rewards: List[float] = []
        history: List[dict] = [] 
        steps_taken = 0
        score = 0.01
        success = False
        
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        
        try:
            obs = env_reset(task_id)
            done = False
            
            for step in range(1, MAX_STEPS + 1):
                if done:
                    break
                
                # Logic to pick action (passing structured history)
                action_obj = get_llm_action(client, question, obs, step, history)
                action_str = json.dumps(action_obj)
                
                # Execute step
                try:
                    obs = env_step(action_obj)
                    reward = float(obs.get("reward", 0.0))
                    done = bool(obs.get("done", False))
                    error = None
                    
                    # Store in history for the next step's prompt
                    history.append({
                        "step": step,
                        "action": action_obj,
                        "reward": reward,
                        "message": obs.get("message", "ok")
                    })
                except Exception as e:
                    reward = -0.20 # penalty
                    done = True
                    error = str(e)
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                
                if done:
                    break
            
            # Safety: if we hit max_steps without done, force one last check for submitted_answer in state
            if steps_taken >= MAX_STEPS and not done:
                pass # The environment will handle termination at max steps automatically on next call or in state fetch
            
            # Final scoring (normalized)
            try:
                state_resp = requests.get(f"{ENV_BASE_URL}/state").json()
                # Check if the submitted answer matches the correct one
                # state_resp contains 'submitted_answer' and 'correct_answer'
                sub = str(state_resp.get("submitted_answer", "")).strip().lower()
                truth = str(state_resp.get("correct_answer", "")).strip().lower()
                
                # Check for numeric match too
                def is_match(a, b):
                    if not a or not b: return False
                    if a == b: return True
                    try:
                        return abs(float(a) - float(b)) < 1e-2
                    except:
                        return False

                score = 0.99 if is_match(sub, truth) else 0.01
            except Exception as e:
                # Fallback: if we got a big reward in any step, it's likely correct
                score = 0.99 if any(r > 0.5 for r in rewards) else 0.01
            
            success = score >= SUCCESS_SCORE_THRESHOLD
            
        except Exception as e:
            print(f"[DEBUG] Episode failed: {e}", flush=True, file=sys.stderr)
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    main()
