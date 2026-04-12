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

SYSTEM_PROMPT = """You are an expert data analyst agent operating in a CSV analysis environment.
Your goal is to answer the user's question accurately within 8 steps.

AVAILABLE ACTIONS:
- {"action_type": "list_columns"}: Returns all column names.
- {"action_type": "preview_rows", "n": 5}: Returns the first n rows.
- {"action_type": "get_unique_values", "column": "col_name"}: Returns unique values in a column.
- {"action_type": "filter_rows", "column": "col_name", "operator": "==", "value": "val"}: Filter rows.
- {"action_type": "aggregate", "column": "col_name", "agg": "sum"}: Perform aggregation.
- {"action_type": "groupby_aggregate", "group_column": "col_A", "value_column": "col_B", "agg": "sum"}
- {"action_type": "submit_answer", "answer": "your_answer_as_string"}

CRITICAL RULES:
1. Respond ONLY with a valid JSON action object.
2. You have a maximum of 8 steps. You MUST call "submit_answer" before or at Step 8.
3. NEVER repeat an exploratory action (like list_columns or preview_rows) more than once.
4. If you have already filtered by a column, do not filter by it again unless you are narrowing down.
5. Once you have seen the data you need, call "submit_answer" IMMEDIATELY.

STRATEGY:
- Steps 1-2: Inspect schema and sample data.
- Steps 3-6: Perform specific analysis or filtering.
- Steps 7-8: You MUST call "submit_answer". If this is Step 8, this is your LAST chance to submit."""

def validate_and_coerce_action(action: dict, step: int) -> dict:
    """Fixes common LLM formatting errors and enforces submission on the final step."""
    if not isinstance(action, dict):
        return {"action_type": "list_columns"}
    
    action_type = action.get("action_type")
    
    # If Step 8 and not submitting, try to force it or at least warn
    if step == MAX_STEPS and action_type != "submit_answer":
        # Force submit_answer if we can find an answer or just 'unknown'
        ans = action.get("answer") or action.get("result") or "insufficient_limit"
        return {"action_type": "submit_answer", "answer": str(ans)}

    if action_type not in [
        "list_columns", "preview_rows", "get_unique_values", 
        "filter_rows", "aggregate", "groupby_aggregate", "submit_answer"
    ]:
        action["action_type"] = "list_columns"

    if action_type == "submit_answer":
        if "answer" in action:
            action["answer"] = str(action["answer"])
        else:
            action["answer"] = "unknown"
            
    return action

def get_llm_action(client: OpenAI, question: str, obs: dict, step: int, history: List[str]) -> dict:
    """Calls the LLM with action history and explicit step context."""
    history_block = "\n".join(history[-5:]) if history else "No previous actions."
    message_info = obs.get("message", "No message.")
    visible_data = json.dumps(obs.get("visible_data", {}), indent=2)[:2000] # Cap size
    
    # Forceful warning for the final step
    final_warning = ""
    if step == MAX_STEPS:
        final_warning = "\n⚠️ WARNING: THIS IS YOUR LAST STEP. YOU MUST USE 'submit_answer' NOW OR YOU WILL FAIL."
    elif step == MAX_STEPS - 1:
        final_warning = "\n⚠️ WARNING: NEXT STEP IS THE FINAL STEP. START CONCLUDING YOUR ANALYSIS."

    user_prompt = (
        f"QUESTION: {question}\n"
        f"CURRENT STEP: {step} of {MAX_STEPS}{final_warning}\n"
        f"ENV STATUS: {message_info}\n"
        f"ACTION HISTORY:\n{history_block}\n"
        f"LATEST OBSERVATION:\n{visible_data}\n\n"
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
            max_tokens=200,
        )
        text = completion.choices[0].message.content or ""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            action_obj = json.loads(text[start:end])
            return validate_and_coerce_action(action_obj, step)
            
    except Exception as exc:
        print(f"[DEBUG] Model failed at Step {step}: {exc}", flush=True, file=sys.stderr)
    
    return {"action_type": "submit_answer", "answer": "error_fallback"} if step == MAX_STEPS else {"action_type": "list_columns"}

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
        action_history: List[str] = [] # Track history for the prompt
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
                
                # Logic to pick action (passing history)
                action_obj = get_llm_action(client, question, obs, step, action_history)
                action_str = json.dumps(action_obj)
                
                # Execute step
                try:
                    obs = env_step(action_obj)
                    reward = float(obs.get("reward", 0.0))
                    done = bool(obs.get("done", False))
                    error = None
                    
                    # Store in history for the next step's prompt
                    action_history.append(f"Step {step}: {action_str} -> {obs.get('message', 'ok')}")
                except Exception as e:
                    reward = -0.20 # penalty
                    done = True
                    error = str(e)
                
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action=action_str, reward=reward, done=done, error=error)
                
                if done:
                    break
            
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
