import os
import json
from typing import List, Optional
from openai import OpenAI

# Config
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Meta-Llama-3-8B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy_key"
MAX_STEPS = 8

def get_client():
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

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
    action_type = raw.get("action_type")
    if step >= max_steps and action_type != "submit_answer":
        ans = raw.get("answer") or raw.get("result") or "unknown"
        return {"action_type": "submit_answer", "answer": str(ans)}
    if action_type not in ["list_columns", "preview_rows", "get_unique_values", "filter_rows", "aggregate", "groupby_aggregate", "submit_answer"]:
        seen_list = any(h.get("action", {}).get("action_type") == "list_columns" for h in history)
        if not seen_list: return {"action_type": "list_columns"}
        return {"action_type": "preview_rows", "n": 5}
    if action_type == "preview_rows":
        return {"action_type": "preview_rows", "n": int(raw.get("n") or 5)}
    if action_type == "submit_answer":
        ans = raw.get("answer")
        if ans is None:
            vis = obs.get("visible_data", {}) or {}
            ans = vis.get("result") or vis.get("answer") or "unknown"
        return {"action_type": "submit_answer", "answer": str(ans)}
    return raw

def get_llm_action(client: OpenAI, question: str, obs: dict, step: int, history: List[dict]) -> tuple[dict, str]:
    # Build a more descriptive history that includes short summaries of results
    hist_entries = []
    for h in history[-4:]:
        action_str = json.dumps(h["action"])
        # Include a snippet of the observation to give the LLM memory
        obs_snippet = json.dumps(h.get("observation", {}).get("visible_data", {}))[:200]
        hist_entries.append(f"Step {h['step']}: {action_str} -> Result: {obs_snippet}...")
    
    history_text = "\n".join(hist_entries) if hist_entries else "None yet."
    visible_data = json.dumps(obs.get("visible_data", {}), indent=2)[:2000] 
    
    user_prompt = (
        f"Question: {question}\n"
        f"Stage: {step} of {MAX_STEPS}\n\n"
        f"Last Observation:\n{visible_data}\n\n"
        f"Action History (recent steps):\n{history_text}\n\n"
        "Output ONLY the next JSON action:"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        text = completion.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            return coerce_action(json.loads(text[start:end]), step, MAX_STEPS, obs, history), f"LLM ({MODEL_NAME})"
    except Exception:
        pass
    return coerce_action(heuristic_action(step), step, MAX_STEPS, obs, history), "Heuristic Fallback"
