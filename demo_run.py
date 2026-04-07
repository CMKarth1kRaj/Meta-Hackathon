#!/usr/bin/env python3
"""
demo_run.py — Quick demonstration of the CSVAnalystEnv loop.

Runs a single hard-coded episode to prove that reset → step → … → done
works end-to-end, then prints a grading report.
"""

import json

from environment import CSVAnalystEnv
from grader import grade_episode
from models import CSVAction


def separator(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main() -> None:
    # ── Load tasks ──────────────────────────────────────────────
    with open("tasks/tasks.json") as f:
        tasks = json.load(f)

    env = CSVAnalystEnv("data/orders.csv", tasks)

    # ── Episode: answer q1 (Delivered orders in South = 8) ─────
    separator("RESET — picking task q1")
    obs = env.reset(task_id="q1")
    print(f"Question : {obs.question}")
    print(f"Message  : {obs.message}")
    print(f"Reward   : {obs.reward}")
    print(f"Columns  : {obs.visible_data.get('columns')}")

    # Step 1 — list columns
    separator("STEP 1 — list_columns")
    obs = env.step(CSVAction(action_type="list_columns"))
    print(f"Columns  : {obs.visible_data}")
    print(f"Reward   : {obs.reward}")

    # Step 2 — preview first 3 rows
    separator("STEP 2 — preview_rows(n=3)")
    obs = env.step(CSVAction(action_type="preview_rows", n=3))
    for row in obs.visible_data.get("rows", []):
        print(f"  {row}")
    print(f"Reward   : {obs.reward}")

    # Step 3 — filter delivered + South
    separator("STEP 3 — filter_rows(status == Delivered)")
    obs = env.step(CSVAction(
        action_type="filter_rows",
        column="status",
        operator="==",
        value="Delivered",
    ))
    print(f"Matches  : {obs.visible_data.get('match_count')}")
    print(f"Reward   : {obs.reward}")

    # Step 4 — now filter region == South on full data and combine mentally
    separator("STEP 4 — filter_rows(region == South)")
    obs = env.step(CSVAction(
        action_type="filter_rows",
        column="region",
        operator="==",
        value="South",
    ))
    print(f"Matches  : {obs.visible_data.get('match_count')}")
    print(f"Reward   : {obs.reward}")

    # Step 5 — submit answer
    separator("STEP 5 — submit_answer('8')")
    obs = env.step(CSVAction(action_type="submit_answer", answer="8"))
    print(f"Message  : {obs.message}")
    print(f"Reward   : {obs.reward}")
    print(f"Done     : {obs.done}")

    # ── Grading report ─────────────────────────────────────────
    separator("GRADING REPORT")
    state = env.state().model_dump()
    report = grade_episode(state, obs.question)
    print(report.summary)
    print(f"\nFull report:")
    print(f"  Episode   : {report.episode_id}")
    print(f"  Question  : {report.question}")
    print(f"  Expected  : {report.correct_answer}")
    print(f"  Submitted : {report.submitted_answer}")
    print(f"  Correct   : {report.is_correct}")
    print(f"  Steps     : {report.steps_used}/{report.max_steps}")
    print(f"  Invalid   : {report.invalid_actions}")
    print(f"  Reward    : {report.cumulative_reward}")
    print()


if __name__ == "__main__":
    main()
