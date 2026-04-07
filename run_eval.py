#!/usr/bin/env python3
"""
run_eval.py — Benchmarking script for CSVAnalystEnv.

This script runs a basic "heuristic agent" (or just iterates through all tasks)
to demonstrate how the environment is used for evaluation and to produce
aggregate performance metrics.
"""

import json
from environment import CSVAnalystEnv
from grader import grade_episode, grade_batch
from models import CSVAction

def main():
    # 1. Load tasks
    with open("tasks/tasks.json") as f:
        tasks = json.load(f)

    env = CSVAnalystEnv("data/orders.csv", tasks)
    reports = []

    print(f"🚀 Starting benchmark over {len(tasks)} tasks...")
    print(f"{'ID':<5} | {'Status':<10} | {'Steps':<6} | {'Reward':<8} | {'Question'}")
    print("-" * 80)

    for task in tasks:
        # Reset to specific task
        obs = env.reset(task_id=task["id"])
        
        # --- Simple Heuristic Agent Loop ---
        # 1. List columns (Step 1)
        obs = env.step(CSVAction(action_type="list_columns"))
        
        # 2. Preview rows (Step 2)
        obs = env.step(CSVAction(action_type="preview_rows", n=2))
        
        # 3. Submit the correct answer (Simulating a perfect agent)
        # Note: In a real eval, your LLM/Agent would determine this value.
        obs = env.step(CSVAction(action_type="submit_answer", answer=task["answer"]))
        
        # --- Grade the episode ---
        state = env.state().model_dump()
        report = grade_episode(state, task["question"])
        reports.append(report)
        
        status = "✅ PASS" if report.is_correct else "❌ FAIL"
        print(f"{task['id']:<5} | {status:<10} | {report.steps_used:<6} | {report.cumulative_reward:<8.2f} | {task['question'][:40]}...")

    # --- Aggregate Results ---
    batch_report = grade_batch(reports)
    
    print("\n" + "="*40)
    print("       FINAL BENCHMARK REPORT")
    print("="*40)
    print(f"Total Tasks      : {batch_report.total_episodes}")
    print(f"Success Seed     : {batch_report.correct}")
    print(f"Accuracy         : {batch_report.accuracy:.1%}")
    print(f"Avg Steps        : {batch_report.avg_steps:.2f}")
    print(f"Avg Reward       : {batch_report.avg_reward:.2f}")
    print(f"Invalid Actions  : {batch_report.total_invalid_actions}")
    print("="*40)

if __name__ == "__main__":
    main()
