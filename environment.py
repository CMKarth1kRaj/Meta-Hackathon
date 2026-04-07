"""
CSVAnalystEnv — an OpenEnv-compatible reinforcement-learning environment
for evaluating data-analysis agents on structured CSV reasoning tasks.

Core loop:  reset() → step(action) → step(action) → … → done
"""

from __future__ import annotations

import random
import uuid
from typing import Any, List

import numpy as np
import pandas as pd

from models import CSVAction, CSVObservation, CSVState


def _jsonable(obj: Any) -> Any:
    """Convert numpy/pandas types to JSON-serialisable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


class CSVAnalystEnv:
    """OpenEnv-style environment for CSV data analysis tasks.

    Methods
    -------
    reset()   — Start a new episode with a random task.
    step(a)   — Execute one action and return an observation.
    state()   — Return the current episode metadata.
    """

    # ------------------------------------------------------------------
    # Reward constants
    # ------------------------------------------------------------------
    STEP_COST: float = -0.05
    INVALID_PENALTY: float = -0.20
    CORRECT_REWARD: float = 1.0
    INCORRECT_PENALTY: float = -1.0
    EFFICIENCY_BONUS: float = 0.20      # awarded when solved in ≤ 4 steps
    TIMEOUT_PENALTY: float = -0.50      # forced termination at max steps
    GROUPBY_STEP_COST: float = -0.08    # slightly higher for groupby

    def __init__(
        self,
        csv_path: str,
        tasks: List[dict],
        max_steps: int = 8,
    ) -> None:
        self.df: pd.DataFrame = pd.read_csv(csv_path)
        self.tasks = tasks
        self.max_steps = max_steps
        self.current_task: dict | None = None
        self.state_obj: CSVState | None = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> CSVObservation:
        """Start a new episode.

        Parameters
        ----------
        task_id : str, optional
            If given, use that specific task instead of a random one.
        """
        if task_id is not None:
            matches = [t for t in self.tasks if t["id"] == task_id]
            if not matches:
                raise ValueError(f"Unknown task_id: {task_id}")
            self.current_task = matches[0]
        else:
            self.current_task = random.choice(self.tasks)

        self.state_obj = CSVState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            max_steps=self.max_steps,
            question_id=self.current_task["id"],
            correct_answer=str(self.current_task["answer"]),
            history=[],
        )

        return CSVObservation(
            question=self.current_task["question"],
            visible_data={"columns": self.df.columns.tolist(), "row_count": len(self.df)},
            message="Episode started. Use actions to inspect the CSV and submit an answer.",
            reward=0.0,
            done=False,
        )

    def state(self) -> CSVState:
        """Return current episode metadata."""
        if self.state_obj is None:
            raise RuntimeError("Call reset() before accessing state().")
        return self.state_obj

    def step(self, action: CSVAction) -> CSVObservation:
        """Execute *action* and return the resulting observation."""
        if self.state_obj is None:
            raise RuntimeError("Call reset() before step().")
        if self.state_obj.step_count >= self.max_steps:
            raise RuntimeError("Episode already terminated (max steps reached).")

        self.state_obj.step_count += 1
        reward: float = self.STEP_COST
        done: bool = False
        visible_data: dict = {}
        message: str = "Action executed."

        try:
            if action.action_type == "list_columns":
                visible_data = {"columns": self.df.columns.tolist()}

            elif action.action_type == "preview_rows":
                n = action.n or 5
                visible_data = {"rows": self.df.head(n).to_dict(orient="records")}

            elif action.action_type == "get_unique_values":
                self._require(action.column, "column")
                self._assert_column(action.column)
                vals = self.df[action.column].dropna().unique().tolist()
                visible_data = {"column": action.column, "unique_values": _jsonable(vals)}

            elif action.action_type == "filter_rows":
                self._require(action.column, "column")
                self._require(action.operator, "operator")
                self._require(action.value is not None, "value")
                self._assert_column(action.column)
                out = self._apply_filter(action.column, action.operator, action.value)
                visible_data = {
                    "match_count": len(out),
                    "sample_rows": _jsonable(out.head(5).to_dict(orient="records")),
                }

            elif action.action_type == "aggregate":
                self._require(action.column, "column")
                self._require(action.agg, "agg")
                self._assert_column(action.column)
                result = self._aggregate(self.df[action.column], action.agg)
                visible_data = {"column": action.column, "agg": action.agg, "result": _jsonable(result)}

            elif action.action_type == "groupby_aggregate":
                reward = self.GROUPBY_STEP_COST  # override base step cost
                self._require(action.group_column, "group_column")
                self._require(action.value_column, "value_column")
                self._require(action.agg, "agg")
                self._assert_column(action.group_column)
                self._assert_column(action.value_column)
                grouped = (
                    self.df.groupby(action.group_column)[action.value_column]
                    .agg(action.agg)
                    .reset_index()
                )
                visible_data = {
                    "group_column": action.group_column,
                    "value_column": action.value_column,
                    "agg": action.agg,
                    "rows": _jsonable(grouped.to_dict(orient="records")),
                }

            elif action.action_type == "submit_answer":
                self._require(action.answer is not None, "answer")
                pred = str(action.answer).strip().lower()
                truth = str(self.state_obj.correct_answer).strip().lower()
                # Exact match or numeric-tolerant match
                if self._answers_match(pred, truth):
                    reward = self.CORRECT_REWARD
                    if self.state_obj.step_count <= 4:
                        reward += self.EFFICIENCY_BONUS
                    message = "✅ Correct answer submitted."
                else:
                    reward = self.INCORRECT_PENALTY
                    message = "❌ Incorrect answer submitted."
                done = True
                self.state_obj.submitted_answer = str(action.answer)

            else:
                raise ValueError(f"Unknown action type: {action.action_type}")

        except Exception as exc:
            reward += self.INVALID_PENALTY
            self.state_obj.invalid_actions += 1
            message = f"⚠️ Invalid action: {exc}"

        # Forced termination when step budget is exhausted
        if self.state_obj.step_count >= self.max_steps and not done:
            done = True
            reward += self.TIMEOUT_PENALTY
            message = "⏰ Max steps reached. Episode terminated."

        # Record in history
        self.state_obj.history.append({
            "step": self.state_obj.step_count,
            "action": action.model_dump(),
            "reward": round(reward, 4),
            "done": done,
            "message": message,
        })

        return CSVObservation(
            question=self.current_task["question"],
            visible_data=_jsonable(visible_data),
            message=message,
            reward=round(reward, 4),
            done=done,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require(condition: Any, name: str) -> None:
        if not condition:
            raise ValueError(f"Missing required parameter: '{name}'")

    def _assert_column(self, col: str) -> None:
        if col not in self.df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available: {self.df.columns.tolist()}"
            )

    def _apply_filter(self, column: str, operator: str, value: Any) -> pd.DataFrame:
        series = self.df[column]
        # Attempt numeric coercion for comparison operators
        if operator in (">", "<", ">=", "<="):
            try:
                value = float(value)
                series = pd.to_numeric(series, errors="coerce")
            except (ValueError, TypeError):
                pass

        ops = {
            "==": lambda s, v: s == v,
            "!=": lambda s, v: s != v,
            ">":  lambda s, v: s > v,
            "<":  lambda s, v: s < v,
            ">=": lambda s, v: s >= v,
            "<=": lambda s, v: s <= v,
        }
        if operator not in ops:
            raise ValueError(f"Unsupported operator: {operator}")
        mask = ops[operator](series, value)
        return self.df[mask]

    @staticmethod
    def _aggregate(series: pd.Series, agg: str) -> Any:
        fn = {
            "sum":   series.sum,
            "mean":  series.mean,
            "count": series.count,
            "max":   series.max,
            "min":   series.min,
        }.get(agg)
        if fn is None:
            raise ValueError(f"Unsupported aggregation: {agg}")
        return fn()

    @staticmethod
    def _answers_match(pred: str, truth: str) -> bool:
        """Exact string match, with numeric-tolerance fallback."""
        if pred == truth:
            return True
        try:
            return abs(float(pred) - float(truth)) < 1e-2
        except (ValueError, TypeError):
            return False
