"""Typed Pydantic models for CSVAnalystEnv actions, observations, and state."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class CSVAction(BaseModel):
    """An action the agent can take in the CSV analysis environment."""

    action_type: Literal[
        "list_columns",
        "preview_rows",
        "get_unique_values",
        "filter_rows",
        "aggregate",
        "groupby_aggregate",
        "submit_answer",
    ]
    column: Optional[str] = None
    value_column: Optional[str] = None
    group_column: Optional[str] = None
    operator: Optional[Literal["==", "!=", ">", "<", ">=", "<="]] = None
    value: Optional[Any] = None
    agg: Optional[Literal["sum", "mean", "count", "max", "min"]] = None
    answer: Optional[str] = None
    n: Optional[int] = 5


class CSVObservation(BaseModel):
    """Observation returned by the environment after each step."""

    question: str
    visible_data: Dict[str, Any] = Field(default_factory=dict)
    message: str
    reward: float = 0.0
    done: bool = False


class CSVState(BaseModel):
    """Internal state of the environment for a single episode."""

    episode_id: str
    step_count: int
    max_steps: int
    question_id: str
    invalid_actions: int = 0
    submitted_answer: Optional[str] = None
    correct_answer: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
