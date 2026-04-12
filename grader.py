"""
Programmatic grader for CSVAnalystEnv episodes.

Evaluates agent performance based on correctness, efficiency,
and action validity — producing a structured report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

def normalize_answer(raw: str) -> str:
    """Lowercase, strip whitespace, remove trailing '.0' for whole numbers."""
    s = str(raw).strip().lower()
    # "7.0" → "7"  (common with pandas numeric output)
    try:
        f = float(s)
        if f == int(f):
            s = str(int(f))
    except (ValueError, TypeError):
        pass
    return s


def is_correct(pred: str, truth: str) -> bool:
    """Check if predicted answer matches ground truth.

    Supports exact string match and numeric tolerance (±0.01).
    """
    p, t = normalize_answer(pred), normalize_answer(truth)
    if p == t:
        return True
    try:
        return abs(float(p) - float(t)) < 1e-2
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Episode report
# ---------------------------------------------------------------------------

@dataclass
class EpisodeReport:
    """Structured grading output for one episode."""

    episode_id: str
    question_id: str
    question: str
    correct_answer: str
    submitted_answer: str | None
    is_correct: bool
    steps_used: int
    max_steps: int
    invalid_actions: int
    cumulative_reward: float
    efficiency_bonus: bool
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Human-readable one-line summary."""
        status = "✅ Correct" if self.is_correct else "❌ Incorrect"
        inv = f", {self.invalid_actions} invalid" if self.invalid_actions else ""
        bonus = " +efficiency bonus" if self.efficiency_bonus else ""
        return (
            f"{status} in {self.steps_used}/{self.max_steps} steps"
            f"{inv} | reward={self.cumulative_reward:.2f}{bonus}"
        )


def grade_episode(state_dict: dict, question: str) -> EpisodeReport:
    """Grade a completed episode from its state dictionary.

    Parameters
    ----------
    state_dict : dict
        The output of ``env.state().model_dump()``.
    question : str
        The question text for this episode.
    """
    submitted = state_dict.get("submitted_answer")
    truth = state_dict.get("correct_answer", "")
    correct = is_correct(submitted, truth) if submitted is not None else False

    history = state_dict.get("history", [])
    cumulative_reward = sum(h.get("reward", 0.0) for h in history)
    steps = state_dict.get("step_count", 0)

    return EpisodeReport(
        episode_id=state_dict.get("episode_id", ""),
        question_id=state_dict.get("question_id", ""),
        question=question,
        correct_answer=truth,
        submitted_answer=submitted,
        is_correct=correct,
        steps_used=steps,
        max_steps=state_dict.get("max_steps", 0),
        invalid_actions=state_dict.get("invalid_actions", 0),
        cumulative_reward=round(cumulative_reward, 4),
        efficiency_bonus=correct and steps <= 4,
        history=history,
    )


# ---------------------------------------------------------------------------
# Normalised score (0.01 – 0.99) — required by OpenEnv task spec
# ---------------------------------------------------------------------------

def score_episode(state_dict: dict) -> dict:
    """Return a normalized score dict suitable for OpenEnv leaderboards.

    Returns
    -------
    dict with keys: task_id, score (0.01-0.99), correct (bool), steps_used, invalid_actions
    """
    submitted = state_dict.get("submitted_answer")
    truth = state_dict.get("correct_answer", "")
    correct = is_correct(submitted, truth) if submitted is not None else False

    return {
        "task_id": state_dict.get("question_id", ""),
        "score": 0.99 if correct else 0.01,
        "correct": correct,
        "steps_used": state_dict.get("step_count", 0),
        "invalid_actions": state_dict.get("invalid_actions", 0),
    }


# ---------------------------------------------------------------------------
# Batch grading
# ---------------------------------------------------------------------------

@dataclass
class BatchReport:
    """Aggregate statistics across multiple episodes."""

    total_episodes: int
    correct: int
    accuracy: float
    avg_steps: float
    avg_reward: float
    total_invalid_actions: int

    @property
    def summary(self) -> str:
        return (
            f"Batch: {self.correct}/{self.total_episodes} correct "
            f"({self.accuracy:.1%}) | avg steps={self.avg_steps:.1f} | "
            f"avg reward={self.avg_reward:.2f} | "
            f"invalid actions={self.total_invalid_actions}"
        )


def grade_batch(reports: List[EpisodeReport]) -> BatchReport:
    """Aggregate a list of episode reports into a batch summary."""
    n = len(reports)
    if n == 0:
        return BatchReport(0, 0, 0.0, 0.0, 0.0, 0)

    correct = sum(1 for r in reports if r.is_correct)
    return BatchReport(
        total_episodes=n,
        correct=correct,
        accuracy=correct / n,
        avg_steps=sum(r.steps_used for r in reports) / n,
        avg_reward=sum(r.cumulative_reward for r in reports) / n,
        total_invalid_actions=sum(r.invalid_actions for r in reports),
    )
