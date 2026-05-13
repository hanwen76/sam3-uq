from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SamPrediction:
    masks: np.ndarray
    scores: np.ndarray
    boxes_xyxy: np.ndarray | None = None
    presence_score: float | None = None
    prompt_name: str = "unknown"
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class UQResult:
    scores: dict[str, float]
    instance_scores: list[dict[str, float]]
    pixel_uncertainty: np.ndarray
    consensus_mask: np.ndarray
    sam_predictions: dict[str, SamPrediction]
