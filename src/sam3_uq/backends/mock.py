from __future__ import annotations

import numpy as np

from sam3_uq.masks import as_bool_mask
from sam3_uq.prompts import Prompt
from sam3_uq.types import SamPrediction

from .base import SamBackend


class MockSamBackend(SamBackend):
    def __init__(self, reference_mask: np.ndarray, shift: int = 1):
        self.reference_mask = as_bool_mask(reference_mask)
        self.shift = shift

    def predict(self, image: np.ndarray, prompt: Prompt) -> SamPrediction:
        del image
        mask = self.reference_mask.copy()
        if prompt.kind == "text":
            mask = np.roll(mask, self.shift, axis=1)
            score = 0.72
        elif prompt.kind == "box":
            x0, y0, x1, y1 = prompt.value
            box_mask = np.zeros_like(mask, dtype=bool)
            box_mask[y0 : y1 + 1, x0 : x1 + 1] = True
            mask = mask & box_mask
            score = 0.84
        else:
            score = 0.5
        return SamPrediction(
            masks=mask[None, ...],
            scores=np.asarray([score], dtype=np.float32),
            boxes_xyxy=None,
            presence_score=score,
            prompt_name=prompt.name,
        )
