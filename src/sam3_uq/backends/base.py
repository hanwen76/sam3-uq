from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sam3_uq.prompts import Prompt
from sam3_uq.types import SamPrediction


class SamBackend(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray, prompt: Prompt) -> SamPrediction:
        raise NotImplementedError
