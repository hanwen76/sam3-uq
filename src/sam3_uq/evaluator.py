from __future__ import annotations

import numpy as np

from .backends.base import SamBackend
from .masks import as_bool_mask, connected_components, fuse_masks
from .metrics import boundary_disagreement, dice, iou, pixel_uncertainty_map, prompt_instability
from .prompts import build_prompts
from .types import SamPrediction, UQResult


class Sam3UQEvaluator:
    def __init__(
        self,
        backend: SamBackend,
        weights: dict[str, float] | None = None,
        min_component_area: int = 16,
    ):
        self.backend = backend
        self.weights = weights or {
            "model_sam": 0.35,
            "prompt": 0.25,
            "boundary": 0.20,
            "presence": 0.20,
        }
        self.min_component_area = min_component_area

    def evaluate(self, image: np.ndarray, model_mask: np.ndarray, concept: str) -> UQResult:
        model_mask = as_bool_mask(model_mask)
        prompts = build_prompts(
            model_mask=model_mask,
            concept=concept,
            min_component_area=self.min_component_area,
        )
        predictions = {prompt.name: self.backend.predict(image, prompt) for prompt in prompts}
        representative_masks = [_representative_mask(pred, model_mask.shape) for pred in predictions.values()]
        consensus = fuse_masks(representative_masks)

        u_model_sam = 1.0 - dice(model_mask, consensus)
        u_prompt = prompt_instability(representative_masks)
        u_boundary = boundary_disagreement(model_mask, consensus)
        score_values = [float(pred.scores.max()) for pred in predictions.values() if pred.scores.size]
        u_presence = 1.0 - float(np.mean(score_values)) if score_values else 1.0
        u_presence = float(np.clip(u_presence, 0.0, 1.0))

        u_image = _weighted_sum(
            self.weights,
            {
                "model_sam": u_model_sam,
                "prompt": u_prompt,
                "boundary": u_boundary,
                "presence": u_presence,
            },
        )
        uncertainty_map = pixel_uncertainty_map(model_mask, representative_masks, consensus)
        instances = self._instance_scores(model_mask, consensus, representative_masks)
        data_value = _data_value_proxy(u_image, model_mask, uncertainty_map)

        return UQResult(
            scores={
                "u_image": float(u_image),
                "u_model_sam": float(u_model_sam),
                "u_prompt": float(u_prompt),
                "u_boundary": float(u_boundary),
                "u_presence": float(u_presence),
                "data_value_proxy": float(data_value),
                "sam_prompt_count": float(len(predictions)),
            },
            instance_scores=instances,
            pixel_uncertainty=uncertainty_map,
            consensus_mask=consensus,
            sam_predictions=predictions,
        )

    def _instance_scores(
        self,
        model_mask: np.ndarray,
        consensus: np.ndarray,
        sam_masks: list[np.ndarray],
    ) -> list[dict[str, float]]:
        rows = []
        for idx, comp in enumerate(connected_components(model_mask, min_area=self.min_component_area)):
            comp_sam_iou = iou(comp, consensus)
            prompt_scores = [iou(comp, sam_mask) for sam_mask in sam_masks]
            rows.append(
                {
                    "instance_id": float(idx),
                    "area": float(comp.sum()),
                    "iou_to_consensus": float(comp_sam_iou),
                    "u_instance": float(1.0 - comp_sam_iou),
                    "prompt_iou_mean": float(np.mean(prompt_scores)) if prompt_scores else 0.0,
                }
            )
        return rows


def _representative_mask(prediction: SamPrediction, shape: tuple[int, int]) -> np.ndarray:
    masks = np.asarray(prediction.masks)
    if masks.size == 0:
        return np.zeros(shape, dtype=bool)
    if masks.ndim == 2:
        return masks.astype(bool)
    if masks.ndim == 3:
        return masks.any(axis=0)
    raise ValueError(f"Unsupported mask shape from SAM backend: {masks.shape}")


def _weighted_sum(weights: dict[str, float], values: dict[str, float]) -> float:
    total_w = sum(float(weights.get(k, 0.0)) for k in values)
    if total_w <= 0:
        raise ValueError("At least one uncertainty weight must be positive")
    return sum(float(weights.get(k, 0.0)) * float(v) for k, v in values.items()) / total_w


def _data_value_proxy(u_image: float, model_mask: np.ndarray, uncertainty_map: np.ndarray) -> float:
    area_ratio = float(as_bool_mask(model_mask).mean())
    boundary_or_error_mass = float(uncertainty_map.mean())
    return float(np.clip(0.70 * u_image + 0.20 * boundary_or_error_mass + 0.10 * min(area_ratio * 4.0, 1.0), 0.0, 1.0))
