from __future__ import annotations

import itertools

import numpy as np

from .masks import as_bool_mask, boundary, normalize01


def dice(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> float:
    a = as_bool_mask(a)
    b = as_bool_mask(b)
    inter = float(np.logical_and(a, b).sum())
    denom = float(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter + eps) / (denom + eps)


def iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-7) -> float:
    a = as_bool_mask(a)
    b = as_bool_mask(b)
    union = float(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    inter = float(np.logical_and(a, b).sum())
    return (inter + eps) / (union + eps)


def prompt_instability(masks: list[np.ndarray]) -> float:
    if len(masks) < 2:
        return 0.0
    values = [dice(a, b) for a, b in itertools.combinations(masks, 2)]
    return float(1.0 - np.mean(values))


def boundary_disagreement(a: np.ndarray, b: np.ndarray) -> float:
    ba = boundary(a)
    bb = boundary(b)
    return 1.0 - dice(ba, bb)


def pixel_uncertainty_map(model_mask: np.ndarray, sam_masks: list[np.ndarray], consensus: np.ndarray) -> np.ndarray:
    if not sam_masks:
        return np.zeros_like(as_bool_mask(model_mask), dtype=np.float32)
    stack = np.stack([as_bool_mask(m).astype(np.float32) for m in sam_masks], axis=0)
    variance = stack.var(axis=0)
    conflict = np.logical_xor(as_bool_mask(model_mask), as_bool_mask(consensus)).astype(np.float32)
    boundary_conflict = np.logical_xor(boundary(model_mask), boundary(consensus)).astype(np.float32)
    return normalize01(0.5 * variance + 0.35 * conflict + 0.15 * boundary_conflict)
