from __future__ import annotations

from collections import deque

import numpy as np


def as_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 0


def connected_components(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    mask = as_bool_mask(mask)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            comp = np.zeros_like(mask, dtype=bool)
            q: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            area = 0
            while q:
                cy, cx = q.popleft()
                comp[cy, cx] = True
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if area >= min_area:
                comps.append(comp)
    return comps


def mask_to_box_xyxy(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    mask = as_bool_mask(mask)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w - 1, int(xs.max()) + pad)
    y1 = min(h - 1, int(ys.max()) + pad)
    return x0, y0, x1, y1


def erode4(mask: np.ndarray) -> np.ndarray:
    mask = as_bool_mask(mask)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]
    return center & up & down & left & right


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = as_bool_mask(mask)
    return mask & ~erode4(mask)


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    max_v = float(arr.max()) if arr.size else 0.0
    min_v = float(arr.min()) if arr.size else 0.0
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_v) / (max_v - min_v)


def fuse_masks(predictions: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    masks = []
    for pred in predictions:
        arr = np.asarray(pred)
        if arr.ndim == 3:
            arr = arr.any(axis=0)
        masks.append(as_bool_mask(arr))
    if not masks:
        raise ValueError("Cannot fuse an empty mask list")
    stack = np.stack(masks, axis=0).astype(np.float32)
    return stack.mean(axis=0) >= threshold
