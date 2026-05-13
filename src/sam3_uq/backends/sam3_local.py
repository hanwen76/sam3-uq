from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from sam3_uq.prompts import Prompt
from sam3_uq.types import SamPrediction

from .base import SamBackend


class LocalSam3Backend(SamBackend):
    def __init__(
        self,
        sam3_root: str | Path,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        checkpoint_path: str | None = None,
    ):
        root = Path(sam3_root).expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        bpe_path = root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.exists():
            bpe_path = root / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        kwargs = {"bpe_path": str(bpe_path)}
        if checkpoint_path:
            kwargs["checkpoint_path"] = checkpoint_path
            kwargs["load_from_HF"] = False
        self.model = build_sam3_image_model(**kwargs)
        self.processor = Sam3Processor(
            self.model,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    def predict(self, image: np.ndarray, prompt: Prompt) -> SamPrediction:
        pil_image = _to_pil(image)
        width, height = pil_image.size
        state = self.processor.set_image(pil_image)
        self.processor.reset_all_prompts(state)

        if prompt.kind == "text":
            state = self.processor.set_text_prompt(prompt=str(prompt.value), state=state)
        elif prompt.kind == "box":
            state = self.processor.add_geometric_prompt(
                box=_xyxy_to_normalized_cxcywh(prompt.value, width, height),
                label=True,
                state=state,
            )
        else:
            raise ValueError(f"Unsupported SAM3 prompt kind: {prompt.kind}")

        masks = _tensor_to_numpy(state.get("masks"))
        scores = _tensor_to_numpy(state.get("scores"))
        boxes = _tensor_to_numpy(state.get("boxes"))
        if masks.ndim == 4:
            masks = masks[:, 0]
        return SamPrediction(
            masks=masks.astype(bool),
            scores=scores.astype(np.float32),
            boxes_xyxy=boxes.astype(np.float32) if boxes is not None else None,
            presence_score=float(scores.mean()) if scores.size else 0.0,
            prompt_name=prompt.name,
        )


def _to_pil(image: np.ndarray) -> Image.Image:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = arr - arr.min()
        max_v = arr.max()
        if max_v > 0:
            arr = arr / max_v
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr[..., :3])


def _tensor_to_numpy(value):
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _xyxy_to_normalized_cxcywh(box, width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    cx = ((x0 + x1) / 2.0) / width
    cy = ((y0 + y1) / 2.0) / height
    bw = max(1.0, x1 - x0 + 1.0) / width
    bh = max(1.0, y1 - y0 + 1.0) / height
    return [cx, cy, bw, bh]
