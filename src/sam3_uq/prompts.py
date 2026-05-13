from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .masks import connected_components, mask_to_box_xyxy


@dataclass(frozen=True)
class Prompt:
    name: str
    kind: str
    value: object


def build_prompts(
    model_mask: np.ndarray,
    concept: str,
    min_component_area: int = 16,
    box_pad: int = 2,
) -> list[Prompt]:
    prompts = [Prompt(name="text", kind="text", value=concept)]
    for idx, comp in enumerate(connected_components(model_mask, min_area=min_component_area)):
        box = mask_to_box_xyxy(comp, pad=box_pad)
        if box is not None:
            prompts.append(Prompt(name=f"box_{idx:03d}", kind="box", value=box))
    return prompts
