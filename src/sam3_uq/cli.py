from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .backends.mock import MockSamBackend
from .backends.sam3_local import LocalSam3Backend
from .evaluator import Sam3UQEvaluator
from .io import load_array, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate segmentation uncertainty with SAM3-UQ.")
    parser.add_argument("--image", required=True, help="Input image path: png/jpg/npy.")
    parser.add_argument("--mask", required=True, help="Model prediction mask path: png/npy.")
    parser.add_argument("--concept", required=True, help="Target concept prompt for SAM3.")
    parser.add_argument("--backend", choices=["mock", "sam3-local"], default="mock")
    parser.add_argument("--sam3-root", default="../sam3-main", help="Local SAM3 repository path.")
    parser.add_argument("--checkpoint-path", default=None, help="Optional local SAM3 checkpoint path.")
    parser.add_argument("--device", default="cuda", help="SAM3 device, e.g. cuda or cpu.")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-component-area", type=int, default=16)
    parser.add_argument("--output", required=True, help="Output directory.")
    args = parser.parse_args()

    image = load_array(args.image)
    mask = load_array(args.mask)
    if args.backend == "mock":
        backend = MockSamBackend(reference_mask=mask)
    else:
        backend = LocalSam3Backend(
            sam3_root=args.sam3_root,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
            checkpoint_path=args.checkpoint_path,
        )

    evaluator = Sam3UQEvaluator(backend=backend, min_component_area=args.min_component_area)
    result = evaluator.evaluate(image=image, model_mask=mask, concept=args.concept)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(
        out / "scores.json",
        {
            "scores": result.scores,
            "instances": result.instance_scores,
        },
    )
    np.save(out / "uncertainty_pixel.npy", result.pixel_uncertainty.astype(np.float32))
    np.save(out / "consensus_mask.npy", result.consensus_mask.astype(np.uint8))
    for name, pred in result.sam_predictions.items():
        np.save(out / f"sam_masks_{name}.npy", pred.masks.astype(np.uint8))

    print(f"Wrote SAM3-UQ outputs to {out}")
    print(f"u_image={result.scores['u_image']:.4f}, data_value_proxy={result.scores['data_value_proxy']:.4f}")


if __name__ == "__main__":
    main()
