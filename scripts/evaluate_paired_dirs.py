from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sam3_uq.backends.mock import MockSamBackend
from sam3_uq.backends.sam3_local import LocalSam3Backend
from sam3_uq.evaluator import Sam3UQEvaluator
from sam3_uq.io import load_array, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAM3-UQ on paired image/mask directories.")
    parser.add_argument("--root", required=True, help="Dataset root.")
    parser.add_argument("--split", action="append", default=[], help="Split name, e.g. train. Repeatable.")
    parser.add_argument("--image-template", default="imgs/{split}")
    parser.add_argument("--mask-template", default="masks/all/{split}")
    parser.add_argument("--concept", default="prostate")
    parser.add_argument("--backend", choices=["mock", "sam3-local"], default="sam3-local")
    parser.add_argument("--sam3-root", default="/home/zhanghanwen/sam3-main")
    parser.add_argument("--checkpoint-path", default="/home/zhanghanwen/checkpoints/sam3.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-component-area", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--ext", default="png")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-sample-arrays", action="store_true")
    args = parser.parse_args()

    splits = args.split or ["train", "test"]
    root = Path(args.root)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    shared_backend = None
    if args.backend == "sam3-local":
        shared_backend = LocalSam3Backend(
            sam3_root=args.sam3_root,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
            checkpoint_path=args.checkpoint_path or None,
        )

    aggregate = []
    for split in splits:
        image_dir = root / args.image_template.format(split=split)
        mask_dir = root / args.mask_template.format(split=split)
        split_out = out_root / split
        rows = evaluate_split(split, image_dir, mask_dir, split_out, shared_backend, args)
        write_csv(split_out / "summary.csv", rows)
        aggregate.append(summarize(split, rows))

    aggregate.sort(key=lambda row: row["mean_data_value_proxy"], reverse=True)
    write_csv(out_root / "split_statistics.csv", aggregate)
    print(f"wrote {out_root / 'split_statistics.csv'}")


def evaluate_split(split: str, image_dir: Path, mask_dir: Path, split_out: Path, shared_backend, args) -> list[dict[str, float | str]]:
    if not image_dir.is_dir():
        raise SystemExit(f"Image dir does not exist for split {split}: {image_dir}")
    if not mask_dir.is_dir():
        raise SystemExit(f"Mask dir does not exist for split {split}: {mask_dir}")

    image_paths = sorted(image_dir.glob(f"*.{args.ext}"))
    if args.stride > 1:
        image_paths = image_paths[:: args.stride]
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    rows = []
    print(f"split={split} samples={len(image_paths)}")
    for idx, image_path in enumerate(image_paths, start=1):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            rows.append({"sample": image_path.stem, "status": "missing_mask"})
            continue

        image = load_array(image_path)
        mask = load_array(mask_path)
        backend = shared_backend if shared_backend is not None else MockSamBackend(reference_mask=mask)
        evaluator = Sam3UQEvaluator(backend=backend, min_component_area=args.min_component_area)
        result = evaluator.evaluate(image=image, model_mask=mask, concept=args.concept)

        sample_out = split_out / image_path.stem
        sample_out.mkdir(parents=True, exist_ok=True)
        save_json(sample_out / "scores.json", {"scores": result.scores, "instances": result.instance_scores})
        if args.save_sample_arrays:
            np.save(sample_out / "uncertainty_pixel.npy", result.pixel_uncertainty.astype(np.float32))
            np.save(sample_out / "consensus_mask.npy", result.consensus_mask.astype(np.uint8))

        row = {"sample": image_path.stem, "status": "ok"}
        row.update(result.scores)
        rows.append(row)
        print(
            f"  [{idx}/{len(image_paths)}] {image_path.stem} "
            f"u_image={result.scores['u_image']:.4f} "
            f"value={result.scores['data_value_proxy']:.4f}"
        )
    return rows


def summarize(split: str, rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    ok = [row for row in rows if row.get("status") == "ok"]
    values = np.asarray([float(row["data_value_proxy"]) for row in ok], dtype=np.float32)
    uncertainties = np.asarray([float(row["u_image"]) for row in ok], dtype=np.float32)
    if values.size == 0:
        return {
            "split": split,
            "samples": 0,
            "mean_data_value_proxy": 0.0,
            "median_data_value_proxy": 0.0,
            "p75_data_value_proxy": 0.0,
            "p90_data_value_proxy": 0.0,
            "max_data_value_proxy": 0.0,
            "mean_u_image": 0.0,
        }
    return {
        "split": split,
        "samples": int(values.size),
        "mean_data_value_proxy": float(values.mean()),
        "median_data_value_proxy": float(np.percentile(values, 50)),
        "p75_data_value_proxy": float(np.percentile(values, 75)),
        "p90_data_value_proxy": float(np.percentile(values, 90)),
        "max_data_value_proxy": float(values.max()),
        "mean_u_image": float(uncertainties.mean()),
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    for preferred in ("split", "sample"):
        if preferred in keys:
            keys.remove(preferred)
            keys = [preferred] + keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
