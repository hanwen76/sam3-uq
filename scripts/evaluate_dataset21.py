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
    parser = argparse.ArgumentParser(
        description="Evaluate SAM3-UQ data value for nnU-Net-style Dataset21* prostate datasets."
    )
    parser.add_argument("--root", default="/mnt/diskB/zhw/A_DATASET")
    parser.add_argument("--pattern", default="Dataset21*")
    parser.add_argument("--concept", default="prostate")
    parser.add_argument("--backend", choices=["mock", "sam3-local"], default="sam3-local")
    parser.add_argument("--sam3-root", default="/home/zhanghanwen/sam3-main")
    parser.add_argument("--checkpoint-path", default="/home/zhanghanwen/checkpoints/sam3.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-threshold", type=float, default=0.3)
    parser.add_argument("--min-component-area", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Per-dataset limit. 0 means all samples.")
    parser.add_argument("--stride", type=int, default=1, help="Evaluate every Nth sorted sample.")
    parser.add_argument("--output", default="/mnt/diskB/zhw/A_DATASET/sam3_uq_dataset21_eval")
    parser.add_argument("--save-sample-arrays", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = sorted([p for p in root.glob(args.pattern) if p.is_dir()])
    if not datasets:
        raise SystemExit(f"No datasets matched {root / args.pattern}")

    backend = None
    if args.backend == "sam3-local":
        backend = LocalSam3Backend(
            sam3_root=args.sam3_root,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
            checkpoint_path=args.checkpoint_path or None,
        )

    aggregate_rows = []
    for dataset_dir in datasets:
        dataset_out = out_root / dataset_dir.name
        dataset_out.mkdir(parents=True, exist_ok=True)
        rows = evaluate_dataset(dataset_dir, dataset_out, backend, args)
        write_csv(dataset_out / "summary.csv", rows)
        aggregate_rows.append(summarize_dataset(dataset_dir.name, rows))

    aggregate_rows.sort(key=lambda row: row["mean_data_value_proxy"], reverse=True)
    write_csv(out_root / "dataset_ranking.csv", aggregate_rows)
    print(f"wrote {out_root / 'dataset_ranking.csv'}")


def evaluate_dataset(dataset_dir: Path, dataset_out: Path, shared_backend, args) -> list[dict[str, float | str]]:
    image_dir = dataset_dir / "imagesTr"
    label_dir = dataset_dir / "labelsTr"
    if not image_dir.is_dir() or not label_dir.is_dir():
        print(f"skip {dataset_dir.name}: missing imagesTr or labelsTr")
        return []

    images = sorted(image_dir.glob("*_0000.png"))
    if args.stride > 1:
        images = images[:: args.stride]
    if args.limit > 0:
        images = images[: args.limit]

    rows = []
    print(f"dataset={dataset_dir.name} samples={len(images)}")
    for idx, image_path in enumerate(images, start=1):
        label_name = image_path.name.replace("_0000.png", ".png")
        mask_path = label_dir / label_name
        if not mask_path.exists():
            rows.append({"sample": image_path.stem, "status": "missing_mask"})
            continue

        image = load_array(image_path)
        mask = load_array(mask_path)
        backend = shared_backend if shared_backend is not None else MockSamBackend(reference_mask=mask)
        evaluator = Sam3UQEvaluator(backend=backend, min_component_area=args.min_component_area)
        result = evaluator.evaluate(image=image, model_mask=mask, concept=args.concept)

        sample_name = image_path.stem.replace("_0000", "")
        sample_out = dataset_out / sample_name
        sample_out.mkdir(parents=True, exist_ok=True)
        save_json(
            sample_out / "scores.json",
            {"scores": result.scores, "instances": result.instance_scores},
        )
        if args.save_sample_arrays:
            np.save(sample_out / "uncertainty_pixel.npy", result.pixel_uncertainty.astype(np.float32))
            np.save(sample_out / "consensus_mask.npy", result.consensus_mask.astype(np.uint8))

        row = {"sample": sample_name, "status": "ok"}
        row.update(result.scores)
        rows.append(row)
        print(
            f"  [{idx}/{len(images)}] {sample_name} "
            f"u_image={result.scores['u_image']:.4f} "
            f"value={result.scores['data_value_proxy']:.4f}"
        )
    return rows


def summarize_dataset(name: str, rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    values = np.asarray([float(row["data_value_proxy"]) for row in ok_rows], dtype=np.float32)
    uncertainties = np.asarray([float(row["u_image"]) for row in ok_rows], dtype=np.float32)
    if values.size == 0:
        return {
            "dataset": name,
            "samples": 0,
            "mean_data_value_proxy": 0.0,
            "mean_u_image": 0.0,
            "p75_data_value_proxy": 0.0,
            "max_data_value_proxy": 0.0,
        }
    return {
        "dataset": name,
        "samples": int(values.size),
        "mean_data_value_proxy": float(values.mean()),
        "mean_u_image": float(uncertainties.mean()),
        "p75_data_value_proxy": float(np.percentile(values, 75)),
        "max_data_value_proxy": float(values.max()),
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    if "dataset" in keys:
        keys.remove("dataset")
        keys = ["dataset"] + keys
    if "sample" in keys:
        keys.remove("sample")
        keys = ["sample"] + keys
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
