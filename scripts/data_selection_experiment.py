from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class RunResult:
    strategy: str
    k: int
    seed: int
    epochs: int
    train_samples: int
    test_samples: int
    final_loss: float
    mean_dice: float
    mean_iou: float


class ProstatePairs(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, samples: list[str]):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        name = self.samples[idx]
        image = np.asarray(Image.open(self.image_dir / f"{name}.png").convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(self.mask_dir / f"{name}.png").convert("L"), dtype=np.float32)
        mask = (mask > 0).astype(np.float32)
        image_t = torch.from_numpy(image).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return image_t, mask_t


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        y = self.up2(x3)
        y = self.dec2(torch.cat([y, x2], dim=1))
        y = self.up1(y)
        y = self.dec1(torch.cat([y, x1], dim=1))
        return self.out(y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether SAM3-UQ data value improves downstream segmentation.")
    parser.add_argument("--train-image-dir", default="/mnt/diskB/zhw/A_DATASET/Prostate_resize/imgs/train")
    parser.add_argument("--train-mask-dir", default="/mnt/diskB/zhw/A_DATASET/Prostate_resize/masks/all/train")
    parser.add_argument("--test-image-dir", default="/mnt/diskB/zhw/A_DATASET/Prostate_resize/imgs/test")
    parser.add_argument("--test-mask-dir", default="/mnt/diskB/zhw/A_DATASET/Prostate_resize/masks/all/test")
    parser.add_argument("--uq-summary", default="/mnt/diskB/zhw/SAM3_UQ_RESULTS/sam3_uq_real_prostate_eval_all/train/summary.csv")
    parser.add_argument("--output", default="/mnt/diskB/zhw/SAM3_UQ_RESULTS/data_selection_real_prostate")
    parser.add_argument("--strategies", default="top,bottom,random")
    parser.add_argument("--ks", default="100,300")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_config(out / "config.json", vars(args))

    rows = read_uq_rows(Path(args.uq_summary))
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    test_samples = sorted(p.stem for p in Path(args.test_image_dir).glob("*.png"))
    if args.max_test > 0:
        test_samples = test_samples[: args.max_test]

    results: list[RunResult] = []
    for k in ks:
        for strategy in strategies:
            for seed in seeds:
                selected = select_samples(rows, strategy, k, seed)
                run_name = f"{strategy}_k{k}_seed{seed}"
                run_dir = out / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "selected_samples.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
                result = train_and_eval(
                    train_image_dir=Path(args.train_image_dir),
                    train_mask_dir=Path(args.train_mask_dir),
                    test_image_dir=Path(args.test_image_dir),
                    test_mask_dir=Path(args.test_mask_dir),
                    train_samples=selected,
                    test_samples=test_samples,
                    run_dir=run_dir,
                    strategy=strategy,
                    k=k,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    base_channels=args.base_channels,
                    num_workers=args.num_workers,
                    device=args.device,
                )
                results.append(result)
                write_results(out / "results.csv", results)


def read_uq_rows(path: Path) -> list[dict[str, float | str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            rows.append({"sample": row["sample"], "value": float(row["data_value_proxy"])})
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    return rows


def select_samples(rows: list[dict[str, float | str]], strategy: str, k: int, seed: int) -> list[str]:
    if strategy == "top":
        chosen = sorted(rows, key=lambda r: float(r["value"]), reverse=True)[:k]
    elif strategy == "bottom":
        chosen = sorted(rows, key=lambda r: float(r["value"]))[:k]
    elif strategy == "random":
        rng = random.Random(seed)
        chosen = rows.copy()
        rng.shuffle(chosen)
        chosen = chosen[:k]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return [str(row["sample"]) for row in chosen]


def train_and_eval(
    train_image_dir: Path,
    train_mask_dir: Path,
    test_image_dir: Path,
    test_mask_dir: Path,
    train_samples: list[str],
    test_samples: list[str],
    run_dir: Path,
    strategy: str,
    k: int,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    base_channels: int,
    num_workers: int,
    device: str,
) -> RunResult:
    set_seed(seed)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    train_loader = DataLoader(
        ProstatePairs(train_image_dir, train_mask_dir, train_samples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device_t.type == "cuda",
    )
    test_loader = DataLoader(
        ProstatePairs(test_image_dir, test_mask_dir, test_samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_t.type == "cuda",
    )

    model = TinyUNet(base=base_channels).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    epoch_rows = []
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for image, mask in train_loader:
            image = image.to(device_t, non_blocking=True)
            mask = mask.to(device_t, non_blocking=True)
            logits = model(image)
            loss = bce(logits, mask) + soft_dice_loss(logits, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else 0.0
        epoch_rows.append({"epoch": epoch, "loss": final_loss})
        print(f"{strategy} k={k} seed={seed} epoch={epoch}/{epochs} loss={final_loss:.4f}", flush=True)

    mean_dice, mean_iou = evaluate(model, test_loader, device_t)
    write_csv(run_dir / "train_log.csv", epoch_rows)
    torch.save(model.state_dict(), run_dir / "model.pt")
    print(
        f"RESULT strategy={strategy} k={k} seed={seed} dice={mean_dice:.4f} iou={mean_iou:.4f}",
        flush=True,
    )
    return RunResult(
        strategy=strategy,
        k=k,
        seed=seed,
        epochs=epochs,
        train_samples=len(train_samples),
        test_samples=len(test_samples),
        final_loss=final_loss,
        mean_dice=mean_dice,
        mean_iou=mean_iou,
    )


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    dices = []
    ious = []
    for image, mask in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = torch.sigmoid(model(image)) > 0.5
        target = mask > 0.5
        inter = (pred & target).sum(dim=(1, 2, 3)).float()
        pred_sum = pred.sum(dim=(1, 2, 3)).float()
        target_sum = target.sum(dim=(1, 2, 3)).float()
        union = (pred | target).sum(dim=(1, 2, 3)).float()
        dice = torch.where(pred_sum + target_sum > 0, 2 * inter / (pred_sum + target_sum), torch.ones_like(inter))
        iou = torch.where(union > 0, inter / union, torch.ones_like(inter))
        dices.extend(dice.detach().cpu().tolist())
        ious.extend(iou.detach().cpu().tolist())
    return float(np.mean(dices)), float(np.mean(ious))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_results(path: Path, rows: list[RunResult]) -> None:
    write_csv(path, [asdict(row) for row in rows])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
