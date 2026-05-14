#!/usr/bin/env bash
set -euo pipefail

# Batch runner for prostate-style datasets:
#
#   <DATA_ROOT>/
#   ├── data_npy/
#   ├── label_npy/
#   ├── val_data_npy/
#   └── val_label_npy/
#
# MASK_DIR should normally point to model prediction masks. For a quick pipeline
# smoke test, it can point to label_npy or val_label_npy.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/path/to/prostate}"
CONCEPT="${CONCEPT:-prostate}"
BACKEND="${BACKEND:-mock}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/sam3_uq_out}"

SAM3_ROOT="${SAM3_ROOT:-/path/to/sam3-main}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
DEVICE="${DEVICE:-cuda}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.3}"
MIN_COMPONENT_AREA="${MIN_COMPONENT_AREA:-16}"
LIMIT="${LIMIT:-0}"
EXT="${EXT:-npy}"

if [[ "${IMAGE_DIR:-}" == "" ]]; then
  if [[ -d "${DATA_ROOT}/val_data_npy" ]]; then
    IMAGE_DIR="${DATA_ROOT}/val_data_npy"
  else
    IMAGE_DIR="${DATA_ROOT}/data_npy"
  fi
fi

if [[ "${MASK_DIR:-}" == "" ]]; then
  if [[ -d "${DATA_ROOT}/val_label_npy" ]]; then
    MASK_DIR="${DATA_ROOT}/val_label_npy"
  else
    MASK_DIR="${DATA_ROOT}/label_npy"
  fi
fi

if [[ ! -d "${IMAGE_DIR}" ]]; then
  echo "IMAGE_DIR does not exist: ${IMAGE_DIR}" >&2
  exit 1
fi

if [[ ! -d "${MASK_DIR}" ]]; then
  echo "MASK_DIR does not exist: ${MASK_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "SAM3-UQ batch run"
echo "  repo:       ${REPO_ROOT}"
echo "  data root:  ${DATA_ROOT}"
echo "  image dir:  ${IMAGE_DIR}"
echo "  mask dir:   ${MASK_DIR}"
echo "  output:     ${OUTPUT_DIR}"
echo "  concept:    ${CONCEPT}"
echo "  backend:    ${BACKEND}"
echo "  ext:        ${EXT}"
echo "  limit:      ${LIMIT}"

export PYTHONPATH="${REPO_ROOT}/src:${SAM3_ROOT}:${PYTHONPATH:-}"

count=0
processed=0
missing=0

while IFS= read -r image_path; do
  name="$(basename "${image_path}")"
  stem="${name%.*}"
  mask_path="${MASK_DIR}/${stem}.${EXT}"

  if [[ ! -f "${mask_path}" ]]; then
    echo "skip missing mask: ${mask_path}" >&2
    missing=$((missing + 1))
    continue
  fi

  sample_out="${OUTPUT_DIR}/${stem}"
  mkdir -p "${sample_out}"

  cmd=(
    python3 -m sam3_uq.cli
    --image "${image_path}"
    --mask "${mask_path}"
    --concept "${CONCEPT}"
    --backend "${BACKEND}"
    --output "${sample_out}"
    --min-component-area "${MIN_COMPONENT_AREA}"
  )

  if [[ "${BACKEND}" == "sam3-local" ]]; then
    cmd+=(
      --sam3-root "${SAM3_ROOT}"
      --device "${DEVICE}"
      --confidence-threshold "${CONFIDENCE_THRESHOLD}"
    )
    if [[ "${CHECKPOINT_PATH}" != "" ]]; then
      cmd+=(--checkpoint-path "${CHECKPOINT_PATH}")
    fi
  fi

  echo "run ${stem}"
  "${cmd[@]}"

  processed=$((processed + 1))
  count=$((count + 1))
  if [[ "${LIMIT}" != "0" && "${count}" -ge "${LIMIT}" ]]; then
    break
  fi
done < <(find "${IMAGE_DIR}" -maxdepth 1 -type f -name "*.${EXT}" | sort)

python3 - "${OUTPUT_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = []
for score_path in sorted(out.glob("*/scores.json")):
    with score_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    row = {"sample": score_path.parent.name}
    row.update(payload.get("scores", {}))
    rows.append(row)

if rows:
    keys = ["sample"] + sorted(k for k in rows[0] if k != "sample")
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out / 'summary.csv'}")
else:
    print("no scores found")
PY

echo "done: processed=${processed}, missing_masks=${missing}, output=${OUTPUT_DIR}"
