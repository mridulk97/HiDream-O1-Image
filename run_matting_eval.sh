#!/usr/bin/env bash
# Score a trained matting adapter: run inference, compute metrics, print a table.
#
#   bash run_matting_eval.sh /scratch/.../adapters/step_2000.pth
#
# Configuration is read from the checkpoint itself -- resolution, prompt,
# datasets, and whether it was trained with a bounding box -- so a bbox model is
# never silently scored without one. Override any of it with the env vars below.
#
#   EVAL_OUT          output root (default: <adapter dir>/../eval_<step>)
#   EVAL_DATASETS     override the checkpoint's datasets, e.g. "d646 am2k"
#   EVAL_NUM_SAMPLES  default 16
#   EVAL_SPLIT        train (default) | test
#   EVAL_STEPS        sampler steps, default 28
#   EVAL_GUIDANCE     CFG scale, default 1.0
#   EVAL_SKIP_GAPS=1  score only the main run, no shuffled comparisons
set -euo pipefail

adapter="${1:?Usage: bash run_matting_eval.sh /path/to/adapters/step_N.pth}"
[[ -f "$adapter" ]] || { echo "No such adapter: $adapter" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"
python_bin="${MATTING_PYTHON:-/home/mridul/.conda/envs/hidream/bin/python}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/projects/ml4science/HF_CACHE}"

# Read what the checkpoint says it was trained as.
meta=$("$python_bin" - "$adapter" <<'PY'
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("metadata", {})
print(m.get("resolution", 1024))
print(int(bool(m.get("use_bbox", False))))
print(m.get("bbox_resolution", 512))
print(" ".join(m.get("datasets", ["d646"])) or "d646")
print(m.get("prompt", ""))
PY
)
resolution=$(sed -n 1p <<<"$meta")
use_bbox=$(sed -n 2p <<<"$meta")
bbox_res=$(sed -n 3p <<<"$meta")
datasets="${EVAL_DATASETS:-$(sed -n 4p <<<"$meta")}"
prompt=$(sed -n 5p <<<"$meta")

step=$(basename "$adapter" .pth)
out="${EVAL_OUT:-$(dirname "$(dirname "$adapter")")/eval_${step}}"
n="${EVAL_NUM_SAMPLES:-16}"
split="${EVAL_SPLIT:-train}"
steps="${EVAL_STEPS:-28}"
guidance="${EVAL_GUIDANCE:-1.0}"
mkdir -p "$out"

echo "adapter:    $adapter"
echo "resolution: $resolution   datasets: $datasets   split: $split"
echo "samples:    $n   sampler steps: $steps   guidance: $guidance"
echo "bbox:       $([[ "$use_bbox" == "1" ]] && echo "yes (layout at ${bbox_res}px)" || echo no)"
echo "output:     $out"
echo

bbox_args=(); [[ "$use_bbox" == "1" ]] && bbox_args=(--use_bbox --bbox_resolution "$bbox_res")
common=(--adapter_path "$adapter" --size "$resolution" --num_samples "$n"
        --datasets $datasets --steps "$steps" --guidance_scale "$guidance"
        --overfit_samples "$n")
[[ -n "$prompt" ]] && common+=(--prompt "$prompt")

run_infer () {  # name, extra args...
  local name="$1"; shift
  if [[ -f "$out/$name/results.json" ]]; then
    echo "[skip] $name already sampled"
  else
    echo "[infer] $name"
    "$python_bin" -m matting.sample_matting "${common[@]}" "${bbox_args[@]}" \
      --output_dir "$out/$name" "$@" > "$out/$name.log" 2>&1 \
      || { echo "  FAILED -- see $out/$name.log" >&2; exit 1; }
  fi
}

run_infer correct
if [[ "${EVAL_SKIP_GAPS:-0}" != "1" ]]; then
  # Swapping the reference photo: does the model use the image at all?
  run_infer shuffled_image --shuffle_conditions
  # Swapping only the box: does the model use the box at all? Only meaningful
  # for a bbox-trained checkpoint, and the number that decides whether bbox
  # conditioning did anything, since the box is redundant on this data.
  [[ "$use_bbox" == "1" ]] && run_infer shuffled_box --box_from shuffled
fi

echo
cmp_args=()
[[ "${EVAL_SKIP_GAPS:-0}" != "1" ]] && cmp_args=(--compare_dir "$out/shuffled_image"
                                                 --compare_label "shuffled image")
"$python_bin" -m matting.evaluate_matting --pred_dir "$out/correct" \
  --datasets $datasets --split "$split" --resolution "$resolution" \
  --num_samples "$n" --overfit_samples "$n" "${cmp_args[@]}"

if [[ "$use_bbox" == "1" && "${EVAL_SKIP_GAPS:-0}" != "1" ]]; then
  echo
  echo "=== box gap: same weights, a different sample's box ==="
  "$python_bin" -m matting.evaluate_matting --pred_dir "$out/correct" \
    --datasets $datasets --split "$split" --resolution "$resolution" \
    --num_samples "$n" --overfit_samples "$n" \
    --compare_dir "$out/shuffled_box" --compare_label "shuffled box" \
    --output "$out/correct/metrics_boxgap.json" | tail -6
fi

echo
echo "predictions and metrics under $out"
