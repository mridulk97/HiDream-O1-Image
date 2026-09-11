#!/usr/bin/env bash
# Score a trained matting adapter on one or more evaluation datasets.
#
#   bash run_matting_eval.sh /scratch/.../adapters/step_5000.pth
#
# Runs each dataset separately and completely, and lays the results out as:
#
#   <EVAL_OUT>/
#     aim500/
#       individual_samples/<id>.png    the predicted matte on its own
#       grid_samples/<id>.png          RGB | RGB+box | generated | GT
#       metrics.json
#       shuffled_image/ shuffled_box/  controls, only when gaps are requested
#     am2k_val/
#       ...
#
# Evaluation uses the COMPLETE dataset by default -- 500 for AIM-500, 200 for
# AM-2k validation. EVAL_NUM_SAMPLES exists only for quick checks.
#
# Configuration comes from the checkpoint: resolution, prompt, datasets and
# whether it was trained with a bounding box, so a bbox model is never silently
# scored without one.
#
# TWO RESOLUTIONS, deliberately independent:
#   generation resolution   what the model samples at; defaults to the trained
#                           size, since anything else is off-grid.
#   metric resolution       what predictions and ground truth are compared at.
#                           'native' is Edit2Perceive's protocol -- score at the
#                           ground truth's own size -- and is what makes a
#                           number comparable to published results.
#
#   EVAL_RESOLUTION=512 EVAL_RES_SCOPE=metric   generate at trained size, score at 512
#   EVAL_RESOLUTION=512 EVAL_RES_SCOPE=both     generate AND score at 512
#
# Other variables:
#   EVAL_DATASETS     space-separated, default from the checkpoint.
#                     Benchmarks: aim500 am2k_val. In-domain: d646 am2k
#   EVAL_OUT          output root (default: <run dir>/eval_<step>)
#   EVAL_NUM_SAMPLES  subset for a quick check; unset means the whole dataset
#   EVAL_STEPS        sampler steps, default 28
#   EVAL_GUIDANCE     CFG scale, default 1.0
#   EVAL_GAPS=1       also sample the shuffled-image and shuffled-box controls
set -euo pipefail

adapter="${1:-}"
if [[ -z "$adapter" ]]; then
  echo "No adapter path given." >&2
  echo "Usage: bash run_matting_eval.sh /path/to/adapters/step_N.pth" >&2
  echo "(if you used a shell variable like \$CK, it was empty -- pass the path" >&2
  echo " literally, or set the variable in the same shell)" >&2
  exit 2
fi
[[ -f "$adapter" ]] || { echo "No such adapter: $adapter" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$here"
python_bin="${MATTING_PYTHON:-/home/mridul/.conda/envs/hidream/bin/python}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/projects/ml4science/HF_CACHE}"

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
trained_res=$(sed -n 1p <<<"$meta"); use_bbox=$(sed -n 2p <<<"$meta")
bbox_res=$(sed -n 3p <<<"$meta"); prompt=$(sed -n 5p <<<"$meta")
datasets="${EVAL_DATASETS:-$(sed -n 4p <<<"$meta")}"

scope="${EVAL_RES_SCOPE:-metric}"
if [[ -n "${EVAL_RESOLUTION:-}" ]]; then
  case "$scope" in
    metric) gen_res="$trained_res";     metric_res="$EVAL_RESOLUTION" ;;
    both)   gen_res="$EVAL_RESOLUTION"; metric_res="$EVAL_RESOLUTION" ;;
    *) echo "EVAL_RES_SCOPE must be 'metric' or 'both', got '$scope'" >&2; exit 2 ;;
  esac
else
  gen_res="$trained_res"; metric_res="native"
fi

step=$(basename "$adapter" .pth)
out="${EVAL_OUT:-$(dirname "$(dirname "$adapter")")/eval_${step}}"
steps="${EVAL_STEPS:-28}"; guidance="${EVAL_GUIDANCE:-1.0}"
mkdir -p "$out"

echo "adapter:      $adapter"
echo "datasets:     $datasets"
echo "generate at:  ${gen_res}px$([[ "$gen_res" != "$trained_res" ]] && echo "   (trained at ${trained_res}px -- OFF-GRID)")"
echo "metrics at:   ${metric_res}$([[ "$metric_res" == native ]] && echo "  (E2P protocol)")"
echo "bbox:         $([[ "$use_bbox" == "1" ]] && echo "yes (layout at ${bbox_res}px)" || echo no)"
echo "output:       $out"
echo

bbox_args=(); [[ "$use_bbox" == "1" ]] && bbox_args=(--use_bbox --bbox_resolution "$bbox_res")

for ds in $datasets; do
  echo "================ $ds ================"
  dsout="$out/$ds"; mkdir -p "$dsout"
  # 0 = the whole dataset. A subset is only for quick checks.
  subset="${EVAL_NUM_SAMPLES:-0}"
  nsamp="${EVAL_NUM_SAMPLES:-1000000}"
  common=(--adapter_path "$adapter" --size "$gen_res" --num_samples "$nsamp"
          --datasets "$ds" --steps "$steps" --guidance_scale "$guidance"
          --overfit_samples "$subset")
  [[ -n "$prompt" ]] && common+=(--prompt "$prompt")

  run_infer () {
    local dir="$1"; shift
    if [[ -f "$dir/results.json" ]]; then echo "[skip] $(basename "$dir") already sampled"
    else
      echo "[infer] $(basename "$dir")"
      # tee, not a plain redirect: a full pass is 700 images over a couple of
      # hours, and the progress bar has to reach the terminal. The log keeps a
      # copy, carriage returns and all.
      set -o pipefail
      "$python_bin" -m matting.sample_matting "${common[@]}" "${bbox_args[@]}" \
        --output_dir "$dir" "$@" 2>&1 | tee "$dir.log" \
        || { echo "  FAILED -- see $dir.log" >&2; exit 1; }
    fi
  }

  run_infer "$dsout"
  cmp_args=()
  if [[ "${EVAL_GAPS:-0}" == "1" ]]; then
    run_infer "$dsout/shuffled_image" --shuffle_conditions
    cmp_args=(--compare_dir "$dsout/shuffled_image" --compare_label "shuffled image")
    [[ "$use_bbox" == "1" ]] && run_infer "$dsout/shuffled_box" --box_from shuffled
  fi

  "$python_bin" -m matting.evaluate_matting --pred_dir "$dsout" \
    --datasets "$ds" --resolution "$gen_res" --metric_resolution "$metric_res" \
    --num_samples "$nsamp" --overfit_samples "$subset" \
    --output "$dsout/metrics.json" "${cmp_args[@]}"

  if [[ "$use_bbox" == "1" && "${EVAL_GAPS:-0}" == "1" ]]; then
    echo "--- box gap ---"
    "$python_bin" -m matting.evaluate_matting --pred_dir "$dsout" \
      --datasets "$ds" --resolution "$gen_res" --metric_resolution "$metric_res" \
      --num_samples "$nsamp" --overfit_samples "$subset" \
      --compare_dir "$dsout/shuffled_box" --compare_label "shuffled box" \
      --output "$dsout/metrics_boxgap.json" | grep -E "mean gap|median ratio|samples moved|NOTE"
  fi
  echo
done
echo "results under $out"
