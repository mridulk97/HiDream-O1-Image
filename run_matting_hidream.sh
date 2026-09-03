#!/usr/bin/env bash
# Launch a HiDream-O1-Image matting run.
#
#   MATTING_RUN_NAME=d646_uniform_20k bash run_matting_hidream.sh --max_steps 20000
#
# Anything after the script name is forwarded to the trainer verbatim, so any
# flag in `train_hidream_matting.py --help` overrides the config file.
#
# Environment:
#   MATTING_RUN_NAME   run name and W&B name (default hidream-matting-d646_<ts>)
#   MATTING_CONFIG     default matting/configs/d646_1024_20k.yaml
#   MATTING_RUN_ROOT   default /scratch/mridul/runs/matting/hidream
#   MATTING_RUN_DIR    explicit directory, overrides ROOT/NAME
#   MATTING_RESUME     auto (default) | 0 | /path/to/step_N.pth
#   MATTING_DETACH     0 (default) prints to the terminal; 1 detaches to a log
#   MATTING_NO_WANDB   1 to disable W&B
#   MATTING_SKIP_DATA_SETUP=1 to skip the D-646 extract/verify pass
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

timestamp="$(date +%Y%m%d_%H%M%S)"
run_name="${MATTING_RUN_NAME:-hidream-matting-d646_${timestamp}}"
run_root="${MATTING_RUN_ROOT:-/scratch/mridul/runs/matting/hidream}"
work_dir="${MATTING_RUN_DIR:-${run_root}/${run_name}}"
config="${MATTING_CONFIG:-matting/configs/d646_1024_20k.yaml}"

if [[ ! -f "$config" ]]; then
  echo "Config not found: $config" >&2; exit 2
fi

# The hidream env specifically: this model needs torch 2.12/cu13 and the
# transformers pin, and picking up a different env silently produces a model
# that will not load.
python_bin="${MATTING_PYTHON:-/home/mridul/.conda/envs/hidream/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "Python not executable: $python_bin" >&2; exit 2
fi
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/projects/ml4science/HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/projects/ml4science/HF_CACHE/hub}"

# D-646 lives on scratch and gets reaped: extraction stamps every file with the
# archive's own build date, so an age-based cleanup treats a fresh extract as
# years stale. Same reason PixelDiT re-runs this before every launch.
if [[ "${MATTING_SKIP_DATA_SETUP:-0}" != "1" ]]; then
  setup="/home/mridul/matting/PixelDiT/t2i/setup_d646_data.sh"
  [[ -x "$setup" || -f "$setup" ]] && bash "$setup" || \
    echo "warning: $setup not found, skipping data verification" >&2
fi

mkdir -p "$work_dir"

# Resume. A 20k run is many hours; it will be interrupted at least once.
resume_arg=()
resume_from="${MATTING_RESUME:-auto}"
if [[ "$resume_from" == "auto" ]]; then
  latest="$(ls "$work_dir"/adapters/step_*.pth 2>/dev/null \
            | sed 's/.*step_\([0-9]*\)\.pth/\1 &/' | sort -rn | head -1 | cut -d' ' -f2- || true)"
  [[ -n "$latest" ]] && resume_arg=(--resume_from "$latest")
elif [[ "$resume_from" != "0" ]]; then
  resume_arg=(--resume_from "$resume_from")
fi

wandb_arg=(--wandb --wandb_name "$run_name")
[[ "${MATTING_NO_WANDB:-0}" == "1" ]] && wandb_arg=()
export WANDB_DIR="$work_dir"
export WANDB_CACHE_DIR="$work_dir/.wandb_cache"

runtime="$("$python_bin" -c 'import sys,torch,transformers;print(f"torch={torch.__version__} transformers={transformers.__version__} gpus={torch.cuda.device_count()}")')"

echo "HiDream matting run: $run_name"
echo "Config:              $config"
echo "Run directory:       $work_dir"
echo "Runtime:             $runtime"
if ((${#resume_arg[@]})); then echo "Resuming from:       ${resume_arg[1]}"
else echo "Resuming from:       (fresh start)"; fi
echo "W&B:                 ${MATTING_NO_WANDB:-0}" | sed 's/0$/enabled/;s/1$/disabled/'

# -u: unbuffered. Python block-buffers stdout whenever it is not a TTY, so
# without this both the detached log and a piped foreground run arrive in
# chunks minutes late instead of line by line.
cmd=("$python_bin" -u -m matting.train_hidream_matting
     --config "$config" --run_dir "$work_dir"
     "${resume_arg[@]}" "${wandb_arg[@]}" "$@")

# Default is foreground. Inside screen/tmux that is what you want: the terminal
# IS the log, and screen already provides the persistence that setsid gives.
if [[ "${MATTING_DETACH:-0}" == "1" ]]; then
  # setsid puts the trainer in its own session, so it is not in the launching
  # shell's process group and survives that shell being killed. A previous run
  # here died mid-training when its parent group went away; nohup alone was not
  # enough because that is a group kill, not a SIGHUP.
  setsid nohup "${cmd[@]}" > "$work_dir/stdout.log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$work_dir/train.pid"
  echo "Detached PID:        $pid"
  echo "Log:                 $work_dir/stdout.log"
  echo
  echo "  tail -f $work_dir/stdout.log"
  echo "  kill \$(cat $work_dir/train.pid)"
else
  # Foreground: straight to the terminal, no redirect. This is the mode to use
  # inside screen or tmux -- those already provide the persistence that
  # MATTING_DETACH=1 uses setsid for, and the scrollback is the log.
  # Set MATTING_TEE=1 to also keep stdout.log; nothing else writes it in this
  # mode, and a file copy is what made the last crash diagnosable.
  echo "Mode:                foreground (Ctrl-C stops it)"
  echo
  if [[ "${MATTING_TEE:-0}" == "1" ]]; then
    exec "${cmd[@]}" 2>&1 | tee -a "$work_dir/stdout.log"
  else
    exec "${cmd[@]}" 2>&1
  fi
fi
