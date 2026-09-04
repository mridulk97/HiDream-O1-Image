"""Standard matting metrics over a directory of predictions.

**The metric implementations are Edit2Perceive's, not ours.** SAD / MSE / MAD /
Grad / Conn are the canonical P3M-Net implementations that matting papers
report, and reimplementing them invites subtle mismatches that silently make our
numbers incomparable to published ones. `compute_matting_metrics`
(`Edit2Perceive/utils/metric.py:689`) is imported directly.

What is written here is the driver, because E2P's `eval_matting.py` does not fit:

* it expects a *provided* trimap, and neither of our datasets ships one -- D-646
  has none at all, AM-2k's are left inside the zips by `setup_am2k_data.sh`;
* its `gen_trimap` picks a random kernel size and iteration count. That is
  reasonable augmentation during training and wrong for evaluation, where two
  runs of the same checkpoint must produce the same number. It also needs cv2,
  which this environment does not have;
* it reports one pooled figure, and we train on a mixture whose halves differ
  sharply in difficulty.

So the trimap is built deterministically from the ground-truth alpha with
PixelDiT's `unknown_band` (separable max-pooling, no cv2, fixed radius), and
results are reported per dataset.

Note on scale: metrics take alpha in **[0, 1]**, not [0, 255]. Under that
convention `mse_whole` is exactly the `generated_mse` the trainer reports, and
comparable to MATTING.md's trivial baselines. SAD is in units of 1000 pixels.

Usage::

    python -m matting.evaluate_matting --pred_dir <run>/eval_correct \\
        --compare_dir <run>/eval_shuffled --compare_label shuffled
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

E2P_PATH = "/home/mridul/matting/Edit2Perceive"
PIXELDIT_T2I = "/home/mridul/matting/PixelDiT/t2i"

from matting.data import DEFAULT_PROMPT, build_dataset  # noqa: E402

# Alpha buckets. Whole-image means hide the soft region, which is a few percent
# of pixels and the entire matting problem -- MATTING.md measured MAD 30-56x
# worse inside it than outside while the headline number looked fine.
BUCKETS = (("background", 0.0, 0.02), ("near-transparent", 0.02, 0.3),
           ("half", 0.3, 0.7), ("near-opaque", 0.7, 0.98),
           ("foreground", 0.98, 1.01))


def _import_metrics():
    if E2P_PATH not in sys.path:
        sys.path.insert(0, E2P_PATH)
    from utils.metric import compute_matting_metrics
    return compute_matting_metrics


def _unknown_band(alpha, radius):
    """Deterministic trimap unknown region, via PixelDiT's separable dilation."""
    if PIXELDIT_T2I not in sys.path:
        sys.path.insert(0, PIXELDIT_T2I)
    from diffusion.model.matting_losses import unknown_band
    t = torch.from_numpy(alpha)[None, None].float()
    return unknown_band(t, radius)[0, 0].numpy()


def _trimap(alpha, radius):
    """0 background / 128 unknown / 255 foreground, as the metrics expect."""
    band = _unknown_band(alpha, radius) > 0.5
    tri = np.where(alpha >= 0.98, 255.0, 0.0)
    tri[band] = 128.0
    return tri.astype(np.float32)


def _load_pred(path, shape):
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if a.shape != shape:
        a = np.asarray(
            Image.fromarray((a * 255).astype(np.uint8)).resize(
                (shape[1], shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
    return a


def evaluate_dir(pred_dir, gt_by_id, radius, compute_matting_metrics):
    """Metrics per sample for every prediction present in `pred_dir`."""
    rows = []
    for sid, (gt, dataset) in gt_by_id.items():
        path = os.path.join(pred_dir, f"{sid}.png")
        if not os.path.exists(path):
            continue
        pred = _load_pred(path, gt.shape)
        mse, mad, sad, grad, conn = compute_matting_metrics(
            pred, gt, _trimap(gt, radius), whole=True)
        row = {"sample_id": sid, "dataset": dataset, "mse": float(mse),
               "mad": float(mad), "sad": float(sad), "grad": float(grad),
               "conn": float(conn),
               # Per-sample baselines: all-black scores mean(gt**2), which is
               # just foreground coverage and varies 0.07-0.60 across D-646, so
               # a single global constant is not a meaningful reference.
               "all_black_mse": float(np.mean(gt ** 2)),
               "const_mean_mse": float(np.mean((gt - gt.mean()) ** 2))}
        for name, lo, hi in BUCKETS:
            m = (gt >= lo) & (gt < hi)
            row[f"mad_{name}"] = float(np.abs(pred[m] - gt[m]).mean()) if m.any() else float("nan")
            row[f"px_{name}"] = float(m.mean())
        rows.append(row)
    return rows


def _agg(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if not np.isnan(r.get(k, np.nan))]
        out[k] = float(np.mean(v)) if v else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--compare_dir", default=None,
                    help="a second prediction dir -- shuffled conditioning, a "
                         "shuffled box, or no box -- to report a gap against")
    ap.add_argument("--compare_label", default="compare")
    ap.add_argument("--datasets", nargs="+", default=["d646"],
                    choices=["d646", "am2k"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--overfit_samples", type=int, default=32)
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--band_radius", type=int, default=10)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    compute_matting_metrics = _import_metrics()

    ds = build_dataset(names=args.datasets, resolution=args.resolution,
                       split=args.split, overfit_samples=args.overfit_samples,
                       prompt=args.prompt)
    n = min(args.num_samples, len(ds))
    gt_by_id = {}
    for i in range(n):
        it = ds[i]
        gt = ((it["alpha_rgb"][0].float() + 1) / 2).clamp(0, 1).numpy()
        gt_by_id[it["sample_id"]] = (gt, it.get("dataset", "?"))

    rows = evaluate_dir(args.pred_dir, gt_by_id, args.band_radius,
                        compute_matting_metrics)
    if not rows:
        raise SystemExit(f"no predictions matched in {args.pred_dir}")

    metric_keys = ["mse", "mad", "sad", "grad", "conn"]
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    print(f"\n{len(rows)} predictions from {args.pred_dir}")
    print(f"\n{'dataset':>10} {'n':>4} {'MSE':>9} {'MAD':>9} {'SAD':>9} "
          f"{'Grad':>9} {'Conn':>9} {'vs all-black':>13}")
    print("-" * 82)
    for name in sorted(by_ds) + (["ALL"] if len(by_ds) > 1 else []):
        rs = rows if name == "ALL" else by_ds[name]
        a = _agg(rs, metric_keys)
        blk = np.mean([r["all_black_mse"] for r in rs])
        print(f"{name:>10} {len(rs):>4} {a['mse']:>9.5f} {a['mad']:>9.5f} "
              f"{a['sad']:>9.3f} {a['grad']:>9.3f} {a['conn']:>9.3f} "
              f"{a['mse'] / max(blk, 1e-9):>12.0%}")

    print(f"\nMAD by ground-truth alpha (the soft buckets are the task):")
    print(f"{'bucket':>18} {'MAD':>9} {'% of pixels':>12}")
    print("-" * 42)
    for name, _, _ in BUCKETS:
        a = _agg(rows, [f"mad_{name}"])[f"mad_{name}"]
        px = np.nanmean([r[f"px_{name}"] for r in rows])
        print(f"{name:>18} {a:>9.5f} {px:>11.2%}")

    summary = {"pred_dir": os.path.abspath(args.pred_dir), "n": len(rows),
               "overall": _agg(rows, metric_keys),
               "per_dataset": {k: _agg(v, metric_keys) for k, v in by_ds.items()},
               "per_sample": rows}

    if args.compare_dir:
        crows = evaluate_dir(args.compare_dir, gt_by_id, args.band_radius,
                             compute_matting_metrics)
        c = _agg(crows, metric_keys)
        base = _agg(rows, metric_keys)
        gap = (c["mse"] - base["mse"]) / c["mse"] if c["mse"] else 0.0
        print(f"\n{args.compare_label} ({len(crows)} preds): MSE {c['mse']:.5f} "
              f"vs {base['mse']:.5f}")
        print(f"gap: {gap * 100:.1f}%"
              + ("   <-- under 10%: the conditioning is not being used"
                 if gap < 0.10 else ""))
        summary["compare"] = {"label": args.compare_label, "dir": args.compare_dir,
                              "overall": c, "gap": gap}

    out = args.output or os.path.join(args.pred_dir, "metrics.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
