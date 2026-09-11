"""Standard matting metrics over a directory of predictions.

Self-contained: the metric implementations live in `matting/metrics.py`,
vendored from P3M-Net via Edit2Perceive, so evaluation imports nothing from a
sibling repo. See that module for the licence and for the scale convention --
alpha in [0, 1], SAD in units of 1000 pixels, and `mse` identical to the
`generated_mse` the trainer reports.

Two things this driver adds over a plain metric call:

* **Per dataset.** We train on a D-646 + AM-2k mixture whose halves differ
  sharply in difficulty (AM-2k mattes are 0.7-3.9% soft pixels, D-646's median
  6.9%), so a pooled number lets the easy half mask regressions in the hard one.
* **MAD bucketed by ground-truth alpha.** The whole-image mean is dominated by
  the ~82% of pixels that are flat background or flat foreground. The soft
  region is a few percent of the frame and is the entire matting problem;
  measured on a real checkpoint it ran 44x worse than the background while the
  headline MSE looked excellent.

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
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from matting.data import DEFAULT_PROMPT, build_dataset  # noqa: E402
from matting.metrics import compute_matting_metrics  # noqa: E402

# Alpha buckets. Whole-image means hide the soft region, which is a few percent
# of pixels and the entire matting problem -- MATTING.md measured MAD 30-56x
# worse inside it than outside while the headline number looked fine.
BUCKETS = (("background", 0.0, 0.02), ("near-transparent", 0.02, 0.3),
           ("half", 0.3, 0.7), ("near-opaque", 0.7, 0.98),
           ("foreground", 0.98, 1.01))


def _load_pred(path, shape):
    """Load a prediction and resize it to the ground truth's shape.

    Bilinear with `align_corners=True`, matching Edit2Perceive's `resize_tensor`
    (`utils/eval_matting.py:12`) exactly. PIL's BILINEAR is not the same
    operation -- it does not offer the align_corners convention -- and on a
    ~1.6x upsample the difference is visible in the gradient and connectivity
    terms, which is precisely where a matting metric is meant to be sensitive.
    """
    import torch
    import torch.nn.functional as F
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if a.shape != shape:
        t = torch.from_numpy(a)[None, None].float()
        a = F.interpolate(t, size=(shape[0], shape[1]), mode="bilinear",
                          align_corners=True).squeeze().numpy()
    # E2P zeroes non-finite values before scoring rather than letting them
    # propagate into the sums.
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def evaluate_dir(pred_dir, gt_by_id):
    """Metrics per sample for every prediction present in `pred_dir`."""
    from tqdm import tqdm

    rows = []
    # Connectivity and the Gaussian-derivative gradient are the expensive terms
    # and they run at native resolution, so 700 images is minutes, not seconds.
    for sid, (gt, dataset) in tqdm(gt_by_id.items(), total=len(gt_by_id),
                                   desc="scoring", unit="img", dynamic_ncols=True,
                                   leave=False):
        # Predictions live in individual_samples/; the flat layout is still
        # accepted so older result directories keep scoring.
        path = os.path.join(pred_dir, "individual_samples", f"{sid}.png")
        if not os.path.exists(path):
            path = os.path.join(pred_dir, f"{sid}.png")
        if not os.path.exists(path):
            continue
        pred = _load_pred(path, gt.shape)
        # whole=True needs no trimap, which is why none is synthesised here.
        mse, mad, sad, grad, conn = compute_matting_metrics(pred, gt, whole=True)
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
                    choices=["d646", "am2k", "aim500", "am2k_val"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--overfit_samples", type=int, default=32)
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--metric_resolution", default="native",
                    help="resolution the METRICS are computed at: 'native' for "
                         "the ground truth's own size (Edit2Perceive's protocol, "
                         "and what makes numbers comparable to published ones), "
                         "or an integer like 512/1024. Independent of the "
                         "resolution the model generated at")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    # concat, not the training interleave: evaluation wants every image once.
    ds = build_dataset(names=args.datasets, resolution=args.resolution,
                       split=args.split, overfit_samples=args.overfit_samples,
                       prompt=args.prompt, concat=True)
    n = min(args.num_samples, len(ds))
    if args.num_samples > len(ds):
        print(f"note: asked for {args.num_samples} but "
              f"{'+'.join(args.datasets)} has {len(ds)}; scoring all of them")
    want_native = str(args.metric_resolution).lower() == "native"
    have_native = hasattr(ds, "native_alpha")
    if want_native and not have_native:
        # Composited datasets have no single "native" size -- the composite is
        # built at the foreground's resolution, which differs per sample -- so
        # fall back rather than silently scoring something else.
        print(f"note: {'+'.join(args.datasets)} has no native-resolution ground "
              f"truth; scoring at {args.resolution}px instead")
    gt_by_id = {}
    for i in range(n):
        it = ds[i]
        if want_native and have_native:
            gt, _tri = ds.native_alpha(i)
        else:
            gt = ((it["alpha_rgb"][0].float() + 1) / 2).clamp(0, 1).numpy()
            if not want_native:
                target = int(args.metric_resolution)
                if gt.shape[0] != target:
                    gt = np.asarray(
                        Image.fromarray((gt * 255).astype(np.uint8)).resize(
                            (target, target), Image.BILINEAR),
                        dtype=np.float32) / 255.0
        gt_by_id[it["sample_id"]] = (gt, it.get("dataset", "?"))
    shapes = {v[0].shape for v in gt_by_id.values()}
    where = ("ground-truth native resolution (E2P protocol)" if want_native and have_native
             else f"{args.metric_resolution}px")
    print(f"metrics computed at {where}"
          + (f" -- sizes {sorted(shapes)[:3]}" if len(shapes) <= 3 else
             f" -- {len(shapes)} distinct sizes"))

    lines = []

    def out(text=""):
        """Print and record, so metrics.txt is exactly what you saw."""
        print(text)
        lines.append(text)

    rows = evaluate_dir(args.pred_dir, gt_by_id)
    if not rows:
        raise SystemExit(f"no predictions matched in {args.pred_dir}")

    metric_keys = ["mse", "mad", "sad", "grad", "conn"]
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    out(f"\n{len(rows)} predictions from {args.pred_dir}")
    # Column order follows Edit2Perceive's own eval printout
    # (`utils/eval_matting.py:167`): MSE, MAD, SAD, Grad, Conn.
    out(f"\n{'dataset':>10} {'n':>4} {'MSE':>9} {'MAD':>9} {'SAD':>9} "
        f"{'Grad':>9} {'Conn':>9} {'vs all-black':>13}")
    out("-" * 82)
    for name in sorted(by_ds) + (["ALL"] if len(by_ds) > 1 else []):
        rs = rows if name == "ALL" else by_ds[name]
        a = _agg(rs, metric_keys)
        blk = np.mean([r["all_black_mse"] for r in rs])
        out(f"{name:>10} {len(rs):>4} {a['mse']:>9.5f} {a['mad']:>9.5f} "
            f"{a['sad']:>9.3f} {a['grad']:>9.3f} {a['conn']:>9.3f} "
            f"{a['mse'] / max(blk, 1e-9):>12.0%}")

    out(f"\nMAD by ground-truth alpha (the soft buckets are the task):")
    out(f"{'bucket':>18} {'MAD':>9} {'% of pixels':>12}")
    out("-" * 42)
    for name, _, _ in BUCKETS:
        a = _agg(rows, [f"mad_{name}"])[f"mad_{name}"]
        px = np.nanmean([r[f"px_{name}"] for r in rows])
        out(f"{name:>18} {a:>9.5f} {px:>11.2%}")

    summary = {"pred_dir": os.path.abspath(args.pred_dir), "n": len(rows),
               "overall": _agg(rows, metric_keys),
               "per_dataset": {k: _agg(v, metric_keys) for k, v in by_ds.items()},
               "per_sample": rows}

    if args.compare_dir:
        crows = evaluate_dir(args.compare_dir, gt_by_id)
        c = _agg(crows, metric_keys)
        base = _agg(rows, metric_keys)
        gap = (c["mse"] - base["mse"]) / c["mse"] if c["mse"] else 0.0

        # The mean gap is a bad statistic on its own: it is dominated by the
        # few samples where the swapped condition happens to be catastrophically
        # wrong. Measured on one checkpoint, a 76.5% mean gap came entirely from
        # 2 of 16 samples moving ~25x while the other 14 moved 1.0x -- and the
        # same checkpoint scored 1.2% on an overlapping set that happened not to
        # include them. Report the per-sample distribution alongside it.
        cm = {r["sample_id"]: r["mse"] for r in rows}
        sm = {r["sample_id"]: r["mse"] for r in crows}
        shared = [k for k in cm if k in sm]
        ratios = sorted(sm[k] / max(cm[k], 1e-9) for k in shared)
        affected = sum(1 for r in ratios if r > 1.5)
        med = ratios[len(ratios) // 2] if ratios else float("nan")

        out(f"\n{args.compare_label} ({len(crows)} preds): MSE {c['mse']:.5f} "
            f"vs {base['mse']:.5f}")
        out(f"  mean gap        {gap * 100:>6.1f}%"
            + ("   <-- under 10%" if gap < 0.10 else ""))
        out(f"  median ratio    {med:>6.2f}x   (1.00 = swapping changed nothing)")
        out(f"  samples moved   {affected:>6}/{len(shared)} by more than 1.5x"
            f"   max {ratios[-1] if ratios else 0:.1f}x")
        if affected and affected <= max(1, len(shared) // 5):
            out(f"  NOTE: the mean is carried by {affected} outlier(s); read the "
                f"median and the moved count, not the mean")
        summary["compare"] = {"label": args.compare_label, "dir": args.compare_dir,
                              "overall": c, "gap": gap, "median_ratio": med,
                              "n_moved_1.5x": affected, "n_shared": len(shared)}

    out_json = args.output or os.path.join(args.pred_dir, "metrics.json")
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)

    # The same table as text, so a result can be read or pasted without
    # re-deriving it from the JSON. Header records what the numbers mean:
    # SAD/Grad/Conn are pixel sums and shift by ~8x between 512 and native, so
    # a number is meaningless without the resolution it was computed at.
    out_txt = os.path.splitext(out_json)[0] + ".txt"
    with open(out_txt, "w") as fh:
        fh.write(f"# {' '.join(args.datasets)}  |  metrics at {args.metric_resolution}"
                 f"  |  generated at {args.resolution}px\n")
        fh.write(f"# predictions: {os.path.abspath(args.pred_dir)}\n")
        fh.write("# MSE/MAD are per-pixel means; SAD/Grad/Conn are sums and "
                 "scale with pixel count.\n")
        fh.write("\n".join(lines).lstrip("\n") + "\n")
    print(f"\nwrote {out_json}")
    print(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
