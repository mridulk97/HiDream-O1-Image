"""Does the model actually read the bounding box?

This is the most important number in the bbox change, because the box is
*redundant* on this training data. D-646 and AM-2k both have one foreground per
image, so the model can solve every training sample perfectly while ignoring
the box entirely -- and a metric that only reports `generated_mse` would never
show it.

Three conditions on the same weights and the same images:

  correct   the sample's own box
  shuffled  a different sample's box (deterministic derangement)
  none      no layout image at all, K=1

Read it like this:

  correct ~= shuffled   the box is being ignored. The change is doing nothing,
                        whatever the headline MSE says.
  correct << shuffled   the box is load-bearing.
  none much worse       the box is contributing information beyond localisation.

It also checks for the degenerate solution. With one object per image and a
pixel-exact box, `alpha = box interior` scores well on these datasets and is
useless anywhere else. `iou_with_box` near 1.0 with a *rectangular* prediction
means the model learned to fill the rectangle rather than to matte; raise
`bbox_jitter` if so.
"""

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from matting.bbox import render_layout_image  # noqa: E402
from matting.data import DEFAULT_PROMPT, build_dataset  # noqa: E402
from matting.sample_matting import load_adapter, sample_one  # noqa: E402
from matting.train_hidream_matting import DEFAULT_MODEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--datasets", nargs="+", default=["d646"],
                    choices=["d646", "am2k"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--overfit_samples", type=int, default=32)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--bbox_resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoProcessor

    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    device, dtype = torch.device("cuda"), torch.bfloat16
    # jitter 0: the honest box, so a gap reflects the box's content and not
    # noise added on top of it.
    ds = build_dataset(names=args.datasets, resolution=args.size,
                       overfit_samples=args.overfit_samples, prompt=args.prompt,
                       use_bbox=True, bbox_jitter=0.0)
    n = min(args.num_samples, len(ds))
    items = [ds[i] for i in range(n)]
    boxes = [it["bbox"] for it in items]

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="cuda")
    model_config = model.config
    if args.adapter_path:
        model, _ = load_adapter(model, args.adapter_path)
    else:
        print("[box] no adapter -- probing the pretrained checkpoint")
    model.eval()

    def run(item, box, seed):
        layout = None if box is None else render_layout_image(
            box, args.bbox_resolution, args.bbox_resolution)
        return sample_one(
            model, item["condition"], args.prompt, args.size, tokenizer,
            processor, model_config, device, dtype, args.steps,
            args.guidance_scale, args.shift, seed,
            layout_image=layout, layout_size=args.bbox_resolution)

    res = {"correct": [], "shuffled": [], "none": []}
    box_iou = []
    print(f"\n{'sample':>26} {'correct':>9} {'shuffled':>9} {'none':>9} {'IoU(box)':>9}")
    print("-" * 68)
    for i, it in enumerate(items):
        gt = ((it["alpha_rgb"][0].float() + 1) / 2).clamp(0, 1).numpy()
        seed = args.seed + i          # fixed per sample, shared across conditions
        row = {}
        for cond, box in (("correct", boxes[i]),
                          ("shuffled", boxes[(i + 1) % n]),
                          ("none", None)):
            a = run(it, box, seed)
            row[cond] = float(np.mean((a - gt) ** 2))
            res[cond].append(row[cond])
            if cond == "correct":
                # Did it just fill the rectangle?
                x1, y1, x2, y2 = boxes[i]
                m = np.zeros_like(gt)
                m[int(y1 * gt.shape[0]):int(y2 * gt.shape[0]),
                  int(x1 * gt.shape[1]):int(x2 * gt.shape[1])] = 1.0
                pb, mb = a > 0.5, m > 0.5
                inter, union = (pb & mb).sum(), (pb | mb).sum()
                box_iou.append(float(inter / max(union, 1)))
        print(f"{it['sample_id']:>26} {row['correct']:>9.5f} "
              f"{row['shuffled']:>9.5f} {row['none']:>9.5f} {box_iou[-1]:>9.3f}")

    c, s, nn = (float(np.mean(res[k])) for k in ("correct", "shuffled", "none"))
    gap = (s - c) / s if s > 0 else 0.0
    print(f"\n{'mean':>26} {c:>9.5f} {s:>9.5f} {nn:>9.5f} {np.mean(box_iou):>9.3f}")
    print(f"\nbox gap (shuffled vs correct): {gap * 100:.1f}%")
    print(f"box vs no-box:                 {(nn - c) / max(nn, 1e-9) * 100:+.1f}%")
    if gap < 0.10:
        print("\nGATE: FAIL -- the model is ignoring the box "
              "(<10% gap). The bbox conditioning is not doing anything.")
    elif np.mean(box_iou) > 0.9:
        print("\nGATE: SUSPECT -- high IoU with the box interior. Check the "
              "predictions are mattes and not filled rectangles; raise bbox_jitter.")
    else:
        print("\nGATE: PASS -- the box is load-bearing.")


if __name__ == "__main__":
    main()
