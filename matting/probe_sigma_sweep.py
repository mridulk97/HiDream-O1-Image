"""Where along the noise schedule is the model actually any good?

Training reports one pooled x0 error, which hides the shape that decides whether
sampling will work. Sampling starts at sigma ~ 0.999 and walks down; if the
model is weak at the top of that range the first Euler step is taken from a bad
x0 estimate, the trajectory leaves the data manifold, and some fraction of
samples saturate to a flat 0 or 1 -- a good pooled error with garbage samples.

Measured on the sigmoid/logit-normal arm at step 250: x0_mse 0.0100 at sigma
0.05, rising to 0.0475 at 0.90 and 0.2215 at 0.99 -- a 4.7x degradation over the
last tenth of the schedule. Four of eight sampled mattes came out flat.

Mind the units when comparing to the published baselines. x0 lives in [-1, 1]
and MATTING.md's baselines (all-black 0.266, constant-mean 0.193) are [0, 1]
alpha, so an x0_mse must be divided by 4 before the comparison. Under that
conversion the sigma-0.99 figure is 0.055, comfortably *below* all-black -- weak
relative to the rest of the schedule, not useless. The `vs all-black` column
does this conversion; the raw x0_mse column does not.

`sigmoid(randn)` is logit-normal and puts ~1.4% of its mass above sigma 0.9 and
almost none above 0.99, so the top of the trajectory is genuinely undertrained
however long the run goes. Whether that is what makes sampling diverge is the
open question -- UniPC is a multistep solver and can oscillate from a poor
start, and guidance scale matters too. Uniform sampling (the paper's SFT choice,
2605.11061 4.2, "balanced timestep coverage") is the arm this script exists to
compare against.

Read the top row first. If sigma 0.99 is near the all-black baseline, sampling
will diverge no matter how good the pooled number looks.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from matting.data import DEFAULT_D646_ROOT, DEFAULT_PROMPT, D646MattingDataset  # noqa: E402
from matting.sample_builder import build_matting_sample, patchify  # noqa: E402
from matting.sample_matting import load_adapter  # noqa: E402
from matting.train_hidream_matting import DEFAULT_MODEL, NOISE_SCALE  # noqa: E402

ALL_BLACK_MSE = 0.266      # MATTING.md trivial baselines, in [0, 1] alpha space
CONST_MEAN_MSE = 0.193


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_path", default=None,
                    help="omit to probe the pretrained model with no adapter")
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--d646_root", default=DEFAULT_D646_ROOT)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--overfit_samples", type=int, default=32)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.999, 0.99, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.05])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    from transformers import AutoProcessor

    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    device, dtype = torch.device("cuda"), torch.bfloat16
    ds = D646MattingDataset(root=args.d646_root, resolution=args.size,
                            overfit_samples=args.overfit_samples,
                            prompt=args.prompt)
    n = min(args.num_samples, len(ds))

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="cuda")
    model_config = model.config
    if args.adapter_path:
        model, _ = load_adapter(model, args.adapter_path)
    else:
        print("[sweep] no adapter -- probing the pretrained checkpoint")
    model.eval()

    # Built once: the layout is deterministic given (image, prompt, size).
    prepared = []
    for i in range(n):
        item = ds[i]
        prepared.append((
            build_matting_sample(
                cond_image=item["condition"], prompt=args.prompt,
                height=args.size, width=args.size, tokenizer=tokenizer,
                processor=processor, model_config=model_config,
                device=device, dtype=dtype),
            patchify(item["alpha_rgb"]).unsqueeze(0).to(device).float(),
        ))

    print(f"\n{'sigma':>7} | {'x0_mse':>9} | vs all-black")
    print("-" * 38)
    rows = []
    for sigma in args.sigmas:
        total = 0.0
        for sample, x0 in prepared:
            g = torch.Generator().manual_seed(args.seed)
            eps = torch.randn(x0.shape, generator=g).to(device)
            z = (1.0 - sigma) * x0 + sigma * NOISE_SCALE * eps
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype,
                                                 cache_enabled=False):
                out = model(
                    input_ids=sample["input_ids"],
                    position_ids=sample["position_ids"],
                    vinputs=torch.cat([z.to(dtype), sample["ref_patches"]], dim=1),
                    timestep=torch.tensor([1.0 - sigma], device=device),
                    token_types=sample["token_types"],
                    pixel_values=sample["pixel_values"],
                    image_grid_thw=sample["image_grid_thw"],
                    use_flash_attn=False)
            xp = out.x_pred[0][sample["vinput_mask"][0]].unsqueeze(0)
            total += F.mse_loss(xp[:, : sample["tgt_image_len"]].float(), x0).item()
        # x0 lives in [-1, 1]; the published baselines are [0, 1] alpha, so
        # halve the scale before comparing (a factor 2 in range is 4 in MSE).
        mse = total / n
        ratio = (mse / 4.0) / ALL_BLACK_MSE
        flag = "  <-- near-useless" if ratio > 0.5 else ""
        print(f"{sigma:>7.3f} | {mse:>9.5f} | {ratio:>6.1%}{flag}")
        rows.append({"sigma": sigma, "x0_mse": mse, "frac_of_all_black": ratio})

    top = rows[0]
    print(f"\nsampling starts at sigma {top['sigma']}, where x0_mse is "
          f"{top['x0_mse']:.5f} ({top['frac_of_all_black']:.0%} of the all-black "
          f"baseline).")
    print("A trajectory launched from that estimate is the thing to fix first."
          if top["frac_of_all_black"] > 0.5 else
          "The top of the schedule beats the trivial baseline, so divergence "
          "is not explained by this alone -- check the solver and guidance too.")

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump({"adapter": args.adapter_path, "rows": rows}, fh, indent=2)


if __name__ == "__main__":
    main()
