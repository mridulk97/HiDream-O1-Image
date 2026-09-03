"""Stage 1 gate: does `build_matting_sample` actually drive the model?

The parity test (`tests/test_sample_builder.py`) proves the token layout matches
what inference builds, but it never runs the transformer. Two things can still be
wrong in a way parity cannot see, and both are fatal to training:

1. **The target slice.** `x_pred[vinput_mask][:tgt_image_len]` has to be the
   prediction for the noisy target, not for the reference block or an off-by-one
   window into it. Checked by denoising a nearly-clean alpha matte: if the slice
   is right, the model returns approximately what it was given.

2. **The reference stream.** `token_type=2` patches are the entire mechanism by
   which the conditioning image reaches the model. If that stream were dropped,
   mis-positioned, or masked out, training would still run and the loss would
   still fall -- the model would just learn to ignore the condition, which is
   exactly the failure PixelDiT hit (MATTING.md, "Flow regime"). Checked by
   swapping in a *different* image's reference and measuring how much the
   prediction moves. A dead stream gives a change of zero.

Test 2 runs before any training, on the pretrained model, so it measures the
plumbing rather than matting skill. Sensitivity to the reference is expected
here: HiDream is pretrained for image-conditioned editing.
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, "/home/mridul/matting/PixelDiT/t2i")

from matting.sample_builder import build_matting_sample, patchify  # noqa: E402

DEFAULT_MODEL = (
    "/projects/ml4science/HF_CACHE/transformers/"
    "models--HiDream-ai--HiDream-O1-Image-Dev/snapshots/"
    "c0bada0e15c54a9f96a6d1ecc35575b32bc21544"
)
D646_ROOT = "/scratch/mridul/data/matting/distinctions-646"
NOISE_SCALE = 8.0
PROMPT = "Transform to matting map while maintaining original composition"


def load_pairs(root, resolution, n):
    from diffusion.data.datasets.pixdit_datasets import Distinctions646MattingDataset

    ds = Distinctions646MattingDataset(
        data_dir=[root],
        resolution=resolution,
        extra={"split": "train", "overfit_samples": n, "cache_composites": True},
    )
    pairs = []
    for i in range(min(n, len(ds))):
        rec = ds[i]
        alpha_rgb, condition, sample_id = rec[0], rec[8], rec[6]
        pairs.append((sample_id, condition, alpha_rgb))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--d646_root", default=D646_ROOT)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--num_pairs", type=int, default=4)
    ap.add_argument("--sigma_recover", type=float, default=0.02)
    ap.add_argument("--sigma_cond", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoProcessor

    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    device, dtype = torch.device("cuda"), torch.bfloat16

    print(f"[wiring] loading {args.num_pairs} D-646 pairs at {args.size}px", flush=True)
    pairs = load_pairs(args.d646_root, args.size, args.num_pairs)
    for sid, cond, alpha in pairs:
        print(f"  {sid}: cond{tuple(cond.shape)} alpha{tuple(alpha.shape)} "
              f"alpha range [{alpha.min():.2f}, {alpha.max():.2f}]")

    print(f"[wiring] loading model", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="cuda"
    )
    model.eval()

    def build(cond):
        return build_matting_sample(
            cond_image=cond, prompt=PROMPT, height=args.size, width=args.size,
            tokenizer=tokenizer, processor=processor, model_config=model.config,
            device=device, dtype=dtype,
        )

    @torch.no_grad()
    def predict(sample, x0, sigma, seed):
        g = torch.Generator("cpu").manual_seed(seed)
        eps = torch.randn(x0.shape, generator=g, dtype=torch.float32).to(device)
        z = (1.0 - sigma) * x0 + sigma * NOISE_SCALE * eps
        vinputs = torch.cat([z.to(dtype), sample["ref_patches"]], dim=1)
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            out = model(
                input_ids=sample["input_ids"],
                position_ids=sample["position_ids"],
                vinputs=vinputs,
                timestep=torch.tensor([1.0 - sigma], device=device),
                token_types=sample["token_types"],
                pixel_values=sample["pixel_values"],
                image_grid_thw=sample["image_grid_thw"],
                use_flash_attn=False,
            )
        x_pred = out.x_pred[0][sample["vinput_mask"][0]].unsqueeze(0).float()
        return x_pred[:, : sample["tgt_image_len"]]

    print(f"\n[wiring] TEST 1 -- target slice, sigma={args.sigma_recover}")
    print(f"{'sample':>16} | {'MSE(x_pred, alpha)':>19} | {'const baseline':>14} "
          f"| {'x better':>9}")
    print("-" * 68)
    recover_ok = True
    for sid, cond, alpha in pairs:
        sample = build(cond)
        x0 = patchify(alpha).unsqueeze(0).to(device).float()
        pred = predict(sample, x0, args.sigma_recover, args.seed)
        mse = (pred - x0).pow(2).mean().item()
        const = (x0 - x0.mean()).pow(2).mean().item()
        print(f"{sid:>16} | {mse:>19.5f} | {const:>14.5f} | {const / mse:>8.1f}x")
        # A mis-sliced window would land near the constant baseline, not several
        # times under it, so "beats constant by 5x" separates wiring from skill.
        # The model has never seen a matte, so exact recovery is not the claim.
        recover_ok &= mse < 0.2 * const

    # Where the input stops being signal. z = (1-s)*x0 + s*8*eps, so the signal
    # and noise magnitudes cross at s = std(x0) / (8 + std(x0)) -- around 0.08
    # for these mattes. Above that the target is buried, which is worth knowing
    # before choosing how to sample sigma.
    stds = [patchify(a).float().std().item() for _, _, a in pairs]
    mean_std = sum(stds) / len(stds)
    print(f"\n[wiring] alpha std {mean_std:.3f} -> SNR=1 at sigma "
          f"{mean_std / (NOISE_SCALE + mean_std):.3f}; above that the input is "
          f"noise-dominated")

    print(f"\n[wiring] TEST 2 -- reference stream alive, sigma={args.sigma_cond}")
    print("swapping the conditioning image at a FIXED noise seed. The null is the")
    print("same conditioning at the same seed, which is deterministic, so d_same")
    print("is numerical zero and any real dependence shows as d_cond >> d_same.")
    print("(An earlier version compared against a fresh noise draw instead. That")
    print(" is not a fair null: at sigma=0.6 with noise scale 8 the input is")
    print(" ~92% noise by magnitude, so reseeding moves the output more than any")
    print(" conditioning change could, whether or not the stream works.)")
    print(f"\n{'sample':>16} | {'d_cond':>10} | {'d_same':>10} | {'pred var':>10} "
          f"| {'d_cond/var':>10}")
    print("-" * 70)
    cond_alive = True
    for i, (sid, cond, alpha) in enumerate(pairs):
        other_cond = pairs[(i + 1) % len(pairs)][1]
        x0 = patchify(alpha).unsqueeze(0).to(device).float()

        p_right = predict(build(cond), x0, args.sigma_cond, args.seed)
        p_wrong = predict(build(other_cond), x0, args.sigma_cond, args.seed)
        p_again = predict(build(cond), x0, args.sigma_cond, args.seed)

        d_cond = (p_right - p_wrong).pow(2).mean().item()
        d_same = (p_right - p_again).pow(2).mean().item()
        var = p_right.var().item()
        print(f"{sid:>16} | {d_cond:>10.6f} | {d_same:>10.6f} | {var:>10.5f} "
              f"| {d_cond / max(var, 1e-12):>10.4f}")
        # Alive means the reference measurably changes the prediction, well
        # above the deterministic-rerun floor and not a rounding artefact.
        cond_alive &= (d_cond > 100 * max(d_same, 1e-9)) and (d_cond > 0.01 * var)

    print(f"\ntarget slice recovers alpha        : {recover_ok}")
    print(f"reference stream moves prediction  : {cond_alive}")
    print(f"\nGATE: {'PASS' if (recover_ok and cond_alive) else 'INSPECT'}")


if __name__ == "__main__":
    main()
