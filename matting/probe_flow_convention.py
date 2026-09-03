"""Stage 0: verify the flow-matching conventions before writing a trainer.

Three things have to be true for the training loop in `train_hidream_matting.py`
to be wired correctly, and all three are cheap to check with forward passes:

1. ``x_pred`` is an x0-prediction (the clean image), not a velocity. The paper
   says so -- "a linear prediction head maps each output token back to the
   corresponding clean image patch" (2605.11061 3.2) -- and ``pipeline.py``
   documents it, but nothing in the code asserts it.
2. The model takes ``t = 1 - sigma``, not sigma, and rescales by 1000 itself.
3. The noise is scaled by ``NOISE_SCALE = 8.0`` rather than being unit variance.

The test: interpolate a real image toward noise at a range of sigmas, ask the
model to denoise, and measure ``MSE(x_pred, x0)``. Under the correct convention
the error is low and falls monotonically as sigma -> 0.

**This test settles (1) and (2) but NOT (3).** Measured on IP_2.jpg, the two
noise-scale arms interleave -- s=8 wins at sigma 0.75 and 0.5, s=1 wins at 0.25
and below -- so single-step recovery does not discriminate between them. The
likely reason is that every decoder block opens with an RMSNorm, so a global
rescale of the input largely washes out. Use ``probe_sampler.py`` for the noise
scale: over a full 28-step trajectory the wrong scale collapses to flat grey
(std 0.029) while 8.0 produces a real image (std 0.386).

The constant-prediction baseline is reported for scale, but only means something
at low sigma. At sigma ~ 1 the input carries no information about *this* image,
so the model emits some other plausible image and scores worse than the mean --
that is correct behaviour, not a failure.
"""

import argparse
import os
import sys

import einops
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pipeline import (  # noqa: E402
    NOISE_SCALE,
    PATCH_SIZE,
    TENSOR_TRANSFORM,
    build_t2i_text_sample,
)

DEFAULT_MODEL = (
    "/projects/ml4science/HF_CACHE/transformers/"
    "models--HiDream-ai--HiDream-O1-Image-Dev/snapshots/"
    "c0bada0e15c54a9f96a6d1ecc35575b32bc21544"
)


def patchify(img_t):
    """(C, H, W) in [-1, 1] -> (1, num_patches, C*p*p), the model's vinput layout."""
    return einops.rearrange(
        img_t, "C (H p1) (W p2) -> (H W) (C p1 p2)", p1=PATCH_SIZE, p2=PATCH_SIZE
    ).unsqueeze(0)


def unpatchify(x, h_patches, w_patches):
    return einops.rearrange(
        x, "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
        H=h_patches, W=w_patches, p1=PATCH_SIZE, p2=PATCH_SIZE,
    )


def load_image(path, size):
    pil = Image.open(path).convert("RGB")
    # Center crop to square first so the aspect ratio is not distorted -- a
    # squashed image is still a valid input, but it makes the error curve
    # harder to read against the model's natural-image prior.
    w, h = pil.size
    side = min(w, h)
    pil = pil.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    pil = pil.resize((size, size), resample=Image.BICUBIC)
    return TENSOR_TRANSFORM(pil)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--image", default="assets/IP_2.jpg")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--prompt", default="a photo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--noise_scales", type=float, nargs="+", default=[NOISE_SCALE, 1.0],
        help="hypotheses to test; first is the expected one",
    )
    ap.add_argument(
        "--sigmas", type=float, nargs="+",
        default=[0.999, 0.9, 0.75, 0.5, 0.25, 0.1, 0.02],
    )
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    from transformers import AutoProcessor

    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[probe] loading model from {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="cuda"
    )
    model.eval()
    tokenizer = getattr(processor, "tokenizer", processor)

    size = args.size
    h_patches = w_patches = size // PATCH_SIZE

    img = load_image(args.image, size)
    x0 = patchify(img).to(device, dtype)  # (1, HW, C*p*p), values in [-1, 1]
    print(f"[probe] image {args.image} -> {size}x{size}, "
          f"{x0.shape[1]} target tokens, x0 std {x0.float().std().item():.4f}")

    sample = build_t2i_text_sample(
        args.prompt, size, size, tokenizer, processor, model.config
    )
    sample = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
    vinput_mask = sample["vinput_mask"][0]
    print(f"[probe] sequence: {sample['position_ids'].shape[-1]} tokens, "
          f"{int(vinput_mask.sum())} of them target tokens")

    # A model that ignores its input entirely and predicts the per-image mean
    # gets this MSE. Any arm that does not beat it has learned nothing here.
    const_mse = (x0.float() - x0.float().mean()).pow(2).mean().item()

    @torch.no_grad()
    def denoise_mse(sigma, noise_scale, generator):
        eps = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(device)
        z = (1.0 - sigma) * x0.float() + sigma * noise_scale * eps
        t = torch.tensor([1.0 - sigma], device=device)
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            out = model(
                input_ids=sample["input_ids"],
                position_ids=sample["position_ids"],
                vinputs=z.to(dtype),
                timestep=t,
                token_types=sample["token_types"],
                use_flash_attn=False,
            )
        x_pred = out.x_pred[0][vinput_mask].unsqueeze(0).float()
        mse = (x_pred - x0.float()).pow(2).mean().item()
        return mse, x_pred

    results = {}
    previews = {}
    for s in args.noise_scales:
        row = []
        for sigma in args.sigmas:
            g = torch.Generator("cpu").manual_seed(args.seed)
            mse, x_pred = denoise_mse(sigma, s, g)
            row.append(mse)
            previews[(s, sigma)] = x_pred
            print(f"  noise_scale={s:>5.1f}  sigma={sigma:<6.3f}  "
                  f"MSE(x_pred, x0)={mse:.5f}", flush=True)
        results[s] = row

    print()
    print(f"{'sigma':>8} | " + " | ".join(f"s={s:<8.1f}" for s in args.noise_scales))
    print("-" * (10 + 12 * len(args.noise_scales)))
    for i, sigma in enumerate(args.sigmas):
        cells = " | ".join(f"{results[s][i]:<10.5f}" for s in args.noise_scales)
        print(f"{sigma:>8.3f} | {cells}")
    print(f"\nconstant-prediction baseline MSE: {const_mse:.5f}")

    # Verdict. What this test can actually establish: that `x_pred` is an
    # x0-prediction wired to the right tokens with the right timestep. If it
    # were a velocity, or the timestep convention were inverted, the error
    # would not fall smoothly toward zero as the input approaches the clean
    # image. The noise-scale comparison is reported but deliberately not gated
    # on -- see the module docstring and `probe_sampler.py`.
    expected = args.noise_scales[0]
    monotone = all(
        results[expected][i] >= results[expected][i + 1]
        for i in range(len(args.sigmas) - 1)
    )
    recovers = results[expected][-1] < 0.05 * const_mse
    print(f"\nerror decreases monotonically as sigma->0 : {monotone}")
    print(f"recovers x0 at low sigma (< 5% of const)  : {recovers} "
          f"({results[expected][-1]:.5f} vs {const_mse:.5f})")
    print(f"\nGATE (x0 head + timestep convention): "
          f"{'PASS' if (monotone and recovers) else 'INSPECT'}")
    print("noise scale is NOT decided here -- run probe_sampler.py")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for (s, sigma), x_pred in previews.items():
            arr = unpatchify(x_pred.cpu(), h_patches, w_patches)[0]
            arr = ((arr + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)
            Image.fromarray(np.round(arr * 255).astype(np.uint8)).save(
                os.path.join(args.out_dir, f"s{s:g}_sigma{sigma:g}.png")
            )
        print(f"\nwrote {len(previews)} previews to {args.out_dir}")


if __name__ == "__main__":
    main()
