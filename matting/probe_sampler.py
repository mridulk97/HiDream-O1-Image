"""Stage 0b: settle the noise scale and the 1024 question by actually sampling.

`probe_flow_convention.py` measures single-step x0 recovery, and that turned out
not to discriminate between noise scales -- every decoder block opens with an
RMSNorm, so a global rescale of the input largely washes out and both arms land
in a similar place.

Generation does discriminate. The noise scale sets the distribution the sampler
starts from, so a value the model was not trained under compounds over 28 steps
into visible garbage. This script runs the shipped pipeline unchanged except for
the two knobs under test.

It also answers the resolution question in the same pass. `find_closest_resolution`
matches on aspect ratio only, so a 1024x1024 request silently snaps to 2048x2048;
the paper (2605.11061 4.1) trains Stage II at 1024x1024, so 1024 should be a
real operating point. Patching that function out is the only way to ask.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import models.pipeline as pipeline_mod  # noqa: E402

DEFAULT_MODEL = (
    "/projects/ml4science/HF_CACHE/transformers/"
    "models--HiDream-ai--HiDream-O1-Image-Dev/snapshots/"
    "c0bada0e15c54a9f96a6d1ecc35575b32bc21544"
)
DEFAULT_PROMPT = (
    "A close-up photograph of a fluffy white dog sitting on green grass, "
    "sharp fur detail, natural daylight."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance_scale", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--runs", nargs="+", default=["1024:8.0", "1024:1.0", "2048:8.0"],
        help="SIZE:NOISE_SCALE triples to render",
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor

    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    # The pipeline snaps any request to PREDEFINED_RESOLUTIONS, whose smallest
    # square is 2048. Identity here is what lets us ask for 1024 at all.
    pipeline_mod.find_closest_resolution = lambda w, h: (w, h)

    print(f"[probe] loading model from {args.model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    for run in args.runs:
        size_s, scale_s = run.split(":")
        size, scale = int(size_s), float(scale_s)
        print(f"\n[probe] === {size}x{size}, noise_scale={scale} ===", flush=True)
        img = pipeline_mod.generate_image(
            model=model,
            processor=processor,
            prompt=args.prompt,
            height=size,
            width=size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            noise_scale_start=scale,
            noise_scale_end=scale,
        )
        path = os.path.join(args.out_dir, f"gen_{size}_s{scale:g}.png")
        img.save(path)

        # A collapsed sample is usually obvious in the statistics before it is
        # obvious to the eye: near-zero variance (flat) or saturated extremes.
        import numpy as np
        arr = np.asarray(img, dtype=np.float32) / 255.0
        print(f"[probe] wrote {path}  mean={arr.mean():.4f} std={arr.std():.4f} "
              f"frac_saturated={(np.abs(arr - 0.5) > 0.49).mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
