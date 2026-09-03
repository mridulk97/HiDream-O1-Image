"""Sample alpha mattes from a trained adapter, and score them.

Training measures a one-step x0 error; that is not the number the task is judged
on. This runs the real sampler -- the same 28-step schedule and the same
`FlowUniPCMultistepScheduler` the shipped pipeline uses -- and reports
`generated_mse` against the ground-truth matte, which is what
`PixelDiT/t2i/MATTING.md` reports and what its baselines are calibrated to
(all-black 0.266, constant-mean 0.193).

`--shuffle_conditions` pairs each sample with a different image's reference. Run
it both ways and compare: the pilot succeeds when correct-condition MSE is
clearly below shuffled-condition MSE. A gap under ~10% is a failed run, not an
early one, however good the training loss looked.

The sampling loop is `models/pipeline.py:generate_image` reduced to this task --
one reference, a fixed size, no resolution snapping (`find_closest_resolution`
matches on aspect ratio only and would send 1024 to 2048), no layout boxes.
"""

import argparse
import json
import os
import sys

import einops
import numpy as np
import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from models.pipeline import DEFAULT_TIMESTEPS, build_scheduler  # noqa: E402
from matting.data import DEFAULT_D646_ROOT, DEFAULT_PROMPT, D646MattingDataset  # noqa: E402
from matting.sample_builder import build_matting_sample, unpatchify  # noqa: E402
from matting.train_hidream_matting import (  # noqa: E402
    DEFAULT_MODEL, FULL_TRAIN_MODULES, NOISE_SCALE, T_EPS, lora_target_modules,
)

PATCH_SIZE = 32


def load_adapter(model, adapter_path):
    """Re-create the training-time module surgery, then load the weights."""
    from peft import LoraConfig, get_peft_model

    ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    rank = meta.get("lora_rank", 16)
    alpha = meta.get("lora_alpha", rank)

    targets = lora_target_modules(model)
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=targets))

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    loaded = len(ckpt["state_dict"])
    # `missing` is every frozen weight, which is expected; `unexpected` is not.
    if unexpected:
        raise RuntimeError(f"adapter has {len(unexpected)} unexpected keys, "
                           f"first few: {unexpected[:5]}")
    print(f"[adapter] step {ckpt.get('step','?')}, loaded {loaded} tensors "
          f"(r={rank}, alpha={alpha})")
    return model, meta


@torch.no_grad()
def sample_one(model, cond_image, prompt, size, tokenizer, processor,
               model_config, device, dtype, steps, guidance_scale, shift, seed):
    """One conditioned sampling trajectory. Returns alpha in [0, 1], (H, W)."""
    h_patches = w_patches = size // PATCH_SIZE

    samples = [build_matting_sample(
        cond_image=cond_image, prompt=prompt, height=size, width=size,
        tokenizer=tokenizer, processor=processor, model_config=model_config,
        device=device, dtype=dtype)]
    if guidance_scale > 1.0:
        samples.append(build_matting_sample(
            cond_image=cond_image, prompt=" ", height=size, width=size,
            tokenizer=tokenizer, processor=processor, model_config=model_config,
            device=device, dtype=dtype))

    timesteps = DEFAULT_TIMESTEPS if steps == len(DEFAULT_TIMESTEPS) else None
    sched = build_scheduler(steps, timesteps, shift, device, "default")

    noise = NOISE_SCALE * torch.randn(
        (1, 3, size, size), generator=torch.Generator("cpu").manual_seed(seed))
    z = einops.rearrange(
        noise, "B C (H p1) (W p2) -> B (H W) (C p1 p2)",
        p1=PATCH_SIZE, p2=PATCH_SIZE).to(device, dtype)

    for step_t in sched.timesteps:
        t = torch.tensor([1.0 - step_t.float().item() / 1000.0], device=device)
        sigma = max(step_t.float().item() / 1000.0, T_EPS)

        v_preds = []
        for sample in samples:
            vinputs = torch.cat([z, sample["ref_patches"]], dim=1)
            with torch.autocast(device.type, dtype=dtype, cache_enabled=False):
                out = model(
                    input_ids=sample["input_ids"],
                    position_ids=sample["position_ids"],
                    vinputs=vinputs, timestep=t,
                    token_types=sample["token_types"],
                    pixel_values=sample["pixel_values"],
                    image_grid_thw=sample["image_grid_thw"],
                    use_flash_attn=False)
            xp = out.x_pred[0][sample["vinput_mask"][0]].unsqueeze(0)
            xp = xp[:, : sample["tgt_image_len"]].float()
            v_preds.append((xp - z.float()) / sigma)

        v = v_preds[0]
        if len(v_preds) > 1:
            v = v_preds[1] + guidance_scale * (v_preds[0] - v_preds[1])
        z = sched.step(-v, step_t.to(torch.float32), z.float(),
                       return_dict=False)[0].to(dtype)

    img = unpatchify(((z.float() + 1) / 2).cpu(), h_patches, w_patches)[0]
    # Three channels carry the same matte; averaging is the inverse of the
    # replication the dataset does, and it also averages away channel noise.
    return img.mean(0).clamp(0, 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--d646_root", default=DEFAULT_D646_ROOT)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--overfit_samples", type=int, default=32)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=len(DEFAULT_TIMESTEPS))
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle_conditions", action="store_true")
    ap.add_argument("--save_npy", action="store_true")
    args = ap.parse_args()

    from transformers import AutoProcessor

    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    device, dtype = torch.device("cuda"), torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)

    ds = D646MattingDataset(root=args.d646_root, resolution=args.size,
                            overfit_samples=args.overfit_samples,
                            prompt=args.prompt)
    n = min(args.num_samples, len(ds))

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="cuda")
    model_config = model.config
    model, meta = load_adapter(model, args.adapter_path)
    model.eval()

    records, mses = [], []
    for i in range(n):
        item = ds[i]
        # Deterministic derangement, same rule the in-training check uses.
        cond_item = ds[(i + 1) % n] if args.shuffle_conditions else item
        alpha = sample_one(
            model, cond_item["condition"], args.prompt, args.size, tokenizer,
            processor, model_config, device, dtype, args.steps,
            args.guidance_scale, args.shift, args.seed + i)

        gt = ((item["alpha_rgb"][0].float() + 1) / 2).clamp(0, 1).numpy()
        mse = float(np.mean((alpha - gt) ** 2))
        mses.append(mse)

        sid = item["sample_id"]
        Image.fromarray((alpha * 255).round().astype(np.uint8)).save(
            os.path.join(args.output_dir, f"{sid}.png"))
        if args.save_npy:
            np.save(os.path.join(args.output_dir, f"{sid}.npy"),
                    alpha.astype(np.float32))
        records.append({"sample_id": sid, "mse": mse,
                        "condition_from": cond_item["sample_id"]})
        print(f"  {sid:>16}  mse {mse:.5f}"
              f"{'  (shuffled cond)' if args.shuffle_conditions else ''}",
              flush=True)

    mean_mse = float(np.mean(mses))
    summary = {
        "generated_mse": mean_mse,
        "num_samples": n,
        "shuffled": args.shuffle_conditions,
        "adapter": os.path.abspath(args.adapter_path),
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "per_sample": records,
        "baselines": {"all_black": 0.266, "constant_mean": 0.193},
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\ngenerated_mse {mean_mse:.5f} over {n} samples")
    print(f"trivial baselines -- all-black 0.266, constant-mean 0.193")
    verdict = "beats both" if mean_mse < 0.193 else "AT OR WORSE THAN TRIVIAL"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
