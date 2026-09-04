"""Stage 2 gate: the trainer's model surgery must be a no-op at step 0.

`setup_model` does three things to a pretrained checkpoint before a single
gradient is taken: wraps 252 projections in LoRA, unfreezes `x_embedder` and
`final_layer2`, and casts every trainable parameter to fp32. All three are
supposed to leave the *function* untouched -- LoRA's `B` is zero-initialised, the
unfreeze changes no values, and the fp32 cast is a widening that autocast
narrows again per-op.

If any of that is wrong the run still trains, and the damage looks like slow
convergence rather than an error, so it is worth one forward pass to check. The
same discipline as `conditioning_proj_init=zero` in the PixelDiT pilot: start
bit-identical to the checkpoint, and let the adapter grow from nothing.

Also reports peak GPU memory for one training-shaped forward+backward, which is
what decides whether the batch size can go above 1.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from matting.data import (  # noqa: E402
    DEFAULT_D646_ROOT, DEFAULT_PROMPT, build_dataset)
from matting.sample_builder import build_matting_sample, patchify  # noqa: E402
from matting.train_hidream_matting import (  # noqa: E402
    DEFAULT_MODEL, FULL_TRAIN_MODULES, NOISE_SCALE, T_EPS, lora_target_modules,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--d646_root", default=DEFAULT_D646_ROOT)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--use_bbox", action="store_true", default=False)
    ap.add_argument("--bbox_resolution", type=int, default=512)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    device, dtype = torch.device("cuda"), torch.bfloat16

    ds = build_dataset(names=["d646"], resolution=args.size, overfit_samples=1,
                       use_bbox=args.use_bbox, bbox_jitter=0.0)
    item = ds[0]
    layout = None
    if args.use_bbox:
        from matting.bbox import render_layout_image
        layout = render_layout_image(item["bbox"], args.bbox_resolution,
                                     args.bbox_resolution)

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="cuda")
    model_config = model.config
    model.eval()

    sample = build_matting_sample(
        cond_image=item["condition"], prompt=DEFAULT_PROMPT,
        height=args.size, width=args.size, tokenizer=tokenizer,
        processor=processor, model_config=model_config, device=device,
        dtype=dtype, layout_image=layout, layout_size=args.bbox_resolution)
    print(f"[identity] num_refs={sample['num_refs']} "
          f"total_ref_len={sample['total_ref_len']}")

    x0 = patchify(item["alpha_rgb"]).unsqueeze(0).to(device).float()
    g = torch.Generator().manual_seed(args.seed)
    eps = torch.randn(x0.shape, generator=g).to(device)
    s = args.sigma
    z = (1.0 - s) * x0 + s * NOISE_SCALE * eps
    vinputs = torch.cat([z.to(dtype), sample["ref_patches"]], dim=1)
    t = torch.tensor([1.0 - s], device=device)

    def forward():
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            out = model(
                input_ids=sample["input_ids"], position_ids=sample["position_ids"],
                vinputs=vinputs, timestep=t, token_types=sample["token_types"],
                pixel_values=sample["pixel_values"],
                image_grid_thw=sample["image_grid_thw"], use_flash_attn=False)
        xp = out.x_pred[0][sample["vinput_mask"][0]].unsqueeze(0)
        return xp[:, : sample["tgt_image_len"]].float()

    with torch.no_grad():
        before = forward()
    print(f"[identity] baseline x_pred: mean {before.mean():.5f} "
          f"std {before.std():.5f}")

    targets = lora_target_modules(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none", target_modules=targets))
    for name, p in model.named_parameters():
        if any(f".{m}." in name for m in FULL_TRAIN_MODULES):
            p.requires_grad_(True)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    with torch.no_grad():
        after = forward()

    delta = (after - before).abs().max().item()
    rel = delta / before.abs().max().item()
    print(f"[identity] after LoRA + fp32 cast: max|delta| {delta:.3e} "
          f"(relative {rel:.3e})")
    ok = delta < args.tol
    print(f"\nGATE (step-0 output unchanged): {'PASS' if ok else 'FAIL'}")

    # Memory for one training-shaped forward+backward.
    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.cuda.reset_peak_memory_stats()
    x0_pred = forward()
    pred = (z - x0_pred) / max(s, T_EPS)
    loss = F.mse_loss(pred, (NOISE_SCALE * eps - x0))
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n[memory] peak {peak:.1f} GB of {total:.1f} GB "
          f"for batch=1 at {args.size}px ({sample['tgt_image_len']} target tokens)")
    print(f"[memory] headroom suggests batch up to ~{int(total * 0.85 / peak)}x "
          f"before activation growth is accounted for")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
