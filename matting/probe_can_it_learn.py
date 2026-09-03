"""Deep dive: can the training loop drive a single sample to zero error?

The canonical sanity check for a training loop. If one sample at one fixed
sigma cannot be driven to near-zero x0 error, something is structurally wrong
-- gradients are not reaching the weights, the loss is computed against the
wrong slice of the prediction, or the target is not what we think it is. If it
CAN, the loop is sound and a plateau on the full dataset is an optimization or
capacity question, not a bug.

Also reports which parameter groups actually moved, because "LoRA is attached"
and "LoRA is receiving gradient" are different claims and only the second one
matters.
"""

import argparse, os, sys, time
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from matting.data import DEFAULT_D646_ROOT, DEFAULT_PROMPT, D646MattingDataset  # noqa
from matting.sample_builder import build_matting_sample, patchify  # noqa
from matting.train_hidream_matting import (  # noqa
    DEFAULT_MODEL, FULL_TRAIN_MODULES, NOISE_SCALE, T_EPS, setup_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--d646_root", default=DEFAULT_D646_ROOT)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--fixed_noise", action="store_true", default=True)
    args = ap.parse_args()

    device, dtype = torch.device("cuda"), torch.bfloat16
    ds = D646MattingDataset(root=args.d646_root, resolution=args.resolution,
                            overfit_samples=1)
    item = ds[0]
    model, processor, tokenizer, cfg = setup_model(
        args.model_path, 16, 16, dtype, True)
    model.train()

    sample = build_matting_sample(
        cond_image=item["condition"], prompt=DEFAULT_PROMPT,
        height=args.resolution, width=args.resolution, tokenizer=tokenizer,
        processor=processor, model_config=cfg, device=device, dtype=dtype)
    x0 = patchify(item["alpha_rgb"]).unsqueeze(0).to(device).float()

    params = [p for p in model.parameters() if p.requires_grad]
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if p.requires_grad}
    opt = torch.optim.AdamW(params, lr=args.lr)

    g = torch.Generator().manual_seed(0)
    fixed_eps = torch.randn(x0.shape, generator=g).to(device)
    s = args.sigma

    print(f"\nsingle sample '{item['sample_id']}', sigma fixed at {s}, "
          f"lr {args.lr}, {args.steps} steps")
    print(f"{'step':>5} {'x0_mse':>10} {'loss':>10} {'|g|':>8}")
    first = None
    for i in range(1, args.steps + 1):
        eps = fixed_eps if args.fixed_noise else torch.randn(
            x0.shape, generator=g).to(device)
        z = (1.0 - s) * x0 + s * NOISE_SCALE * eps
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            out = model(input_ids=sample["input_ids"],
                        position_ids=sample["position_ids"],
                        vinputs=torch.cat([z.to(dtype), sample["ref_patches"]], dim=1),
                        timestep=torch.tensor([1.0 - s], device=device),
                        token_types=sample["token_types"],
                        pixel_values=sample["pixel_values"],
                        image_grid_thw=sample["image_grid_thw"],
                        use_flash_attn=False)
        x0_pred = out.x_pred[:, sample["vinput_mask"][0]][:, : sample["tgt_image_len"]].float()
        loss = F.mse_loss((z - x0_pred) / max(s, T_EPS), NOISE_SCALE * eps - x0)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        mse = F.mse_loss(x0_pred, x0).item()
        if first is None:
            first = mse
        if i <= 3 or i % 10 == 0 or i == args.steps:
            print(f"{i:>5} {mse:>10.6f} {loss.item():>10.4f} {gn:>8.3f}")

    print(f"\nx0_mse {first:.6f} -> {mse:.6f}  ({first/max(mse,1e-9):.1f}x reduction)")
    ok = mse < 0.1 * first
    print(f"GATE (loop can fit one sample): {'PASS' if ok else 'FAIL'}")

    print("\nparameter groups that actually moved:")
    groups = {}
    for n, p in model.named_parameters():
        if n in before:
            d = (p.detach() - before[n]).abs().max().item()
            key = ("lora" if "lora_" in n else
                   next((m for m in FULL_TRAIN_MODULES if f".{m}." in n), "other"))
            groups.setdefault(key, []).append(d)
    for k, v in sorted(groups.items()):
        moved = sum(1 for x in v if x > 0)
        print(f"  {k:14s} {moved}/{len(v)} tensors changed, max delta {max(v):.3e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
