"""LoRA fine-tune HiDream-O1-Image for alpha matting on Distinctions-646.

The conventions here are not guesses -- each was pinned down before this file
was written, and the probes that pinned them are in this directory:

  * ``x_pred`` is an x0-prediction and the model is handed ``t = 1 - sigma``
    (``probe_flow_convention.py``, gate PASS; paper 2605.11061 3.2).
  * the noise is scaled by 8.0 in *training*, not only at sampling time
    (``probe_sampler.py``; ostris ``hidream_o1_model.py:58-67``).
  * 1024x1024 is a real operating point even though ``find_closest_resolution``
    snaps to 2048 (``probe_sampler.py``; paper 4.1, Stage II).
  * the token layout and the reference stream are live
    (``probe_sample_wiring.py``, ``tests/test_sample_builder.py``).

**The loss is velocity-space, not x0-space.** The head emits x0, so it is
converted before the MSE, exactly as ostris's reference implementation does
(``hidream_o1_model.py:321-345`` and ``:516-520``)::

    pred   = (z - x0_pred) / max(sigma, 1e-3)
    target = 8.0 * eps - x0

That is algebraically ``MSE(x0_pred, x0) / sigma**2`` -- x0 error upweighted at
low sigma. Plain x0-MSE is a *different* objective, not a reparameterisation, so
this is deliberate. The 1/sigma**2 factor also explains the timestep sampler
default: at sigma -> 0 the weight approaches 1e6, and ``sigmoid`` (logit-normal)
keeps samples away from that edge. ``--timestep_type uniform`` follows the
paper's SFT recipe (4.2) instead and is the intended A/B, not a safe default.

Batch size is 1 by construction. ``input_ids`` is not broadcast for us
(``qwen3_vl_transformers.py:1428`` embeds it directly), and every probe and
parity test runs batch-1, so the effective batch comes from accumulation.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from matting.data import (  # noqa: E402
    DEFAULT_D646_ROOT, DEFAULT_PROMPT, build_dataset)
from matting.sample_builder import (  # noqa: E402
    build_matting_sample, collate_samples, patchify)

DEFAULT_MODEL = (
    "/projects/ml4science/HF_CACHE/transformers/"
    "models--HiDream-ai--HiDream-O1-Image-Dev/snapshots/"
    "c0bada0e15c54a9f96a6d1ecc35575b32bc21544"
)

NOISE_SCALE = 8.0   # src/hidream_o1/pipeline.py:15, and models/pipeline.py:15
T_EPS = 0.001       # ostris clamps sigma here before dividing

LORA_PROJECTIONS = {
    "q_proj", "k_proj", "v_proj", "o_proj",      # attention
    "gate_proj", "up_proj", "down_proj",         # SwiGLU MLP
}
# Trained in full rather than through LoRA: they are the only pixel-space input
# and output in the network, they are small (~20M together), and an alpha matte
# is a different output distribution from RGB. Same reasoning as PixelDiT's
# refine head, which also trained in full for want of pretrained weights.
FULL_TRAIN_MODULES = ("x_embedder", "final_layer2")


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #

def sample_sigma(n, kind, shift, device, generator=None, sigma_min=T_EPS):
    """Draw sigma in [sigma_min, 1]. sigma=1 is pure noise, 0 is the clean matte.

    `sigma_min` matters more than it looks. The loss divides by sigma, so its
    weight goes as 1/sigma**2: at the default floor of 1e-3 a draw can carry a
    weight of 1e6, and uniform sampling reaches there roughly 0.1% of the time.
    Measured on this trainer, one such step logged |g| 2554 against a clip
    threshold of 1.0 -- the update survives, but scaled down 2500x, so the step
    is effectively wasted. Raising the floor to 0.05 caps the weight at 400 and
    removes the tail, at the cost of never training the last sliver of the
    schedule. 1e-3 is what the proven uniform run used; 0.05 is untested here.
    """
    if kind == "sigmoid":
        # sigmoid(randn) is the logit-normal distribution -- ai-toolkit's default
        # (`timestep_type`) and the paper's *pre-training* sampler. Concentrated
        # near 0.5, which keeps the 1/sigma**2 loss weight bounded in practice.
        u = torch.sigmoid(torch.randn(n, generator=generator))
    elif kind == "uniform":
        # Paper 4.2: SFT replaces logit-normal with uniform for "balanced
        # timestep coverage" and more weight on late-stage denoising.
        u = torch.rand(n, generator=generator)
    elif kind == "shift":
        u = torch.rand(n, generator=generator)
        u = shift * u / (1.0 + (shift - 1.0) * u)
    else:
        raise ValueError(f"unknown timestep_type {kind!r}")
    return u.clamp(sigma_min, 1.0).to(device)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def lora_target_modules(model):
    """Projection layers inside the text decoder stack, by full module name.

    Scoped with an explicit name list rather than bare suffixes: the Qwen3-VL
    vision tower carries similarly named Linears, and ostris's reference scopes
    LoRA the same way (`get_transformer_block_names` -> language_model.layers).
    """
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and ".language_model.layers." in name:
            if name.rsplit(".", 1)[-1] in LORA_PROJECTIONS:
                names.append(name)
    if not names:
        raise RuntimeError("no LoRA targets matched -- module layout changed?")
    return names


def setup_model(model_path, rank, alpha, dtype, gradient_checkpointing):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map="cuda"
    )
    model_config = model.config  # capture before PEFT wraps attribute access

    if gradient_checkpointing:
        # Must be explicit: the reentrant checkpointer cannot take the kwargs
        # this decoder passes (attention_mask, position_embeddings, ...) and
        # raises "Unexpected keyword arguments" on the first backward.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    targets = lora_target_modules(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
            target_modules=targets,
        ),
    )

    for name, param in model.named_parameters():
        if any(f".{m}." in name for m in FULL_TRAIN_MODULES):
            param.requires_grad_(True)

    # fp32 master weights for everything that trains; the frozen 8B stays bf16.
    # autocast casts back down per-op, so the forward is still bf16 maths.
    n_trainable = 0
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
            n_trainable += param.numel()

    print(f"[setup] LoRA on {len(targets)} projections, r={rank} alpha={alpha}")
    print(f"[setup] fully trained: {', '.join(FULL_TRAIN_MODULES)}")
    print(f"[setup] trainable params: {n_trainable/1e6:.2f}M")
    return model, processor, tokenizer, model_config


def trainable_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if ("lora_" in k) or any(f".{m}." in k for m in FULL_TRAIN_MODULES)}


# --------------------------------------------------------------------------- #
# sample construction
# --------------------------------------------------------------------------- #

def make_sample(item, prompt, args, tokenizer, processor, model_config,
                device, dtype, bbox=None):
    """Item -> model sample, rendering the bbox layout image when enabled.

    Used by training, validation and previews alike so all three build the
    sequence identically -- a mismatch there is the kind of bug that shows up as
    an unexplained metric rather than an error.

    `bbox` overrides the item's own box; the box-gap probe passes a different
    sample's box through here to test whether the model reads it at all.
    """
    layout = None
    if args.use_bbox:
        from matting.bbox import render_layout_image
        box = item["bbox"] if bbox is None else bbox
        layout = render_layout_image(box, args.bbox_resolution, args.bbox_resolution)
    return build_matting_sample(
        cond_image=item["condition"], prompt=prompt,
        height=args.resolution, width=args.resolution, tokenizer=tokenizer,
        processor=processor, model_config=model_config, device=device,
        dtype=dtype, layout_image=layout, layout_size=args.bbox_resolution)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def validate(model, dataset, indices, args, tokenizer, processor, model_config,
             device, dtype, sigmas=(0.999, 0.6, 0.3)):
    """x0 error with the right conditioning image, and with the wrong one.

    The gap between the two is the diagnostic that decided the PixelDiT pilot:
    a model can drive its training loss down for thousands of steps while
    ignoring the conditioning image entirely, and the loss curve cannot see it
    happening (MATTING.md, "Flow regime" -- the gap decayed 44% -> 13% while
    training loss improved 2.3x). Treat a gap under ~10% as a dead run.

    Fixed sigmas and a fixed noise seed, so the number reflects the model rather
    than the draw, and stays comparable across steps and across runs.

    Reported per sigma, not pooled. A pooled figure over a mid-range sigma set
    measures only where a logit-normal sampler already concentrates, which is
    how an earlier A/B here read backwards: the arm that looked 7 points worse
    on a pooled {0.3, 0.6} gap was 2.45x *better* at sigma 0.999. Sampling
    launches at 0.999, so that row is the one that predicts whether generation
    works -- it belongs in the training log rather than in a separate probe.
    """
    was_training = model.training
    model.eval()
    per_sigma = {s: [0.0, 0.0] for s in sigmas}   # sigma -> [correct, shuffled]
    n = 0

    for i, idx in enumerate(indices):
        item = dataset[idx]
        # Deterministic derangement: each sample takes the next one's condition,
        # so nothing is ever paired with its own.
        other = dataset[indices[(i + 1) % len(indices)]]
        x0 = patchify(item["alpha_rgb"]).unsqueeze(0).to(device).float()

        for sigma in sigmas:
            g = torch.Generator().manual_seed(1234 + int(sigma * 1000))
            eps = torch.randn(x0.shape, generator=g).to(device)
            z = (1.0 - sigma) * x0 + sigma * NOISE_SCALE * eps
            t = torch.tensor([1.0 - sigma], device=device)

            for cond, is_correct in ((item["condition"], True),
                                     (other["condition"], False)):
                sample = make_sample(
                    {**item, "condition": cond}, args.prompt, args, tokenizer,
                    processor, model_config, device, dtype)
                vinputs = torch.cat([z.to(dtype), sample["ref_patches"]], dim=1)
                with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
                    out = model(
                        input_ids=sample["input_ids"],
                        position_ids=sample["position_ids"],
                        vinputs=vinputs, timestep=t,
                        token_types=sample["token_types"],
                        pixel_values=sample["pixel_values"],
                        image_grid_thw=sample["image_grid_thw"],
                        use_flash_attn=False)
                xp = out.x_pred[0][sample["vinput_mask"][0]].unsqueeze(0)
                mse = F.mse_loss(xp[:, : sample["tgt_image_len"]].float(), x0).item()
                per_sigma[sigma][0 if is_correct else 1] += mse
        n += 1

    if was_training:
        model.train()

    out = {}
    for sigma, (tot_c, tot_s) in per_sigma.items():
        mse_c, mse_s = tot_c / n, tot_s / n
        out[sigma] = {
            "x0_mse": mse_c,
            "x0_mse_shuffled": mse_s,
            "gap": (mse_s - mse_c) / mse_s if mse_s > 0 else 0.0,
        }
    return out


# --------------------------------------------------------------------------- #
# preview
# --------------------------------------------------------------------------- #

def _overlay_box(rgb, bbox, color=(1.0, 0.0, 0.0), width=6):
    """Outline the box on an RGB float array, in DEFAULT_COLORS[0] red.

    Same colour the layout image is rendered in, so the panel's box and the
    one the model was given read as the same object.
    """
    import numpy as np
    out = rgb.copy()
    h, w = out.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = int(x1 * w), int(x2 * w)
    y1, y2 = int(y1 * h), int(y2 * h)
    c = np.asarray(color, dtype=out.dtype)
    for t in range(width):
        for a, b in ((y1 + t, None), (y2 - t, None)):
            if 0 <= a < h:
                out[a, max(0, x1):min(w, x2)] = c
        for a, b in ((x1 + t, None), (x2 - t, None)):
            if 0 <= a < w:
                out[max(0, y1):min(h, y2), a] = c
    return out


@torch.no_grad()
def log_preview(model, dataset, indices, args, tokenizer, processor,
                model_config, device, dtype, run, step, tag="fixed"):
    """Sample real mattes with the real sampler and log them to W&B.

    This is the only thing in the training loop that runs a full trajectory.
    `validate()` takes single forward passes at fixed sigma, which is cheap but
    measures denoising, not generation -- and the two disagreed badly here: the
    arm with the better single-step conditioning gap (97% vs 71.6%) was the one
    producing flat grey mattes, generated_mse 0.205 against 0.005. Judge a run
    on this panel and on `preview/generated_mse`, not on the validation gap.

    The sampling seed is FIXED across steps (`--preview_seed`), so consecutive
    previews differ only by the model. It used to vary with the step, and the
    resulting curve was uninterpretable: generated_mse swung between 0.017 and
    0.233 on adjacent previews of the *same* four images, because each one
    started from different noise. A moving seed measures model + draw; only a
    fixed one measures the model.

    Costs `num_examples * sampling_steps` forward passes, so roughly 30s for the
    default 2 x 28 -- keep `preview_every` well above `val_every`.
    """
    # Deferred: sample_matting imports names from this module, so a top-level
    # import here would be circular.
    from matting.sample_matting import sample_one

    import numpy as np
    import wandb

    if not indices:
        return None
    was_training = model.training
    model.eval()
    rows, mses, ids, blacks, means = [], [], [], [], []
    for i in indices:
        item = dataset[i]
        layout = None
        if args.use_bbox:
            from matting.bbox import render_layout_image
            layout = render_layout_image(item["bbox"], args.bbox_resolution,
                                         args.bbox_resolution)
        alpha = sample_one(
            model, item["condition"], args.prompt, args.resolution, tokenizer,
            processor, model_config, device, dtype, args.wandb_sampling_steps,
            args.wandb_guidance, args.shift, args.preview_seed + i,
            layout_image=layout, layout_size=args.bbox_resolution)
        ids.append(item["sample_id"])
        gt = ((item["alpha_rgb"][0].float() + 1) / 2).clamp(0, 1).numpy()
        rgb = ((item["condition"].float() + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)

        mses.append(float(np.mean((alpha - gt) ** 2)))
        # Baselines are per-sample, not constants. all-black scores mean(gt**2),
        # which is just foreground coverage: it ranges from 0.07 on a small
        # subject to 0.60 on a large one across D-646. Comparing every panel to
        # MATTING.md's 0.266 -- computed on one particular subset -- makes a
        # rotating panel's flag meaningless, and made a hard fixed pair look
        # like a failing model.
        blacks.append(float(np.mean(gt ** 2)))
        means.append(float(np.mean((gt - gt.mean()) ** 2)))
        # RGB | (RGB+box) | generated alpha | ground truth. The first three
        # columns and their order match PixelDiT's preview grid so the two can
        # be read side by side; the box column is inserted only when bbox
        # conditioning is on.
        #
        # The clean RGB is kept alongside the annotated one on purpose: the
        # overlay covers pixels, and those are exactly the pixels a subject
        # touching the box edge needs. Showing both means the box never hides
        # the thing you are trying to judge. The overlay is display-only -- the
        # conditioning image the model saw is untouched.
        cols = [rgb]
        if args.use_bbox:
            cols.append(_overlay_box(rgb, item["bbox"]))
        cols += [np.stack([alpha] * 3, -1), np.stack([gt] * 3, -1)]
        rows.append(np.concatenate(cols, axis=1))

    grid = (np.concatenate(rows, axis=0) * 255).round().astype("uint8")
    mean_mse = float(np.mean(mses))
    black, cmean = float(np.mean(blacks)), float(np.mean(means))
    # Fraction of the all-black baseline for THESE samples. Below 1.0 beats a
    # blank prediction; this is comparable across panels, the raw mse is not.
    rel = mean_mse / max(black, 1e-9)
    cols = ("RGB | RGB+box | generated | GT" if args.use_bbox
            else "RGB | generated | GT")
    caption = (f"step {step} | {cols} | {tag} | mse {mean_mse:.5f} "
               f"({rel:.0%} of all-black {black:.3f}, const-mean {cmean:.3f}) | "
               + ", ".join(ids))
    if run is not None:
        run.log({f"preview/{tag}_panel": wandb.Image(grid, caption=caption),
                 f"preview/{tag}_generated_mse": mean_mse,
                 f"preview/{tag}_vs_allblack": rel}, step=step)
    print(f"[{step:>6}] PREVIEW[{tag}] generated_mse {mean_mse:.5f}  "
          f"= {rel:.0%} of all-black ({black:.3f}) "
          f"({args.wandb_sampling_steps} steps, g={args.wandb_guidance})"
          f"{'' if rel < 1.0 else '   <-- WORSE THAN A BLANK PREDICTION'}",
          flush=True)
    if was_training:
        model.train()
    return mean_mse


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def train(args):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    run_dir = args.run_dir or os.path.join(
        args.run_root, f"hidream-matting-d646_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(os.path.join(run_dir, "adapters"), exist_ok=True)
    print(f"[run] {run_dir}")

    with open(os.path.join(run_dir, "config_resolved.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2, sort_keys=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dataset = build_dataset(
        names=args.datasets, resolution=args.resolution, split="train",
        overfit_samples=args.overfit_samples, prompt=args.prompt,
        use_bbox=args.use_bbox, bbox_jitter=args.bbox_jitter,
        weights=args.dataset_weights,
    )
    print(f"[data] {'+'.join(args.datasets)}: {len(dataset)} samples at "
          f"{args.resolution}px | batch {args.batch_size} x accum "
          f"{args.grad_accum} = effective {args.batch_size * args.grad_accum}"
          f"{' | bbox on, jitter ' + str(args.bbox_jitter) if args.use_bbox else ''}")
    with open(os.path.join(run_dir, "manifest.json"), "w") as fh:
        ids = dataset.sample_ids() if hasattr(dataset, "sample_ids") else None
        json.dump({"sample_ids": ids, "datasets": list(args.datasets),
                   "overfit_samples": args.overfit_samples}, fh, indent=2)

    model, processor, tokenizer, model_config = setup_model(
        args.model_path, args.lora_rank, args.lora_alpha, dtype,
        args.gradient_checkpointing,
    )
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    start_step = 0
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if unexpected:
            raise RuntimeError(f"checkpoint has unexpected keys: {unexpected[:5]}")
        start_step = int(ckpt.get("step", 0))
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"[resume] step {start_step}, optimizer state restored")
        else:
            # Checkpoints written before optimizer state was saved. Adam
            # re-warms its moments over the first few dozen steps; the weights
            # are what matter.
            print(f"[resume] step {start_step}, WEIGHTS ONLY -- no optimizer "
                  f"state in this checkpoint, Adam moments restart from zero")

    run = None
    if args.wandb:
        import wandb
        # id derived from the run directory so a resumed run continues the same
        # W&B run rather than starting a second one beside it.
        wandb_id = args.wandb_id or hashlib.md5(
            os.path.abspath(run_dir).encode()).hexdigest()[:16]
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or os.path.basename(run_dir),
            id=wandb_id, resume="allow", dir=run_dir, config=vars(args))
        run.summary["run_dir"] = run_dir

    # Fixed validation subset, STRIDED rather than consecutive.
    #
    # D-646 composites each foreground over 100 backgrounds and lays them out
    # consecutively, so range(4) is four views of ONE object -- four different
    # RGB composites sharing a single, byte-identical alpha matte. A preview
    # built that way shows one object and reads as if the model were emitting a
    # constant. MATTING.md stratifies its overfit subset by foreground identity
    # for the same reason; that path is inactive at overfit_samples=0, so stride
    # here instead.
    n_fixed = max(args.val_samples, args.preview_fixed_samples)
    stride = max(1, len(dataset) // max(1, n_fixed))
    # `i * stride + i`, not `i * stride`. A mixture interleaves sources by
    # index parity, so an even stride would land every validation sample on the
    # same dataset -- with d646+am2k and stride 4, indices 0/4/8/12 are all
    # d646 and AM-2k is never validated. The extra +i breaks that alignment
    # while still spanning the dataset.
    val_indices = [(i * stride + i) % len(dataset)
                   for i in range(min(n_fixed, len(dataset)))]
    if len(dataset) > 1:
        cats = {dataset[i]["category"] for i in val_indices}
        srcs = {dataset[i].get("dataset", "?") for i in val_indices}
        print(f"[val] fixed subset: {len(val_indices)} samples spanning "
              f"{len(cats)} distinct foregrounds from {sorted(srcs)} "
              f"(stride {stride})")

    log_path = os.path.join(run_dir, "train_log.jsonl")
    gen = torch.Generator().manual_seed(args.seed + start_step)
    order = list(range(len(dataset)))
    cursor = len(order)
    random.seed(args.seed + start_step)

    step, micro, t_start = start_step, 0, time.time()
    accum_loss = accum_x0 = accum_sigma = 0.0
    sigma_buckets = {"lo": [], "mid": [], "hi": []}
    while step < args.max_steps:
        items = []
        for _ in range(args.batch_size):
            if cursor >= len(order):
                random.shuffle(order)
                cursor = 0
            items.append(dataset[order[cursor]])
            cursor += 1

        # CFG dropout is decided per *batch*, not per sample: " " tokenizes to a
        # different length than the instruction, and collate_samples requires one
        # layout across the batch. At batch 1 the two are the same thing.
        prompt = " " if random.random() < args.cfg_dropout else args.prompt

        samples = [
            make_sample(it, prompt, args, tokenizer, processor, model_config,
                        device, dtype)
            for it in items
        ]
        sample = collate_samples(samples) if len(samples) > 1 else samples[0]

        x0 = torch.cat(
            [patchify(it["alpha_rgb"]).unsqueeze(0) for it in items]
        ).to(device).float()
        eps = torch.randn(x0.shape, generator=gen).to(device)
        sigma = sample_sigma(len(items), args.timestep_type, args.shift, device,
                             gen, sigma_min=args.sigma_min)
        s = sigma.view(-1, 1, 1)
        z = (1.0 - s) * x0 + s * NOISE_SCALE * eps

        vinputs = torch.cat([z.to(dtype), sample["ref_patches"]], dim=1)
        with torch.autocast("cuda", dtype=dtype, cache_enabled=False):
            out = model(
                input_ids=sample["input_ids"],
                position_ids=sample["position_ids"],
                vinputs=vinputs,
                timestep=(1.0 - sigma),
                token_types=sample["token_types"],
                pixel_values=sample["pixel_values"],
                image_grid_thw=sample["image_grid_thw"],
                use_flash_attn=False,
            )

        # vinput_mask rows are identical, so one row indexes the whole batch.
        x0_pred = out.x_pred[:, sample["vinput_mask"][0]]
        x0_pred = x0_pred[:, : sample["tgt_image_len"]].float()

        pred = (z - x0_pred) / s.clamp_min(T_EPS)
        target = (NOISE_SCALE * eps - x0).detach()
        loss = F.mse_loss(pred, target)

        (loss / args.grad_accum).backward()
        # Every reported metric is averaged over the same micro-steps the
        # gradient was. Reporting the last micro-step's x0_mse next to an
        # 8-step mean loss makes the two look inconsistent, because the
        # velocity loss carries a 1/sigma**2 factor and each micro-step drew
        # its own sigma.
        accum_loss += loss.item() / args.grad_accum
        accum_x0 += F.mse_loss(x0_pred, x0).item() / args.grad_accum
        accum_sigma += float(sigma.mean().item()) / args.grad_accum
        # Bucketed by sigma, because the pooled number is not a progress metric:
        # x0 error spans ~100x across the schedule, so a step's value is set
        # mostly by which sigma it drew. Per-bucket medians are comparable
        # across steps; the pooled mean is not.
        with torch.no_grad():
            per = (x0_pred - x0).pow(2).mean(dim=(1, 2))
        for sv, e in zip(sigma.tolist(), per.tolist()):
            sigma_buckets["hi" if sv >= 0.66 else
                          "mid" if sv >= 0.33 else "lo"].append(e)
        micro += 1

        if micro < args.grad_accum:
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step, micro = step + 1, 0

        if step % args.log_every == 0:
            # x0-space MSE is reported alongside the trained loss because the
            # velocity loss is scaled by 1/sigma**2, so it is not comparable
            # across steps that happened to draw different sigmas. x0_mse is.
            rec = {"step": step, "loss": accum_loss, "x0_mse": accum_x0,
                   "sigma": accum_sigma, "grad_norm": float(grad_norm),
                   "lr": optimizer.param_groups[0]["lr"],
                   "sec_per_step": (time.time() - t_start) / args.log_every}
            import statistics as _st
            for name, vals in sigma_buckets.items():
                if vals:
                    rec[f"x0_mse_sigma_{name}"] = _st.median(vals)
            sigma_buckets = {"lo": [], "mid": [], "hi": []}
            buckets = "  ".join(
                f"{n}{rec[f'x0_mse_sigma_{n}']:.4f}"
                for n in ("lo", "mid", "hi") if f"x0_mse_sigma_{n}" in rec)
            print(f"[{step:>6}] loss {accum_loss:.4f}  x0_mse {accum_x0:.5f}  "
                  f"[{buckets}]  |g| {grad_norm:.2f}  "
                  f"{rec['sec_per_step']:.2f}s/step", flush=True)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            if run is not None:
                run.log(rec, step=step)
            t_start = time.time()

        if step % args.save_every == 0 or step == args.max_steps:
            path = os.path.join(run_dir, "adapters", f"step_{step}.pth")
            torch.save({
                "state_dict": trainable_state_dict(model),
                # Adam moments, so a resumed run continues rather than
                # re-warming. Roughly 2x the trainable size, ~500MB here.
                "optimizer": optimizer.state_dict(),
                "step": step,
                "metadata": {
                    "lora_rank": args.lora_rank, "lora_alpha": args.lora_alpha,
                    "full_train_modules": list(FULL_TRAIN_MODULES),
                    "resolution": args.resolution, "prompt": args.prompt,
                    "noise_scale": NOISE_SCALE,
                    "timestep_type": args.timestep_type,
                    # Recorded so evaluation configures itself from the
                    # checkpoint. Scoring a bbox-trained model without a box
                    # silently measures the wrong thing rather than failing.
                    "use_bbox": args.use_bbox,
                    "bbox_resolution": args.bbox_resolution,
                    "datasets": list(args.datasets),
                },
            }, path)
            print(f"[save] {path}")

        if args.val_every and step % args.val_every == 0:
            res = validate(model, dataset, val_indices, args, tokenizer,
                           processor, model_config, device, dtype)
            vrec = {"step": step}
            for sigma, r in res.items():
                key = f"s{sigma:g}".replace(".", "")
                vrec[f"val_x0_mse_{key}"] = r["x0_mse"]
                vrec[f"val_gap_{key}"] = r["gap"]
                # x0 is [-1,1]; MATTING.md's baselines are [0,1] alpha, so the
                # comparable figure is x0_mse/4 against all-black 0.266.
                frac = (r["x0_mse"] / 4.0) / 0.266
                note = "  <-- launch point, near-useless" if (
                    sigma >= 0.99 and frac > 0.5) else ""
                print(f"[{step:>6}] VAL s={sigma:<6.3g} x0_mse {r['x0_mse']:.5f}  "
                      f"shuffled {r['x0_mse_shuffled']:.5f}  "
                      f"gap {r['gap']*100:5.1f}%  ({frac:.0%} of all-black)"
                      f"{note}", flush=True)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(vrec) + "\n")
            if run is not None:
                run.log(vrec, step=step)

        if args.preview_every and step % args.preview_every == 0:
            # Two panels, following PixelDiT (MATTING.md, "Training"): a *fixed*
            # set so `preview/fixed_generated_mse` stays comparable step to step,
            # and a *rotating* set so the images are not the same four every
            # time. The rotating set is drawn without replacement, seeded by the
            # step, so a re-run shows the same images at the same points.
            log_preview(model, dataset, val_indices[:args.preview_fixed_samples],
                        args, tokenizer, processor, model_config, device, dtype,
                        run, step, tag="fixed")
            if args.wandb_num_examples > 0:
                rot = random.Random(step).sample(
                    range(len(dataset)), min(args.wandb_num_examples, len(dataset)))
                log_preview(model, dataset, rot, args, tokenizer, processor,
                            model_config, device, dtype, run, step, tag="rotating")

        accum_loss = accum_x0 = accum_sigma = 0.0

    print(f"[done] {step} steps -> {run_dir}")
    if run is not None:
        run.finish()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None, help="YAML of any of the flags below")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--d646_root", default=DEFAULT_D646_ROOT)
    p.add_argument("--datasets", nargs="+", default=["d646"],
                   choices=["d646", "am2k"])
    p.add_argument("--dataset_weights", nargs="+", type=float, default=None,
                   help="mixture weights, e.g. 1 1 for 50/50; default equal")
    p.add_argument("--use_bbox", action="store_true", default=False,
                   help="append a rendered bbox layout image as a 2nd reference")
    p.add_argument("--bbox_jitter", type=float, default=0.1,
                   help="per-edge random expand/shrink; stops the model copying "
                        "the box as a mask")
    p.add_argument("--bbox_resolution", type=int, default=512,
                   help="the layout image is a rectangle on black, so it needs "
                        "far less resolution than the photo (256 vs 1024 tokens)")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--overfit_samples", type=int, default=32)

    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                   action="store_false")

    p.add_argument("--timestep_type", default="sigmoid",
                   choices=["sigmoid", "uniform", "shift"])
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--sigma_min", type=float, default=T_EPS,
                   help="floor on sampled sigma; 0.05 caps the 1/sigma^2 loss "
                        "weight at 400 instead of 1e6 (default matches the "
                        "validated uniform run)")
    p.add_argument("--cfg_dropout", type=float, default=0.1)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=1,
                   help="samples per forward; effective batch is this x grad_accum")
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--resume_from", default=None,
                   help="adapter .pth to continue from; step count resumes with it")

    p.add_argument("--preview_every", type=int, default=500,
                   help="sample real mattes and log them to W&B; 0 disables")
    p.add_argument("--wandb_num_examples", type=int, default=2,
                   help="rotating panel size; 0 disables the rotating panel")
    p.add_argument("--preview_fixed_samples", type=int, default=2,
                   help="fixed panel size -- this is the comparable metric")
    p.add_argument("--wandb_sampling_steps", type=int, default=28,
                   help="trajectory length for the preview (28 = the Dev default)")
    p.add_argument("--wandb_guidance", type=float, default=1.0)
    p.add_argument("--preview_seed", type=int, default=1234,
                   help="fixed across steps so the preview curve is comparable")
    p.add_argument("--val_every", type=int, default=100,
                   help="0 disables the conditioning-gap check")
    p.add_argument("--val_samples", type=int, default=4)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=250)
    p.add_argument("--run_root", default="/scratch/mridul/runs/matting/hidream")
    p.add_argument("--run_dir", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="hidream-matting")
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_id", default=None,
                   help="defaults to a hash of run_dir, so resumes reattach")
    return p


def parse_args(argv=None):
    """CLI over an optional YAML. Explicit flags always beat the file, so a
    config can be overridden per-launch without editing it."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config:
        import yaml
        with open(args.config) as fh:
            cfg = yaml.safe_load(fh) or {}
        unknown = set(cfg) - set(vars(args))
        if unknown:
            raise SystemExit(f"unknown keys in {args.config}: {sorted(unknown)}")
        on_cli = {a.lstrip("-").replace("-", "_")
                  for a in (argv if argv is not None else sys.argv[1:])
                  if a.startswith("--")}
        for key, value in cfg.items():
            if key not in on_cli:
                setattr(args, key, value)
    return args


if __name__ == "__main__":
    train(parse_args())
