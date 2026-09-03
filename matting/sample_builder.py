"""Build the image-conditioned token layout that HiDream expects, for training.

`models/pipeline.py::generate_image` constructs this layout inline inside its
inference loop, tangled up with CFG batching, multi-reference size heuristics and
layout-bbox handling. Training needs the same layout without any of that, and
needs it per-sample rather than once per generation, so the `ref_image_paths`
branch (pipeline.py:177-305) is lifted here and specialised to K=1 reference.

This is the piece ai-toolkit's HiDream-O1 trainer does not have -- its
`build_conditioning_sample` emits the text-to-image layout (token types 1 and 3
only), with no reference-patch stream. Matting is entirely about that stream, so
it has to be written rather than borrowed.

The sequence the model sees, in order:

    [ text tokens ................................ token_type 0, causal ]
    [ VLM condition tokens (ref image @ 384px) ... token_type 0, causal ]
    [ <boi> <tms> ................................ token_type 3 (timestep) ]
    [ target image tokens (noisy alpha) .......... token_type 1, full attn ]
    [ reference image tokens (clean RGB) ......... token_type 2, full attn ]

`vinput_mask` selects the last two blocks, which is what `vinputs` supplies and
what `x_pred` is read back from. The target occupies the *first* `tgt_image_len`
of those positions -- that slice is the only part the loss is computed on.
"""

import einops
import numpy as np
import torch
from PIL import Image

from models.pipeline import (
    CONDITION_IMAGE_SIZE,
    PATCH_SIZE,
    TENSOR_TRANSFORM,
    TIMESTEP_TOKEN_NUM,
)
from models.utils import (
    calculate_dimensions,
    get_rope_index_fix_point,
    resize_pilimage,
)


def patchify(img_t):
    """(C, H, W) in [-1, 1] -> (num_patches, C*p*p)."""
    return einops.rearrange(
        img_t, "C (H p1) (W p2) -> (H W) (C p1 p2)", p1=PATCH_SIZE, p2=PATCH_SIZE
    )


def unpatchify(x, h_patches, w_patches):
    """(B, num_patches, C*p*p) -> (B, C, H, W)."""
    return einops.rearrange(
        x, "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
        H=h_patches, W=w_patches, p1=PATCH_SIZE, p2=PATCH_SIZE,
    )


def tensor_to_pil(img_t):
    """(C, H, W) in [-1, 1] -> PIL. Matches the uint8 round-trip inference does."""
    arr = ((img_t.float().cpu() + 1) / 2).clamp(0, 1).numpy().transpose(1, 2, 0)
    return Image.fromarray(np.round(arr * 255).astype(np.uint8))


def build_matting_sample(
    cond_image,
    prompt,
    height,
    width,
    tokenizer,
    processor,
    model_config,
    device=None,
    dtype=torch.bfloat16,
):
    """One image-conditioned training sample.

    Args:
        cond_image: the conditioning RGB, either a PIL image or a (C, H, W)
            tensor in [-1, 1] already at `height` x `width`.
        prompt: instruction text. Pass " " for the unconditional branch.
        height, width: target size, both multiples of `PATCH_SIZE`.

    Returns a dict carrying everything the model forward needs except `vinputs`
    (which the caller assembles as `cat([noisy_target, ref_patches])`, since the
    noise changes every step) and `timestep`.
    """
    if height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(
            f"height and width must be multiples of {PATCH_SIZE}, got {height}x{width}"
        )

    image_token_id = model_config.image_token_id
    video_token_id = model_config.video_token_id
    vision_start_token_id = model_config.vision_start_token_id
    spatial_merge_size = model_config.vision_config.spatial_merge_size

    cond_pil = (
        tensor_to_pil(cond_image) if torch.is_tensor(cond_image)
        else cond_image.convert("RGB")
    )
    if cond_pil.size != (width, height):
        raise ValueError(
            f"conditioning image is {cond_pil.size}, expected {(width, height)} -- "
            f"resize before calling, so that compositing and resizing stay in the "
            f"dataset where they can be verified"
        )

    # Both streams derive from `resize_pilimage`'s output, because that is what
    # inference does (pipeline.py:216-224) and train/test preprocessing has to
    # agree. For a square image already at `max_size` it is a verified no-op
    # (the BICUBIC pass at utils.py:232 is exact at unchanged size), so this
    # costs nothing today; it is here so that a non-square or non-1024 target
    # stays aligned with inference instead of silently diverging.
    max_size = max(height, width)  # K == 1
    pil_r = resize_pilimage(cond_pil, max_size, PATCH_SIZE)

    # Reference patch stream: full-resolution RGB, patchified exactly as the
    # noisy target is, and embedded by the same `x_embedder`.
    ref_patches = patchify(TENSOR_TRANSFORM(pil_r)).unsqueeze(0)
    total_ref_len = ref_patches.shape[1]

    # VLM condition stream: the same image at 384px through Qwen3-VL's encoder.
    cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, pil_r.width / pil_r.height)
    cond_pil_vlm = pil_r.resize((cond_w, cond_h), resample=Image.LANCZOS)

    boi_token = getattr(tokenizer, "boi_token", "<|boi_token|>")
    tms_token = getattr(tokenizer, "tms_token", "<|tms_token|>")

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    template_caption = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    proc = processor(
        text=[template_caption], images=[cond_pil_vlm],
        padding="longest", return_tensors="pt",
    )
    input_ids_2 = tokenizer.encode(
        boi_token + tms_token * TIMESTEP_TOKEN_NUM,
        return_tensors="pt", add_special_tokens=False,
    )
    input_ids = torch.cat([proc.input_ids, input_ids_2], dim=-1)

    tgt_image_len = (height // PATCH_SIZE) * (width // PATCH_SIZE)
    image_grid_thw_tgt = torch.tensor(
        [1, height // PATCH_SIZE, width // PATCH_SIZE], dtype=torch.int64
    ).unsqueeze(0)
    image_grid_thw_ref = torch.tensor(
        [1, pil_r.height // PATCH_SIZE, pil_r.width // PATCH_SIZE], dtype=torch.int64
    ).unsqueeze(0)

    igthw_cond = proc.image_grid_thw.clone()
    igthw_cond[0, 1] //= spatial_merge_size
    igthw_cond[0, 2] //= spatial_merge_size
    igthw_all = torch.cat([igthw_cond, image_grid_thw_tgt, image_grid_thw_ref], dim=0)

    # Vision placeholder tokens, target block then reference block. Only the
    # first position of each block carries `vision_start`; the rope helper is
    # told to skip it for the streams that are already positioned.
    vt_tgt = torch.full((1, tgt_image_len), image_token_id, dtype=input_ids.dtype)
    vt_tgt[0, 0] = vision_start_token_id
    vt_ref = torch.full((1, total_ref_len), image_token_id, dtype=input_ids.dtype)
    vt_ref[0, 0] = vision_start_token_id
    vision_tokens = torch.cat([vt_tgt, vt_ref], dim=1)
    input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

    position_ids, _ = get_rope_index_fix_point(
        1, image_token_id, video_token_id, vision_start_token_id,
        input_ids=input_ids_pad, image_grid_thw=igthw_all,
        video_grid_thw=None, attention_mask=None,
        skip_vision_start_token=[0, 1, 1],  # [cond] + [target] + [ref]
    )

    txt_seq_len = input_ids.shape[-1]
    all_seq_len = position_ids.shape[-1]

    token_types_raw = torch.zeros((1, all_seq_len), dtype=input_ids.dtype)
    bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
    end = bgn + tgt_image_len + TIMESTEP_TOKEN_NUM
    token_types_raw[0, bgn:end] = 1                       # timestep + target
    token_types_raw[0, end:end + total_ref_len] = 2       # reference
    token_types_raw[0, txt_seq_len - TIMESTEP_TOKEN_NUM:txt_seq_len] = 3  # tms

    vinput_mask = torch.logical_or(token_types_raw == 1, token_types_raw == 2)
    token_types_bin = (token_types_raw > 0).to(token_types_raw.dtype)

    sample = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_types": token_types_bin,
        "vinput_mask": vinput_mask,
        "pixel_values": proc.pixel_values,
        "image_grid_thw": proc.image_grid_thw,
        "ref_patches": ref_patches,
        "tgt_image_len": tgt_image_len,
        "total_ref_len": total_ref_len,
        "txt_seq_len": txt_seq_len,
    }
    if device is not None:
        sample = {
            k: (v.to(device, dtype) if torch.is_tensor(v) and v.is_floating_point()
                else v.to(device) if torch.is_tensor(v) else v)
            for k, v in sample.items()
        }
    return sample


def target_slice(sample):
    """Positions within the `vinput_mask` selection that hold the target image.

    `x_pred[b][vinput_mask[b]]` yields target tokens followed by reference
    tokens; only the former is predicted and only it enters the loss. Mirrors
    `pipeline.py:357`, which takes `[:tgt_image_len]` for the same reason.
    """
    return slice(0, sample["tgt_image_len"])


def collate_samples(samples):
    """Batch samples that share one prompt and one resolution.

    The token layout is fully determined by (prompt, height, width), so
    `input_ids`, `position_ids`, `token_types` and `vinput_mask` are identical
    across samples and are expanded rather than stacked. Only the two
    image-derived tensors actually differ: `pixel_values` (the 384px VLM stream)
    and `ref_patches` (the full-resolution reference patches).

    Two consequences worth stating, because both are easy to violate:

    * Every sample in a batch must use the same prompt. `" "` tokenizes to a
      different length than the instruction, so CFG dropout has to be decided
      per *batch*, not per sample, or the layouts stop matching.
    * `pixel_values` concatenates along dim 0 rather than stacking. Qwen3-VL
      packs images as one flat run of patches and uses `image_grid_thw` to say
      where each begins, so a batch of B images is `[B * patches, dim]` with B
      rows of grid, not `[B, patches, dim]`.
    """
    if not samples:
        raise ValueError("collate_samples got an empty list")
    ref = samples[0]
    batch = len(samples)

    for i, s in enumerate(samples[1:], start=1):
        if (s["txt_seq_len"] != ref["txt_seq_len"]
                or s["tgt_image_len"] != ref["tgt_image_len"]
                or s["total_ref_len"] != ref["total_ref_len"]):
            raise ValueError(
                f"sample {i} has a different token layout from sample 0 "
                f"(txt {s['txt_seq_len']} vs {ref['txt_seq_len']}, tgt "
                f"{s['tgt_image_len']} vs {ref['tgt_image_len']}). Batching "
                f"requires one prompt and one resolution across the batch.")

    return {
        "input_ids": ref["input_ids"].expand(batch, -1).contiguous(),
        # position_ids is [3, batch, seq]: the batch axis is dim 1, not 0.
        "position_ids": ref["position_ids"].expand(-1, batch, -1).contiguous(),
        "token_types": ref["token_types"].expand(batch, -1).contiguous(),
        # Left as [1, seq]; every row is identical, and indexing x_pred with the
        # single row broadcasts correctly across the batch.
        "vinput_mask": ref["vinput_mask"],
        "pixel_values": torch.cat([s["pixel_values"] for s in samples], dim=0),
        "image_grid_thw": torch.cat([s["image_grid_thw"] for s in samples], dim=0),
        "ref_patches": torch.cat([s["ref_patches"] for s in samples], dim=0),
        "tgt_image_len": ref["tgt_image_len"],
        "total_ref_len": ref["total_ref_len"],
        "txt_seq_len": ref["txt_seq_len"],
        "batch_size": batch,
    }
