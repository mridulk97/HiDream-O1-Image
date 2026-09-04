"""Matting datasets, shaped for the HiDream training loop.

The compositing, background assignment and stratified subset selection all
already exist in PixelDiT's dataset classes (`pixdit_datasets.py`) and are
deliberately not reimplemented. D-646 composites at `__getitem__` following its
own `gen_train.py` -- background upscaled only if it does not already cover the
foreground, cropped to the foreground's shape, blended at *native* resolution
before the resize -- and assigns backgrounds by position so `<fg>_<k>` is
byte-identical on every epoch. Compositing before the resize is the part that
matters: alpha blending is not linear through downsampling, and the difference
lands on exactly the soft edges these datasets exist to exercise.

This module adapts those to dicts, adds the bounding box, and mixes the two
datasets. Both tensors arrive in [-1, 1] at `resolution`.
"""

import os
import sys

import numpy as np
from torch.utils.data import Dataset

PIXELDIT_T2I = "/home/mridul/matting/PixelDiT/t2i"
DEFAULT_D646_ROOT = "/scratch/mridul/data/matting/distinctions-646"
DEFAULT_AM2K_ROOT = "/scratch/mridul/data/matting/am-2k"

# The prompt the probes were run under. Changing it invalidates the wiring gate,
# so it lives in one place.
DEFAULT_PROMPT = "Transform to matting map while maintaining original composition"


def _ensure_pixeldit_importable(path=PIXELDIT_T2I):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"PixelDiT tree not found at {path}; it supplies the dataset classes. "
            f"Pass pixeldit_path= to point elsewhere."
        )
    if path not in sys.path:
        sys.path.insert(0, path)


class _MattingAdapter(Dataset):
    """Turns a PixelDiT matting dataset's 9-tuple into a dict, and adds the box.

    Both `AM2KMattingDataset` and `Distinctions646MattingDataset` return the
    same nine-element tuple, so one adapter covers both.

    Alpha is replicated to three channels because the backbone is an RGB pixel
    model: one 32x32x3 patch embedding, one 32x32x3 output head, and widening
    either would break the pretrained function. Inference averages the three
    channels back down.
    """

    def __init__(self, inner, resolution, prompt, use_bbox, bbox_jitter, name):
        self.inner = inner
        self.resolution = resolution
        self.prompt = prompt
        self.use_bbox = use_bbox
        self.bbox_jitter = bbox_jitter
        self.name = name

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        # (alpha_rgb, prompt, attn_mask, data_info, idx, "prompt", sample_id,
        #  category, condition) -- see pixdit_datasets.py.
        rec = self.inner[idx]
        item = {
            "alpha_rgb": rec[0],
            "condition": rec[8],
            "sample_id": rec[6],
            "category": rec[7],
            "index": idx,
            "dataset": self.name,
        }
        if self.use_bbox:
            # Computed here rather than cached with the composite: the jitter
            # must be resampled every epoch, or the box becomes a fixed
            # per-sample constant the model can memorise.
            from matting.bbox import bbox_from_alpha
            alpha = ((item["alpha_rgb"][0].float() + 1) / 2).clamp(0, 1).numpy()
            item["bbox"] = bbox_from_alpha(alpha, self.bbox_jitter)
        return item

    def sample_ids(self):
        return [r["sample_id"] for r in self.inner.dataset]


def _build(cls_name, root, resolution, split, overfit_samples, overfit_seed,
           prompt, cache_composites, background_dir, pixeldit_path, name,
           use_bbox, bbox_jitter):
    _ensure_pixeldit_importable(pixeldit_path)
    import diffusion.data.datasets.pixdit_datasets as pd

    extra = {
        "split": split,
        "overfit_samples": int(overfit_samples),
        "overfit_seed": int(overfit_seed),
        "default_prompt": prompt,
    }
    if cache_composites is not None:
        extra["cache_composites"] = bool(cache_composites)
    if background_dir is not None:
        extra["background_dir"] = background_dir
    inner = getattr(pd, cls_name)(data_dir=[root], resolution=resolution, extra=extra)
    return _MattingAdapter(inner, resolution, prompt, use_bbox, bbox_jitter, name)


def D646MattingDataset(root=DEFAULT_D646_ROOT, resolution=1024, split="train",
                       overfit_samples=0, overfit_seed=2025, cache_composites=None,
                       prompt=DEFAULT_PROMPT, background_dir=None,
                       pixeldit_path=PIXELDIT_T2I, use_bbox=False, bbox_jitter=0.1):
    """Distinctions-646: 596 foregrounds x 100 backgrounds, composited on the fly.

    This is where transparency lives -- glass, water, veils, fine hair, median
    6.9% soft pixels.
    """
    return _build("Distinctions646MattingDataset", root, resolution, split,
                  overfit_samples, overfit_seed, prompt, cache_composites,
                  background_dir, pixeldit_path, "d646", use_bbox, bbox_jitter)


def AM2KMattingDataset(root=DEFAULT_AM2K_ROOT, resolution=1024, split="train",
                       overfit_samples=0, overfit_seed=2025, cache_composites=None,
                       prompt=DEFAULT_PROMPT, background_dir=None,
                       pixeldit_path=PIXELDIT_T2I, use_bbox=False, bbox_jitter=0.1):
    """AM-2K: 1800 real animal photographs, near-binary mattes (0.7-3.9% soft).

    Real photographs rather than composites, so no compositing cost -- but also
    no glass or water. Its mattes are much easier than D-646's, which is why the
    trainer reports `generated_mse` per dataset rather than pooled.
    """
    return _build("AM2KMattingDataset", root, resolution, split,
                  overfit_samples, overfit_seed, prompt, cache_composites,
                  background_dir, pixeldit_path, "am2k", use_bbox, bbox_jitter)


class MixtureMattingDataset(Dataset):
    """Deterministic interleave of several matting datasets.

    Index i maps to source `i % n_sources`, so a 50/50 mixture really is 50/50
    at every prefix of the sequence, not just in expectation over an epoch.
    Concatenating and shuffling would let the ratio drift within any window --
    which matters here because the two datasets differ sharply in difficulty and
    a run judged before its first full epoch would be reading a biased sample.

    Each source is walked at its own pace and wraps independently, so the
    smaller dataset (AM-2k, 1800) repeats while the larger (D-646, 59,600) is
    still on its first pass. That is intended for a 50/50 weight.
    """

    def __init__(self, datasets, weights=None):
        if not datasets:
            raise ValueError("MixtureMattingDataset needs at least one dataset")
        self.datasets = list(datasets)
        if weights is not None and len(weights) != len(datasets):
            raise ValueError("weights must match datasets")
        # Integer slot pattern, e.g. weights (1, 1) -> [0, 1]; (3, 1) -> [0,0,0,1].
        w = [int(round(x)) for x in (weights or [1] * len(datasets))]
        if min(w) < 0 or sum(w) == 0:
            raise ValueError(f"invalid weights: {weights}")
        self.pattern = [i for i, k in enumerate(w) for _ in range(k)]
        # Long enough that every source is seen; length is nominal, since
        # sources wrap independently.
        self._len = max(len(d) for d in self.datasets) * len(self.pattern)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        src = self.pattern[idx % len(self.pattern)]
        ds = self.datasets[src]
        return ds[(idx // len(self.pattern)) % len(ds)]

    @property
    def names(self):
        return [d.name for d in self.datasets]


def build_dataset(names=("d646",), resolution=1024, split="train",
                  overfit_samples=0, prompt=DEFAULT_PROMPT, use_bbox=False,
                  bbox_jitter=0.1, weights=None, **kwargs):
    """Build one dataset or a mixture, by name."""
    builders = {"d646": D646MattingDataset, "am2k": AM2KMattingDataset}
    if isinstance(names, str):
        names = [names]
    parts = []
    for n in names:
        if n not in builders:
            raise ValueError(f"unknown dataset {n!r}; expected one of {sorted(builders)}")
        parts.append(builders[n](
            resolution=resolution, split=split, overfit_samples=overfit_samples,
            prompt=prompt, use_bbox=use_bbox, bbox_jitter=bbox_jitter, **kwargs))
    return parts[0] if len(parts) == 1 else MixtureMattingDataset(parts, weights)
