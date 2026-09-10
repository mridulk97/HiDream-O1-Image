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
DEFAULT_AIM_ROOT = "/scratch/mridul/data/matting/aim-500"

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

    def __init__(self, inner, resolution, prompt, use_bbox, bbox_jitter,
                 bbox_jitter_prob, name):
        self.inner = inner
        self.resolution = resolution
        self.prompt = prompt
        self.use_bbox = use_bbox
        self.bbox_jitter = bbox_jitter
        self.bbox_jitter_prob = bbox_jitter_prob
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
            item["bbox"] = bbox_from_alpha(alpha, self.bbox_jitter,
                                           self.bbox_jitter_prob)
        return item

    def sample_ids(self):
        return [r["sample_id"] for r in self.inner.dataset]


def _build(cls_name, root, resolution, split, overfit_samples, overfit_seed,
           prompt, cache_composites, background_dir, pixeldit_path, name,
           use_bbox, bbox_jitter, bbox_jitter_prob, aug=None):
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
    cls = getattr(pd, cls_name)
    kw = {}
    if aug and (aug.get("aug_scale") or aug.get("aug_distractors")):
        if cls_name != "Distinctions646MattingDataset":
            raise ValueError(
                "composite augmentation needs foreground/background sources; "
                "AM-2k ships finished photographs, so only D-646 supports it")
        from matting.augment import make_augmented_d646
        cls = make_augmented_d646(cls)
        kw = {k: aug[k] for k in ("aug_scale", "aug_distractors", "aug_prob",
                                  "aug_seed") if k in aug and aug[k] is not None}
        # Augmented composites are not the cached ones: scale, position and
        # distractors are baked in at composite time, so a cache keyed on the
        # sample id is still correct, but only because the choices are seeded
        # per sample id and therefore identical on every epoch.
    inner = cls(data_dir=[root], resolution=resolution, extra=extra, **kw)
    return _MattingAdapter(inner, resolution, prompt, use_bbox, bbox_jitter,
                           bbox_jitter_prob, name)


def D646MattingDataset(root=DEFAULT_D646_ROOT, resolution=1024, split="train",
                       overfit_samples=0, overfit_seed=2025, cache_composites=None,
                       prompt=DEFAULT_PROMPT, background_dir=None,
                       pixeldit_path=PIXELDIT_T2I, use_bbox=False, bbox_jitter=0.05,
                       bbox_jitter_prob=0.2, aug=None):
    """Distinctions-646: 596 foregrounds x 100 backgrounds, composited on the fly.

    This is where transparency lives -- glass, water, veils, fine hair, median
    6.9% soft pixels.
    """
    return _build("Distinctions646MattingDataset", root, resolution, split,
                  overfit_samples, overfit_seed, prompt, cache_composites,
                  background_dir, pixeldit_path, "d646", use_bbox, bbox_jitter,
                  bbox_jitter_prob, aug)


def AM2KMattingDataset(root=DEFAULT_AM2K_ROOT, resolution=1024, split="train",
                       overfit_samples=0, overfit_seed=2025, cache_composites=None,
                       prompt=DEFAULT_PROMPT, background_dir=None,
                       pixeldit_path=PIXELDIT_T2I, use_bbox=False, bbox_jitter=0.05,
                       bbox_jitter_prob=0.2):
    """AM-2K: 1800 real animal photographs, near-binary mattes (0.7-3.9% soft).

    Real photographs rather than composites, so no compositing cost -- but also
    no glass or water. Its mattes are much easier than D-646's, which is why the
    trainer reports `generated_mse` per dataset rather than pooled.
    """
    return _build("AM2KMattingDataset", root, resolution, split,
                  overfit_samples, overfit_seed, prompt, cache_composites,
                  background_dir, pixeldit_path, "am2k", use_bbox, bbox_jitter,
                  bbox_jitter_prob, None)


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

    def __getattr__(self, item):
        # Expose `native_alpha` only when every source can provide it, so a
        # mixture of benchmarks scores at native resolution while a mixture
        # containing a composited dataset falls back instead of half-applying
        # the protocol.
        if item == "native_alpha":
            ds = object.__getattribute__(self, "datasets")
            if all(hasattr(d, "native_alpha") for d in ds):
                def _native(idx):
                    pat = object.__getattribute__(self, "pattern")
                    src = ds[pat[idx % len(pat)]]
                    return src.native_alpha((idx // len(pat)) % len(src))
                return _native
        raise AttributeError(item)


class ConcatMattingDataset(Dataset):
    """Every sample of every source, in order -- for evaluation.

    `MixtureMattingDataset` interleaves to hold a training ratio at every
    prefix, which is the wrong shape for a benchmark: asking it for 700 samples
    across a 500- and a 200-image set yields 350 of one and the other wrapped
    and deduplicated, so you silently score 550 images and miss 150. Evaluation
    wants each image exactly once.
    """

    def __init__(self, datasets):
        self.datasets = list(datasets)
        self.offsets, n = [], 0
        for d in self.datasets:
            self.offsets.append(n)
            n += len(d)
        self._len = n

    def __len__(self):
        return self._len

    def _locate(self, idx):
        for k in range(len(self.datasets) - 1, -1, -1):
            if idx >= self.offsets[k]:
                return self.datasets[k], idx - self.offsets[k]
        raise IndexError(idx)

    def __getitem__(self, idx):
        ds, i = self._locate(idx)
        return ds[i]

    def native_alpha(self, idx):
        ds, i = self._locate(idx)
        if not hasattr(ds, "native_alpha"):
            raise AttributeError("native_alpha")
        return ds.native_alpha(i)

    @property
    def names(self):
        return [d.name for d in self.datasets]


def build_dataset(names=("d646",), resolution=1024, split="train",
                  overfit_samples=0, prompt=DEFAULT_PROMPT, use_bbox=False,
                  bbox_jitter=0.05, bbox_jitter_prob=0.2, weights=None,
                  aug=None, concat=False, **kwargs):
    """Build one dataset, an interleaved mixture, or a concatenation.

    `concat=True` gives every sample of every source exactly once, which is what
    evaluation needs; the default interleave is for training, where holding the
    ratio at every prefix matters.
    """
    builders = {"d646": D646MattingDataset, "am2k": AM2KMattingDataset,
                "aim500": AIM500Dataset, "am2k_val": AM2KValDataset}
    if isinstance(names, str):
        names = [names]
    parts = []
    for n in names:
        if n not in builders:
            raise ValueError(f"unknown dataset {n!r}; expected one of {sorted(builders)}")
        extra_kw = {"aug": aug} if n == "d646" else {}
        if n in ("aim500", "am2k_val"):
            # No compositing, so the PixelDiT-specific knobs do not apply.
            parts.append(builders[n](
                resolution=resolution, prompt=prompt, use_bbox=use_bbox,
                bbox_jitter=bbox_jitter, bbox_jitter_prob=bbox_jitter_prob,
                overfit_samples=overfit_samples))
            continue
        parts.append(builders[n](
            resolution=resolution, split=split, overfit_samples=overfit_samples,
            prompt=prompt, use_bbox=use_bbox, bbox_jitter=bbox_jitter,
            bbox_jitter_prob=bbox_jitter_prob, **extra_kw, **kwargs))
    if len(parts) == 1:
        return parts[0]
    return (ConcatMattingDataset(parts) if concat
            else MixtureMattingDataset(parts, weights))


class FlatMattingEvalDataset(Dataset):
    """A benchmark laid out as matched `original/ mask/ trimap/` triplets.

    Covers AIM-500 and AM-2k's validation split, which share that layout
    exactly. Both are evaluation sets, not training data.

    These are the only sources here that ship real trimaps -- D-646 has none at
    all, and AM-2k's live inside its validation zip and are not extracted by
    `setup_am2k_data.sh`. That matters twice over: the bounding box is derived
    from the trimap rather than the alpha, and the ground truth is available at
    its native resolution, which is the size Edit2Perceive scores at.

    Loaded directly rather than through PixelDiT: there is nothing to composite,
    just matched triplets keyed by filename stem.
    """

    def __init__(self, root=DEFAULT_AIM_ROOT, resolution=1024,
                 prompt=DEFAULT_PROMPT, use_bbox=False, bbox_jitter=0.0,
                 bbox_jitter_prob=0.0, overfit_samples=0, split="test",
                 name="aim500", subdir="", **kwargs):
        import glob
        self.root = root
        # AM-2k nests its split under validation/; AIM-500 is flat.
        root = os.path.join(root, subdir) if subdir else root
        self.resolution = int(resolution)
        self.prompt = prompt
        self.use_bbox = use_bbox
        self.bbox_jitter = bbox_jitter
        self.bbox_jitter_prob = bbox_jitter_prob
        self.name = name
        stems = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(root, "original", "*")))
        if not stems:
            raise FileNotFoundError(
                f"no images under {root}/original -- extract AIM-500 there first")
        self.records = []
        for st in stems:
            img = glob.glob(os.path.join(root, "original", st + ".*"))
            msk = glob.glob(os.path.join(root, "mask", st + ".*"))
            tri = glob.glob(os.path.join(root, "trimap", st + ".*"))
            if img and msk:
                self.records.append(
                    {"sample_id": st, "image": img[0], "mask": msk[0],
                     "trimap": tri[0] if tri else None,
                     # AIM-500 ids carry no category, so the stem stands in --
                     # the strided validation selection only needs distinctness.
                     "category": st})
        if overfit_samples:
            self.records = self.records[:int(overfit_samples)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        import numpy as np
        import torch
        from PIL import Image
        r = self.records[idx]
        n = self.resolution
        img = Image.open(r["image"]).convert("RGB").resize((n, n), Image.BICUBIC)
        alpha = Image.open(r["mask"]).convert("L").resize((n, n), Image.BILINEAR)
        cond = torch.from_numpy(
            np.ascontiguousarray(np.asarray(img, dtype=np.float32) / 255.0)
        ).permute(2, 0, 1) * 2.0 - 1.0
        a = torch.from_numpy(np.asarray(alpha, dtype=np.float32) / 255.0)[None] * 2.0 - 1.0
        item = {"alpha_rgb": a.expand(3, -1, -1).contiguous(), "condition": cond,
                "sample_id": r["sample_id"], "category": r["category"],
                "index": idx, "dataset": self.name}
        if r["trimap"]:
            # NEAREST: a trimap is three labels, and interpolating between 0 and
            # 255 would invent unknown-region pixels that are not there.
            t = Image.open(r["trimap"]).convert("L").resize((n, n), Image.NEAREST)
            item["trimap"] = np.asarray(t, dtype=np.float32)
        if self.use_bbox:
            from matting.bbox import bbox_from_alpha
            # From the TRIMAP where one exists, not the alpha. The trimap's
            # non-background region (unknown 128 plus foreground 255) is what a
            # human annotator marked as "could be foreground", so its extent is
            # the honest localization box -- slightly larger than the alpha's,
            # by the width of the unknown band. Training has to fall back to the
            # alpha because neither D-646 nor AM-2k train ships a trimap; here
            # we have the real thing, so use it.
            src = (item["trimap"] / 255.0 if "trimap" in item
                   else ((item["alpha_rgb"][0].float() + 1) / 2).clamp(0, 1).numpy())
            item["bbox"] = bbox_from_alpha(
                src, self.bbox_jitter, self.bbox_jitter_prob)
        return item

    def native_alpha(self, idx):
        """Ground-truth alpha at its ORIGINAL resolution, plus the trimap.

        Edit2Perceive evaluates at the ground truth's native size, upsampling
        the prediction to meet it (`utils/eval_matting.py:12-23`), rather than
        downsampling the truth to the model's output grid. That is the harder
        and more honest comparison, and it is what makes a number comparable to
        published AIM-500 results -- so the evaluator needs the untouched alpha,
        not the square-resized one `__getitem__` returns.
        """
        import numpy as np
        from PIL import Image
        r = self.records[idx]
        alpha = np.asarray(Image.open(r["mask"]).convert("L"), dtype=np.float32) / 255.0
        tri = None
        if r["trimap"]:
            tri = np.asarray(Image.open(r["trimap"]).convert("L"), dtype=np.float32)
        return alpha, tri

    def sample_ids(self):
        return [r["sample_id"] for r in self.records]


def AIM500Dataset(root=DEFAULT_AIM_ROOT, **kwargs):
    """AIM-500: 500 natural images, flat original/mask/trimap."""
    return FlatMattingEvalDataset(root=root, name="aim500", subdir="", **kwargs)


def AM2KValDataset(root=DEFAULT_AM2K_ROOT, **kwargs):
    """AM-2k's official validation split: 200 animal photographs with trimaps.

    Trimaps must be extracted first -- `setup_am2k_data.sh` deliberately skips
    them to halve the footprint, so they sit unextracted inside
    `validation-*.zip`.
    """
    return FlatMattingEvalDataset(root=root, name="am2k_val",
                                  subdir="validation", **kwargs)
