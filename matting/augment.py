"""Composite augmentation: vary object scale and position, and add distractors.

Both exist to make the bounding box carry information. Measured on the
unaugmented data, the box covers a median **0.69** of the frame on D-646 (0.59
on AM-2k), and covers more than 80% of it on 35% of samples. A box over
two-thirds of the picture says almost nothing about where the subject is, and
with one foreground per image it says nothing about *which* subject either. So
the box is both uninformative and redundant, and the model is free to ignore it
-- which `probe_box_wiring.py` exists to detect.

Applied to `aug_prob` of samples (default half); the rest take gen_train.py's
path untouched, so the original full-frame distribution stays represented and
the model is not trained exclusively on small, oddly-placed subjects.

Two changes, independently switchable:

**Scale and position.** D-646's `gen_train.py` pastes each foreground at native
size and crops the background to match, so the subject always fills the frame.
Scaling it into a random position within the canvas makes the box a real
localization signal instead of a near-constant one.

**Distractors.** Pasting other foregrounds into the same canvas, with the ground
truth still only the target's alpha, is the only way the box becomes
*load-bearing*: with several plausible objects present, the box is the sole
thing that says which one to matte. Without this the model can score perfectly
while ignoring the box entirely, so a healthy box gap is not even meaningful.

Compositing stays at native resolution, before the resize, because alpha
blending is not linear through downsampling and the difference lands exactly on
the soft edges these datasets exist to exercise. Distractors are composited into
the background first and the target blended on top, so the target's alpha is
unaffected by overlap and the ground truth stays exact.

Choices are seeded per sample id, not drawn from global RNG, so `<fg>_<k>` is
byte-identical on every epoch -- the property an overfit subset depends on. With
59,600 composites and well under one epoch of training, fixing them costs no
diversity.
"""

import hashlib
import math
import random

import numpy as np
from PIL import Image


def _rng_for(sample_id, seed):
    """Deterministic per-sample RNG, so a sample is identical on every epoch."""
    h = hashlib.md5(f"{seed}:{sample_id}".encode()).hexdigest()[:8]
    return random.Random(int(h, 16))


def _place(canvas_rgb, canvas_alpha, fg_rgb, fg_alpha, scale, cx, cy):
    """Blend a scaled foreground into the canvas at a normalized centre.

    Returns the alpha actually written, so the caller can keep the target's and
    discard the distractors'.
    """
    ch, cw = canvas_alpha.shape
    fh, fw = fg_alpha.shape
    # Fit to the canvas before applying the requested scale. Foregrounds are
    # not all smaller than the canvas -- D-646's run to 24 megapixels, and a
    # distractor drawn from another record can easily exceed the target's
    # native size -- so scaling alone does not guarantee it fits.
    fit = min(1.0, cw / fw, ch / fh)
    nw, nh = max(1, int(fw * scale * fit)), max(1, int(fh * scale * fit))

    fg_s = np.asarray(
        Image.fromarray(fg_rgb).resize((nw, nh), Image.Resampling.BICUBIC),
        dtype=np.float32)
    a_s = np.asarray(
        Image.fromarray((fg_alpha * 255).astype(np.uint8)).resize(
            (nw, nh), Image.Resampling.BILINEAR),
        dtype=np.float32) / 255.0

    # Centre in pixels, clamped so the object stays fully inside the canvas --
    # a partially cropped subject would make the ground-truth alpha disagree
    # with the foreground that produced it.
    x0 = int(round(cx * (cw - nw))) if cw > nw else 0
    y0 = int(round(cy * (ch - nh))) if ch > nh else 0
    x0 = max(0, min(cw - nw, x0))
    y0 = max(0, min(ch - nh, y0))

    sl = (slice(y0, y0 + nh), slice(x0, x0 + nw))
    a3 = a_s[..., None]
    canvas_rgb[sl] = a3 * fg_s + (1.0 - a3) * canvas_rgb[sl]
    written = np.zeros_like(canvas_alpha)
    written[sl] = a_s
    return written


def make_augmented_d646(base_cls):
    """Subclass a `Distinctions646MattingDataset` with scale/position/distractors.

    Built as a factory because the base class lives in the PixelDiT tree and is
    imported lazily; everything except `composite` is inherited, including the
    positional background assignment that keeps `<fg>_<k>` stable and the
    foreground-stratified subset selection.
    """

    class AugmentedD646(base_cls):
        def __init__(self, *args, aug_scale=None, aug_distractors=0,
                     aug_prob=0.5, aug_seed=2025, **kwargs):
            super().__init__(*args, **kwargs)
            # (min, max) multiplier on the foreground's native size. None keeps
            # gen_train.py's behaviour of pasting at full size.
            self.aug_scale = tuple(aug_scale) if aug_scale else None
            self.aug_distractors = int(aug_distractors)
            # Fraction of samples that get augmented at all. The rest take
            # gen_train.py's path untouched, so the model still sees the
            # original full-frame distribution and the augmentation does not
            # become the only thing it is ever trained on.
            self.aug_prob = float(aug_prob)
            self.aug_seed = int(aug_seed)

        def _load_fg(self, record):
            fg = Image.open(record["foreground_path"]).convert("RGB")
            al = Image.open(record["alpha_path"]).convert("L")
            if al.size != fg.size:
                al = al.resize(fg.size, Image.Resampling.BILINEAR)
            return (np.asarray(fg, dtype=np.uint8),
                    np.asarray(al, dtype=np.float32) / 255.0)

        def composite(self, record):
            if not self.aug_scale and self.aug_distractors == 0:
                return super().composite(record)

            rng = _rng_for(record["sample_id"], self.aug_seed)
            # Decided per sample id, so which samples are augmented is fixed
            # across epochs like everything else here.
            if rng.random() >= self.aug_prob:
                return super().composite(record)

            fg_rgb, fg_alpha = self._load_fg(record)
            h, w = fg_alpha.shape

            # Background fills the canvas, same rule as gen_train.py: upscale
            # only when it does not already cover, then take the top-left crop.
            bg = Image.open(self._background_path(record["background"])).convert("RGB")
            ratio = max(w / bg.size[0], h / bg.size[1])
            if ratio > 1:
                bg = bg.resize((math.ceil(bg.size[0] * ratio),
                                math.ceil(bg.size[1] * ratio)),
                               Image.Resampling.BICUBIC)
            canvas = np.asarray(bg.crop((0, 0, w, h)), dtype=np.float32)
            canvas_alpha = np.zeros((h, w), dtype=np.float32)

            lo, hi = self.aug_scale or (1.0, 1.0)

            # Distractors first, so the target always blends on top and its
            # alpha is exactly the ground truth regardless of overlap. They
            # become part of the background: their alpha is discarded.
            for d in range(self.aug_distractors):
                other = self.dataset[rng.randrange(len(self.dataset))]
                if self.foreground_key(other["sample_id"]) == \
                        self.foreground_key(record["sample_id"]):
                    continue          # never a second copy of the target object
                try:
                    d_rgb, d_alpha = self._load_fg(other)
                except Exception:
                    continue          # a missing distractor must not kill the sample
                _place(canvas, canvas_alpha, d_rgb, d_alpha,
                       rng.uniform(lo, hi), rng.random(), rng.random())

            target_alpha = _place(canvas, canvas_alpha, fg_rgb, fg_alpha,
                                  rng.uniform(lo, hi), rng.random(), rng.random())
            return (Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)),
                    target_alpha)

    AugmentedD646.__name__ = f"Augmented{base_cls.__name__}"
    return AugmentedD646
