"""Bounding-box conditioning: derive a box from an alpha matte, render it for HiDream.

HiDream's bbox conditioning is not a coordinate embedding -- it is *another
reference image*. `draw_bbox_layout` renders the boxes onto a black canvas and
that picture is appended to the reference list, where it goes through the same
384px VLM encoding and the same 32x32 patch stream as any other reference. RoPE
places it in the next offset block automatically. So all this module has to do
is produce the right picture.

Two conventions here are easy to get wrong and silent when you do:

* **HiDream's layout input is `xxyy`** -- `[x1, x2, y1, y2]`, not the usual
  `[x1, y1, x2, y2]` (`models/utils.py:62`). This module uses conventional
  **xyxy** everywhere and converts only at the boundary, in
  `render_layout_image`. Nothing else should touch the xxyy form.
The box itself is computed here rather than borrowed, because a bounding box
from an alpha matte is `np.where` plus min/max -- see `bbox_from_alpha` for the
two Edit2Perceive heuristics that were tried and dropped, and what they cost.

* **Do not call `create_layout_reference_images`.** It renders the layout image
  (wanted) *and* stamps a coloured border inside each subject photo (not
  wanted) at `sqrt(w*h) * 0.04` -- about 41px at 1024. That border binds
  subject to box when there are several of each; with one object it is
  redundant, and it paints over the frame edge, which is exactly where a
  subject touching the border needs its matte.
"""

import random

import numpy as np
from scipy.ndimage import label

from models.utils import draw_bbox_layout, parse_layout_bboxes


def bbox_from_alpha(alpha, jitter=0.05, jitter_prob=0.2, rng=None, threshold=0.0):
    """Bounding box of the foreground, as normalized xyxy.

    Deliberately the simple thing: the extent of every pixel with alpha above
    `threshold`. Two heuristics from Edit2Perceive's `gen_bbox` are *not* used
    here, because their motivation does not carry over:

    * **Largest connected component.** E2P's box selects one object among
      several. Ours localizes, and the ground truth is every foreground pixel,
      so dropping the smaller components contradicts the target -- a photo of
      two cows would get a box round one of them and a matte of both. Measured
      over 40 D-646 samples it put up to 19.9% of the alpha mass outside the
      box, against 0.3% for the full extent.
    * **Symmetric jitter.** E2P perturbs each edge in *or* out. A box that
      randomly excludes part of the subject is an incoherent localization
      claim, and it cost up to 23.7% of alpha mass outside the box. Here the
      jitter only ever expands, so the box always contains the subject while
      its exact edges still vary -- which is all that is needed to stop the
      model treating it as a ready-made mask.

    Args:
        alpha: (H, W) float array in [0, 1].
        jitter: max outward expansion per edge, as a fraction of box size.
        jitter_prob: fraction of samples that get jittered at all. The rest get
            the exact box. Jittering only sometimes means the model mostly sees
            an honest box -- so the box stays a trustworthy signal -- while
            still never being able to rely on it being exact.
        rng: optional `random.Random` for reproducibility.
        threshold: alpha above this counts as foreground.

    Returns:
        [x1, y1, x2, y2] normalized to [0, 1], as a half-open box: x2 and y2
        are one past the last foreground pixel, so `[y1*h:y2*h, x1*w:x2*w]`
        contains the subject exactly.
    """
    if alpha.dtype == np.uint8:
        raise ValueError("alpha must be float in [0, 1], not uint8")
    rng = rng or random
    h, w = alpha.shape

    ys, xs = np.where(alpha > threshold)
    if ys.size == 0:
        return [0.0, 0.0, 1.0, 1.0]        # degenerate matte
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    if jitter > 0 and rng.random() < jitter_prob:
        # Outward only. Each edge independently, so the box is not a scaled
        # copy of the true one.
        bw, bh = x2 - x1, y2 - y1
        x1 -= int(rng.uniform(0, jitter) * bw)
        x2 += int(rng.uniform(0, jitter) * bw)
        y1 -= int(rng.uniform(0, jitter) * bh)
        y2 += int(rng.uniform(0, jitter) * bh)

    x1 = max(0, x1); x2 = min(w, max(x1 + 1, x2))
    y1 = max(0, y1); y2 = min(h, max(y1 + 1, y2))
    return [x1 / w, y1 / h, x2 / w, y2 / h]


def render_layout_image(bbox_xyxy, width, height):
    """Normalized xyxy box -> the black-canvas layout image HiDream expects.

    Delegates to HiDream's own renderer so the convention the model was
    pretrained on -- black ground, `DEFAULT_COLORS[0]` red, line width from
    `get_render_params` -- is reproduced exactly rather than approximated.
    """
    x1, y1, x2, y2 = bbox_xyxy
    # The only place the xxyy quirk is allowed to appear.
    xxyy = [x1, x2, y1, y2]
    parsed = parse_layout_bboxes([xxyy], width, height)
    return draw_bbox_layout(parsed, image_width=width, image_height=height)


def shuffled_bbox(bbox_list, index):
    """A different sample's box, for the box-gap test.

    Deterministic derangement: sample i takes sample i+1's box, so nothing is
    ever paired with its own and the comparison is repeatable.
    """
    return bbox_list[(index + 1) % len(bbox_list)]
