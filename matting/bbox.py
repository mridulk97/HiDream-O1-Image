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


def bbox_from_alpha(alpha, jitter=0.1, rng=None):
    """Bounding box of the largest connected component, as normalized xyxy.

    Follows Edit2Perceive's `gen_bbox`
    (`Edit2Perceive/models/unified_dataset.py:126`): take the largest connected
    component so a stray speck of alpha does not blow the box up to the whole
    frame, then perturb each edge.

    The jitter is not cosmetic. With one foreground per composite the box is
    redundant -- the model can solve the task ignoring it -- and a pixel-exact
    box invites the degenerate solution `alpha ~= box interior`, which scores
    well on these datasets and is useless on anything real. A box that is never
    exact cannot be copied. Each edge moves independently, in or out, by up to
    `jitter` of the box's size.

    Args:
        alpha: (H, W) float array in [0, 1].
        jitter: max fractional perturbation per edge. 0 disables.
        rng: optional `random.Random` for reproducibility.

    Returns:
        [x1, y1, x2, y2] normalized to [0, 1].
    """
    if alpha.dtype == np.uint8:
        raise ValueError("alpha must be float in [0, 1], not uint8")
    rng = rng or random
    h, w = alpha.shape

    binary = alpha > 0
    if not binary.any():
        # Degenerate matte: hand back a box that is valid but carries nothing.
        return [0.0, 0.0, 1.0, 1.0]

    ys, xs = np.where(binary)
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    labeled, n = label(binary)
    if n > 1:
        sizes = np.bincount(labeled.ravel())[1:]      # drop the background label
        if sizes.size:
            coords = np.argwhere(labeled == int(np.argmax(sizes)) + 1)
            y1, x1 = coords.min(axis=0).tolist()
            y2, x2 = coords.max(axis=0).tolist()

    if jitter > 0:
        coe = rng.uniform(0, jitter)
        pad_y, pad_x = int(coe * (y2 - y1)), int(coe * (x2 - x1))
        y1 += rng.choice((-1, 1)) * pad_y
        y2 += rng.choice((-1, 1)) * pad_y
        x1 += rng.choice((-1, 1)) * pad_x
        x2 += rng.choice((-1, 1)) * pad_x
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, x2 = min(x1, x2), max(x1, x2)

    # Clamp, and keep at least one pixel of extent so the renderer accepts it.
    x1 = max(0, min(w - 2, x1)); x2 = max(x1 + 1, min(w - 1, x2))
    y1 = max(0, min(h - 2, y1)); y2 = max(y1 + 1, min(h - 1, y2))
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
