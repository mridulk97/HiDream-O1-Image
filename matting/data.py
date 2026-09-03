"""Distinctions-646, shaped for the HiDream training loop.

The compositing, background assignment and stratified subset selection all
already exist in `Distinctions646MattingDataset` (PixelDiT, `pixdit_datasets.py`
line 411) and are deliberately not reimplemented here: it composites at
`__getitem__` following D-646's own `gen_train.py` -- background upscaled only
if it does not already cover the foreground, cropped to the foreground's shape,
blended at *native* resolution before the resize -- and assigns backgrounds by
position so `<fg>_<k>` is byte-identical on every epoch. Compositing before the
resize is the part that matters: alpha blending is not linear through
downsampling, and the difference lands on exactly the soft edges this dataset
exists to exercise.

All this wrapper does is turn that class's nine-element tuple into a dict and
keep the PixelDiT tree importable. Both tensors arrive in [-1, 1] at
`resolution`, which is the range `build_matting_sample` and the patchifier want.
"""

import os
import sys

from torch.utils.data import Dataset

PIXELDIT_T2I = "/home/mridul/matting/PixelDiT/t2i"
DEFAULT_D646_ROOT = "/scratch/mridul/data/matting/distinctions-646"

# The prompt the probes were run under. Changing it invalidates the wiring gate,
# so it lives in one place.
DEFAULT_PROMPT = "Transform to matting map while maintaining original composition"


def _ensure_pixeldit_importable(path=PIXELDIT_T2I):
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"PixelDiT tree not found at {path}; it supplies the D-646 dataset. "
            f"Pass pixeldit_path= to point elsewhere."
        )
    if path not in sys.path:
        sys.path.insert(0, path)


class D646MattingDataset(Dataset):
    """Yields {'condition': RGB, 'alpha_rgb': 3-channel matte}, both [-1, 1].

    Alpha is replicated to three channels because the backbone is an RGB pixel
    model: it has one 32x32x3 patch embedding and one 32x32x3 output head, and
    widening either would break the pretrained function. Inference averages the
    three channels back down.
    """

    def __init__(
        self,
        root=DEFAULT_D646_ROOT,
        resolution=1024,
        split="train",
        overfit_samples=0,
        overfit_seed=2025,
        cache_composites=None,
        prompt=DEFAULT_PROMPT,
        background_dir=None,
        pixeldit_path=PIXELDIT_T2I,
    ):
        _ensure_pixeldit_importable(pixeldit_path)
        from diffusion.data.datasets.pixdit_datasets import Distinctions646MattingDataset

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

        self.inner = Distinctions646MattingDataset(
            data_dir=[root], resolution=resolution, extra=extra
        )
        self.resolution = resolution
        self.prompt = prompt

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        # (alpha_rgb, prompt, attn_mask, data_info, idx, "prompt", sample_id,
        #  category, condition) -- see pixdit_datasets.py __getitem__.
        rec = self.inner[idx]
        return {
            "alpha_rgb": rec[0],
            "condition": rec[8],
            "sample_id": rec[6],
            "category": rec[7],
            "index": idx,
        }

    def sample_ids(self):
        """The exact selection, in order -- written to the run manifest."""
        return [r["sample_id"] for r in self.inner.dataset]
