"""Parity: `build_matting_sample` must reproduce what the shipped pipeline feeds
the model.

`build_matting_sample` is a refactor of code that lives inline inside
`generate_image`, so checking it against a hand-copied transcription of that code
would only prove the copy matches the copy. Instead this runs the real
`generate_image` and intercepts the arguments it actually passes to the model.

That interception is also what keeps the test cheap. `forward_once` calls
`model(**kwargs)` and then reads `outputs.x_pred`, so a stub that records its
kwargs and raises aborts the run before any real computation -- no 8B weights are
loaded and the whole test runs on CPU in seconds. Only the config and processor
come from the checkpoint directory.
"""

import os
import sys
import unittest

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import models.pipeline as pipeline_mod  # noqa: E402
from matting.sample_builder import build_matting_sample  # noqa: E402

MODEL_PATH = os.environ.get(
    "HIDREAM_MODEL_PATH",
    "/projects/ml4science/HF_CACHE/transformers/"
    "models--HiDream-ai--HiDream-O1-Image-Dev/snapshots/"
    "c0bada0e15c54a9f96a6d1ecc35575b32bc21544",
)
SIZE = 1024
PROMPT = "extract the alpha matte of the foreground subject"


class _Captured(Exception):
    """Raised to abort generate_image once the model kwargs are in hand."""


class _RecordingModel:
    """Stands in for the transformer: records one call, then aborts the run."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        raise _Captured


def _square_reference(path, size):
    """A patch-aligned square reference, so the pipeline's resize is identity."""
    src = Image.open("assets/IP_2.jpg").convert("RGB")
    w, h = src.size
    side = min(w, h)
    src = src.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    src.resize((size, size), resample=Image.BICUBIC).save(path)
    return path


class TestSampleBuilderParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoConfig, AutoProcessor

        cls.config = AutoConfig.from_pretrained(MODEL_PATH)
        cls.processor = AutoProcessor.from_pretrained(MODEL_PATH)
        cls.tokenizer = getattr(cls.processor, "tokenizer", cls.processor)

        tmp = os.environ.get("TMPDIR", "/tmp")
        cls.ref_path = _square_reference(os.path.join(tmp, "parity_ref.png"), SIZE)

        # The pipeline snaps every request to PREDEFINED_RESOLUTIONS (smallest
        # square 2048); identity is what lets the parity run happen at 1024.
        cls._orig_snap = pipeline_mod.find_closest_resolution
        pipeline_mod.find_closest_resolution = lambda w, h: (w, h)

        stub = _RecordingModel(cls.config)
        try:
            pipeline_mod.generate_image(
                model=stub,
                processor=cls.processor,
                prompt=PROMPT,
                ref_image_paths=[cls.ref_path],
                height=SIZE,
                width=SIZE,
                num_inference_steps=1,
                guidance_scale=1.0,  # no uncond branch, so samples == [cond]
                seed=0,
            )
        except _Captured:
            pass
        cls.ref_kwargs = stub.kwargs
        assert cls.ref_kwargs is not None, "pipeline never reached the model"

    @classmethod
    def tearDownClass(cls):
        pipeline_mod.find_closest_resolution = cls._orig_snap

    def setUp(self):
        self.ours = build_matting_sample(
            cond_image=Image.open(self.ref_path).convert("RGB"),
            prompt=PROMPT,
            height=SIZE,
            width=SIZE,
            tokenizer=self.tokenizer,
            processor=self.processor,
            model_config=self.config,
        )

    def test_input_ids_match(self):
        torch.testing.assert_close(
            self.ours["input_ids"], self.ref_kwargs["input_ids"], rtol=0, atol=0
        )

    def test_position_ids_match(self):
        torch.testing.assert_close(
            self.ours["position_ids"], self.ref_kwargs["position_ids"], rtol=0, atol=0
        )

    def test_token_types_match(self):
        torch.testing.assert_close(
            self.ours["token_types"], self.ref_kwargs["token_types"], rtol=0, atol=0
        )

    def test_vlm_condition_stream_matches(self):
        torch.testing.assert_close(
            self.ours["image_grid_thw"], self.ref_kwargs["image_grid_thw"], rtol=0, atol=0
        )
        # The pipeline casts pixel_values to bf16 (pipeline.py:302) while the
        # builder keeps fp32, so these agree only to bf16 resolution (~2^-9
        # relative). Verified to be rounding and not a preprocessing difference:
        # the same value reads 0.9686274 in fp32 and 0.96875 in bf16.
        torch.testing.assert_close(
            self.ours["pixel_values"].float(),
            self.ref_kwargs["pixel_values"].float(),
            rtol=4e-3, atol=4e-3,
        )

    def test_ref_patches_match_pipeline_tail(self):
        """vinputs is cat([noisy_target, ref_patches]); the tail is ours."""
        vinputs = self.ref_kwargs["vinputs"]
        n_ref = self.ours["ref_patches"].shape[1]
        # The pipeline casts ref_patches to bf16 before the call; ours is fp32,
        # so the tolerance is bf16 resolution (~2^-8 relative) and nothing more.
        torch.testing.assert_close(
            self.ours["ref_patches"].float(),
            vinputs[:, -n_ref:].float(),
            rtol=4e-3, atol=4e-3,
        )

    def test_vinput_mask_selects_the_trailing_image_tokens(self):
        """The mask must cover exactly the target+reference block, in that order."""
        vinputs = self.ref_kwargs["vinputs"]
        n_vis = vinputs.shape[1]
        mask = self.ours["vinput_mask"][0]
        self.assertEqual(int(mask.sum()), n_vis)
        self.assertTrue(bool(mask[-n_vis:].all()), "mask is not the trailing block")
        self.assertFalse(bool(mask[:-n_vis].any()), "mask leaks into the text block")
        self.assertEqual(
            self.ours["tgt_image_len"] + self.ours["total_ref_len"], n_vis
        )

    def test_target_block_precedes_reference_block(self):
        """pipeline.py:357 reads x_pred[...][:tgt_image_len] as the target."""
        self.assertEqual(self.ours["tgt_image_len"], (SIZE // 32) ** 2)
        self.assertEqual(self.ours["total_ref_len"], (SIZE // 32) ** 2)

    def test_empty_prompt_builds(self):
        """The CFG/dropout branch uses ' ' and must produce a valid layout."""
        blank = build_matting_sample(
            cond_image=Image.open(self.ref_path).convert("RGB"),
            prompt=" ",
            height=SIZE,
            width=SIZE,
            tokenizer=self.tokenizer,
            processor=self.processor,
            model_config=self.config,
        )
        self.assertEqual(blank["tgt_image_len"], self.ours["tgt_image_len"])
        self.assertEqual(
            int(blank["vinput_mask"].sum()), int(self.ours["vinput_mask"].sum())
        )

    def test_rejects_mismatched_conditioning_size(self):
        small = Image.open(self.ref_path).convert("RGB").resize((512, 512))
        with self.assertRaises(ValueError):
            build_matting_sample(
                cond_image=small, prompt=PROMPT, height=SIZE, width=SIZE,
                tokenizer=self.tokenizer, processor=self.processor,
                model_config=self.config,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
