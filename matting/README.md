# Matting on HiDream-O1-Image

LoRA fine-tune of HiDream-O1-Image-Dev (8B) for alpha matting on Distinctions-646.
The task comes from the PixelDiT pilot (`PixelDiT/t2i/MATTING.md`) — dataset,
compositing, evaluation. The *recipe* comes from HiDream's own paper
(`assets/HiDream-O1-Image.pdf`) and ostris's reference implementation
(`ai-toolkit/extensions_built_in/diffusion_models/hidream/hidream_o1_model.py`).

## Why this backbone needs almost no surgery

A reference image already enters HiDream through two pretrained paths at once:
the Qwen3-VL vision encoder at 384px as condition tokens, and the same image
patchified at full resolution into a `token_type=2` stream that shares the
target's `x_embedder` (`models/pipeline.py:388`). That is the `sequence` mode of
the PixelDiT ablation, native and pretrained.

So the `patch`/`pixel`/`both`/`sequence` ablation and the
`conditioning_proj_init` question do not apply. Nothing is widened, so nothing
can break. Verified: with LoRA attached and every trainable parameter cast to
fp32, a forward pass at step 0 reproduces the stock checkpoint **bit for bit**
(`probe_zero_step_identity.py`, max |Δ| = 0.000e+00).

## Conventions, and how each was pinned

| Convention | Value | Settled by |
| --- | --- | --- |
| Head output | x₀, not velocity | `probe_flow_convention.py`; paper §3.2 |
| Model timestep | `t = 1 − σ` | `probe_flow_convention.py`; `hidream_o1_model.py:436` |
| Noise scale | **8.0**, in training as well as sampling | `probe_sampler.py`; `src/hidream_o1/pipeline.py:15` |
| Resolution | 1024² is a real operating point | `probe_sampler.py`; paper §4.1 Stage II |
| Token layout | 9 parity tests vs the shipped pipeline | `tests/test_sample_builder.py` |
| Reference stream | live — swapping it moves the prediction | `probe_sample_wiring.py` |

The loss is **velocity-space**, following ostris rather than the obvious x₀ MSE:

```python
pred   = (z - x0_pred) / sigma.clamp_min(1e-3)
target = 8.0 * eps - x0
```

which equals `MSE(x0_pred, x0) / σ²`. That is not a reparameterization — it
upweights low σ by up to 10⁶, and that factor drives the open question below.

## Units, which have bitten twice

`x0` lives in `[-1, 1]`; MATTING.md's baselines (all-black 0.266, constant-mean
0.193) are `[0, 1]` alpha. A factor 2 in range is 4 in MSE, so **divide an
x0_mse by 4 before comparing it to a published baseline**. The `vs all-black`
column of `probe_sigma_sweep.py` does this; the raw column does not.

## The finding that matters: measure at σ = 0.999

Sampling launches at σ = 0.999 and walks down. A model that is excellent in the
mid-range and useless at the launch point produces good pooled metrics and
garbage samples — at step 250 the sigmoid arm sampled four good mattes out of
eight and four flat constants.

At matched step 250, over the full schedule:

| σ | sigmoid (logit-normal) | uniform |
| ---: | ---: | ---: |
| **0.999** | **0.7525** — 71% of all-black | **0.3071** — 29% |
| 0.990 | 0.2215 | 0.1131 |
| 0.950 | 0.0709 | 0.0445 |
| 0.600 | 0.0318 | 0.0375 |
| 0.400 | 0.0283 | 0.0326 |
| 0.050 | 0.0100 | 0.0090 |

Uniform trades ~15% in the mid-range for **2.45× at the launch point**.

`sigmoid(randn)` is logit-normal: ~1.4% of its mass sits above σ 0.9 and almost
none above 0.99, so the top of the schedule stays undertrained however long the
run goes. The paper's SFT stage (§4.2) replaces logit-normal with uniform for
exactly this reason — "balanced timestep coverage".

**Validation must include σ = 0.999.** A pooled gap over `{0.3, 0.6}` measures
only where logit-normal already concentrates, and it read this A/B backwards:
the arm that looked 7 points worse on the pooled gap was 2.45× better where
sampling actually starts. `validate()` now reports per σ.

Note also that at σ = 0.999 the noisy input carries no information, so the
reference image is the *only* signal — and that is where the conditioning gap is
smallest (2% on the pretrained model). The hard part of this task is generating
a matte from the reference at high noise, not denoising one at low noise.

## Settled: uniform beats logit-normal by 41x end to end

A/B at matched step 2000, 32-sample overfit subset, 8 sampled mattes each:

| | sigmoid (logit-normal) | uniform |
| --- | ---: | ---: |
| **generated_mse** | **0.2046** | **0.0050** |
| verdict vs baselines | worse than constant-mean (0.193) | beats both |
| samples that blew up | 5 of 8 | **0 of 8** |
| worst sample | 0.597 | 0.029 |
| conditioning gap at σ∈{0.3,0.6} | **97.0%** | 71.6% |

Note the last row. **The arm with the better conditioning gap is the one that
fails at generation.** The gap is measured in the mid-range, which is exactly
where logit-normal is strong and where sampling never has to survive. A metric
can be real, well-behaved and monotonically improving and still point at the
wrong model.

The σ sweep at the same step explains it:

| σ | sigmoid | uniform |
| ---: | ---: | ---: |
| **0.999** (launch) | **0.6072** — 57% of all-black | **0.0224** — 2.1% |
| 0.990 | 0.4288 | 0.0232 |
| 0.950 | 0.0208 | 0.0220 |
| 0.600 | 0.0137 | 0.0203 |
| 0.050 | 0.0090 | 0.0069 |

Uniform is flat across the whole schedule. Sigmoid is excellent from σ 0.95 down
and falls off a cliff above it — and it *degrades* there as it trains, from
0.2215 at step 250 to 0.4288 at step 2000, forgetting a region it never
practises while improving everywhere else.

For scale against `PixelDiT/t2i/MATTING.md`: its best **stochastic** run reached
0.126, worse than the trivial baselines, and only deterministic flow got to
0.00482. This reaches 0.00498 *stochastically*, without the band loss. The
conditioning collapse there looks like a property of a backbone learning
conditioning from scratch, not of stochastic flow itself.

`timestep_type: uniform` is now the default in both configs.

## Open: the 1/σ² weight at small σ

Uniform draws σ near zero, where the velocity weight exceeds 10⁴. Measured over
logged steps:

| | sigmoid | uniform |
| --- | ---: | ---: |
| steps clipped (\|g\| > 1.0) | 0% | **30%** |
| median \|g\| | 0.070 | 0.340 |
| mean \|g\| | 0.107 | 3.415 |
| max \|g\| | 0.48 | **69.02** |

The median is under the threshold, so this is a heavy tail rather than constant
clipping — but a third of optimizer steps being clipped is real distortion. One
logged step drew σ=0.003 and produced |g| 2554 against a clip threshold of 1.0:
the update survives, scaled down 2500x, so the step is effectively wasted.

Uniform won the A/B *with* this handicap, so it is not fatal. `--sigma_min 0.05`
caps the weight at 400 and removes the tail, at the cost of never training the
last sliver of the schedule; the default stays at 1e-3 because that is what the
validated run used. Worth an A/B over a long run, not a silent default change.

## Bounding-box conditioning

HiDream's bbox support is not a coordinate embedding — **it is another reference
image**. `create_layout_reference_images` (`models/utils.py:161`) renders the
boxes onto a black canvas via `draw_bbox_layout` and appends that picture to the
reference list, where it takes the same 384px VLM encoding and the same 32×32
patch stream as any reference. There is no bbox token type and no bbox module.

RoPE places it automatically. Verified with `get_rope_index_fix_point`:

```
                              t              h              w
text prefix              0..164         0..164         0..164
target 1024px              4096     4096..4127     4096..4127
ref RGB 1024px             4128     4128..4159     4128..4159
layout 512px               4160     4160..4175     4160..4175
```

The target is pinned at absolute 4096 by `fix_point`, so text length never
shifts it, and each later raw-patch stream starts at `previous_max + 1`.

**This is not the Kontext/FLUX.2 scheme.** Those shift only the temporal channel
(`t += 10`) and leave h/w *aligned*, so image 2's pixel (5,5) shares h/w with
image 1's. HiDream adds the offset to **all three** mRoPE channels, so
corresponding pixels sit a constant 32 apart in h and w. RoPE encodes relative
position, so that displacement is learnable — it just is not handed over for
free.

Three things this implementation does deliberately:

**Only the layout image is small.** The target and the photo stay at 1024; the
layout renders at 512. The shipped heuristic (`pipeline.py:199`) would shrink
*both* references to 768 at K=2, costing the photo the fine detail matting needs.
Each reference carries its own `image_grid_thw`, so they need not match. Cost:
1024 + 1024 + 256 + ~300 ≈ 2.6k tokens against 2.4k at K=1.

**`create_layout_reference_images` is not used.** It renders the layout image
(wanted) *and* stamps a coloured border inside each photo
(`add_outer_border_keep_size`) at ≈41px on a 1024 image (not wanted). That
border binds subject↔box when there are several of each; with one object it is
redundant, and it paints over the frame edge — exactly where a subject touching
the border needs its matte. `matting/bbox.py` calls `draw_bbox_layout` directly.

**The box is jittered.** Both datasets have one foreground per image, so the box
is *redundant on every training sample* — the model can solve the task ignoring
it — and a pixel-exact box invites `alpha ≈ box interior`, which scores well here
and is useless anywhere else. `bbox_from_alpha` follows E2P's `gen_bbox`: largest
connected component, then each edge moved in or out independently by up to
`bbox_jitter` (0.1).

Watch out for one convention: **HiDream's layout input is `xxyy`** —
`[x1, x2, y1, y2]`, not `[x1, y1, x2, y2]` (`models/utils.py:62`). This codebase
uses conventional xyxy everywhere and converts only inside
`render_layout_image`.

### The number that decides whether it works

`probe_box_wiring.py` samples each image three ways — its own box, a *different*
sample's box, and no box — on the same weights:

```
box gap = (shuffled_mse - correct_mse) / shuffled_mse
```

Under 10% means the model is ignoring the box and the change is doing nothing,
whatever `generated_mse` says. Baseline on the pretrained checkpoint is **−0.9%**
(measured), so any gap after training is attributable to training rather than to
HiDream's pretrained layout ability. The probe also reports IoU between the
prediction and the box interior: near 1.0 with rectangular output means the
degenerate solution, and `bbox_jitter` needs raising.

### Dataset mixture

`build_dataset(names=["d646", "am2k"], weights=[1, 1])` interleaves by index
parity, so a 50/50 mixture is 50/50 at *every prefix*, not merely in expectation
over an epoch — concatenate-and-shuffle would let the ratio drift inside any
window, which matters because the two differ sharply in difficulty. AM-2k is 1800
real animal photographs with near-binary mattes (0.7–3.9% soft pixels); D-646 is
composited and is where transparency lives (median 6.9%). **Report
`generated_mse` per dataset**, or AM-2k's easier mattes will mask D-646
regressions.

One subtlety that bit us: validation indices are `i * stride + i`, not
`i * stride`. With a two-source interleave and an even stride, every validation
sample lands on the same dataset — with `d646+am2k` and stride 4, indices
0/4/8/12 are all D-646 and AM-2k is never validated.

## Four ways the metrics lied

Every one of these was a real measurement pointing at a wrong conclusion. They
cost hours, so they are written down.

**1. Pooled `x0_mse` is not a progress metric.** The loss divides by sigma, and
x0 error spans ~100x across the schedule, so a step's value is set mostly by
which sigma it drew. Measured in a single step: `lo 0.0006, mid 0.0679,
hi 0.1275` — a 165x spread. Worse, the pooled figure is a *mean* over the batch
while the useful signal is a median: over one run the pooled number rose from
0.007 to 0.017 while all three sigma buckets fell monotonically. Same data,
opposite conclusion. Watch `x0_mse_sigma_lo/mid/hi`, never the pooled value.

**2. A moving preview seed makes the curve unreadable.** The preview sampled
with `seed + step`, so consecutive previews of the *same four images* swung
between 0.017 and 0.233 — that is the noise draw, not the model. `--preview_seed`
is now fixed across steps.

**3. Consecutive dataset indices are the same object.** D-646 composites each
foreground over 100 backgrounds, laid out consecutively, so `range(4)` gives
four views of ONE object: four different RGB inputs sharing a byte-identical
alpha matte (verified: pairwise max|delta| 0.0000). A preview built that way
looks like the model is emitting a constant. Fixed indices are strided.

**4. The all-black baseline is per-sample, not 0.266.** It equals
`mean(alpha**2)`, i.e. foreground coverage, and across D-646 it ranges from
0.074 on a small subject to 0.601 on a large one. MATTING.md's 0.266 is that
figure for one particular subset. Comparing a rotating panel to a fixed constant
is meaningless: a fixed pair that happened to include a 23%-soft-pixel subject
read as "worse than trivial" while the rotating panel showed 5x better on the
same weights. `preview/*_vs_allblack` now computes the baseline on the samples
actually shown.

## What the loop is verified to do

Five checks, each reproducible from this directory:

| check | result |
| --- | --- |
| fit a single sample (`probe_can_it_learn.py`) | 49.9x reduction, 0.0590 -> 0.0012 |
| all intended params update | LoRA 504/504, x_embedder 3/3, final_layer2 2/2 |
| loss vs ostris `hidream_o1_model.py` | identical: timestep, x0->velocity, target, reduction |
| batching (`collate_samples`) | bit-exact vs separate batch-1 forwards, max abs delta 0 |
| step-0 identity (`probe_zero_step_identity.py`) | bit-exact vs stock checkpoint, max abs delta 0 |

Note on LoRA at init: only half the LoRA tensors receive gradient on the very
first backward. `B` is zero-initialised and `dL/dA` is proportional to `B`, so
`A` is dead until `B` moves. After one step both train; after 12 steps all
504/504 have changed. This is correct, not a bug.

## Where LoRA goes, and why not everywhere

ostris matches the *top-level class name* and wraps every Linear beneath it —
all 374 in this model. This trainer scopes by name path to the 252 text-decoder
projections, and trains the two pixel-space layers in full instead:

| group | #Linear | params | ostris | here |
| --- | ---: | ---: | --- | --- |
| text decoder (q,k,v,o,mlp) | 252 | 6,946 M | LoRA | **LoRA r16** |
| lm_head (text vocab) | 1 | 622 M | LoRA | frozen |
| vision tower | 116 | 572 M | LoRA | frozen |
| final_layer2 (pixel OUT) | 1 | 12.6 M | LoRA | **full** |
| x_embedder (pixel IN) | 2 | 7.3 M | LoRA | **full** |

`lm_head` is never called on the generation path (`x_pred = final_layer2(h)`),
so LoRA there receives no gradient at all — allocated, optimized over, never
updated. The frozen **vision tower** is the one deliberate gap worth revisiting:
116 Linears that encode the 384px reference, and matting is entirely about
reading that reference. Against that, the reference also arrives as
full-resolution patches through `x_embedder`, which does train.

On rank: E2P used 64 across four matting datasets. D-646 alone has 59,600
composites but only **596 distinct foregrounds** — the alpha matte is identical
across an object's 100 backgrounds — so rank 16 already gives ~107k trainable
parameters per distinct matte. Capacity is unlikely to be the binding
constraint here.

## Running it

```bash
# train
python -m matting.train_hidream_matting --config matting/configs/d646_1024.yaml
python -m matting.train_hidream_matting --config matting/configs/d646_1024.yaml \
    --timestep_type uniform

# gates — run after touching sample_builder.py
python -m unittest matting.tests.test_sample_builder -v
python -m matting.probe_zero_step_identity
python -m matting.probe_sample_wiring

# diagnostics
python -m matting.probe_sigma_sweep --adapter_path <run>/adapters/step_N.pth
python -m matting.sample_matting --adapter_path <run>/adapters/step_N.pth \
    --output_dir <run>/eval --num_samples 8
python -m matting.sample_matting ... --shuffle_conditions   # conditioning gap
```

Judge on `generated_mse` against the trivial baselines and on the
correct-vs-shuffled gap — never on training loss, which carries the 1/σ² weight
and is not comparable across steps that drew different σ.

## Not done yet

Batching (memory is 19.5 GB of 150 GB, so batch 1 leaves ~6× on the table;
`input_ids` is embedded directly at `qwen3_vl_transformers.py:1428` rather than
broadcast, so it needs real work). LPIPS and perceptual DINO (paper §3.4 — part
of HiDream's actual objective; cheap here because `x_pred` *is* the image, but
both networks were trained on natural images and a matte is not one). 2048²
training. Flash attention — FA4 is installed and forward-verified through a shim
at `site-packages/flash_attn_interface.py`, but backward is unverified; expect
~10–15% at this sequence length, more at 2048².
