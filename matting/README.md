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

## What deviates from ostris, and why

Worth being explicit, because these are the places this trainer does not match
the reference implementation:

| | ostris | here | controlled by |
| --- | --- | --- | --- |
| gradient clipping | `max_grad_norm` 1.0 | 1.0 | `max_grad_norm` — same, not a deviation |
| σ floor | none — `T_EPS` 1e-3 only guards the *divisor* | `sigma_min` 0.05 on the sampled σ | `sigma_min` |
| timestep sampling | `sigmoid` (logit-normal) default | uniform | `timestep_type` |
| LoRA scope | every Linear (374), pixel layers included | 252 decoder projections | `lora_scope` |
| pixel layers | LoRA'd like the rest | `x_embedder`/`final_layer2` trained in full | `no_full_train` |

**`matting/configs/d646_1024_ostris.yaml` reproduces their side of that table
exactly** — 374 LoRA layers, no full-training, sigmoid sampling, no σ floor.
It exists as the baseline to measure our deviations against, not as a
recommended recipe: `timestep_type: sigmoid` lost the sampler A/B by 41x
(generated_mse 0.20458 vs 0.00498 at matched step 2000, five of eight sampled
mattes flat). Their 374-layer scope is also *fewer* trainable parameters than
ours — 55.45M against 63.58M — because the two pixel-space layers we train in
full are 20M on their own.

On their sigmoid path specifically: `set_train_timesteps` builds a sorted
1000-entry table from `sigmoid(randn)` and the trainer draws uniformly over its
*indices*, which is drawing from the logit-normal distribution — the same thing
`sample_sigma("sigmoid")` does per step, so that piece was already faithful.

The σ floor is the one worth understanding. Ostris's `T_EPS` is a `clamp_min` at
the point of dividing, so a drawn σ of 0.0004 still yields a loss weight of 10⁶.
They never hit that because their default sampler almost never visits σ < 0.01.
The blowup only appears when ostris's velocity loss (which divides by σ) is
combined with the paper's uniform SFT sampling (§4.2) — two recommendations from
different sources that neither source runs together. Measured here: 28% of
optimizer steps clipped, max |g| 333, one step at 2554. With `sigma_min` 0.05 the
weight caps at 400 and clipping falls to 4%.

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

**The box always contains the subject, and is jittered outward only.** Two
heuristics from E2P's `gen_bbox` were tried and dropped, because their
motivation does not carry over:

* *Largest connected component.* E2P's box selects one object among several.
  Ours localizes, and the ground truth is every foreground pixel, so dropping
  the smaller components contradicts the target — a photo of two cows got a box
  round one and a matte of both. Measured over 40 D-646 samples it put up to
  **19.9%** of the alpha mass outside the box.
* *Symmetric jitter.* Perturbing each edge in *or* out means the box sometimes
  excludes part of the subject, which is an incoherent localization claim. It
  was the larger effect: up to **23.7%** outside.

`bbox_from_alpha` is therefore the simple thing — the extent of every pixel
above threshold, as a half-open box — with jitter that only expands, applied to
`bbox_jitter_prob` (0.2) of samples at up to `bbox_jitter` (0.05). Measured
after the change: **0.0000%** of alpha mass outside the box across 80 samples,
every jittered draw still enclosing the subject, 289 distinct boxes in 300 draws
so it still cannot be memorised as a mask.

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
whatever `generated_mse` says. **Read the median and the moved count, not the
mean** — the evaluator prints all three. On one checkpoint a 76.5% mean gap came
entirely from 2 of 16 samples moving ~25x while the other 14 moved 1.00x, and
the same checkpoint scored 1.2% on an overlapping set that happened to exclude
them. The medians agreed at 1.00x and 1.01x: swapping the box changes nothing
for the typical sample. Baseline on the pretrained checkpoint is **−0.9%**
(measured), so any gap after training is attributable to training rather than to
HiDream's pretrained layout ability. The probe also reports IoU between the
prediction and the box interior: near 1.0 with rectangular output means the
degenerate solution, and `bbox_jitter` needs raising.

**No bbox run has produced a usable answer yet.** The one bbox run so far started
before the `bbox_from_alpha` rewrite, so it trained on boxes that excluded up to
a quarter of their own subject. Any claim about whether the model reads the box
needs a fresh run on the corrected boxes plus this probe.

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

## Alpha losses

The flow loss alone leaves the actual matting problem largely unsolved. On a
trained checkpoint the evaluator reports MAD 0.006 in the background against
0.20 in the half-alpha bucket — **roughly 30x worse in the few percent of pixels
that are the task** — while the headline MSE reads 2% of the all-black baseline.
`--alpha_loss` acts on the alpha directly.

| | `band` | `focal` |
| --- | --- | --- |
| source | E2P `get_cycle_consistency_matting_loss` | RevealLayer, arXiv 2605.11818 eq. 16 |
| form | SAD + MSE + gradient, restricted to the trimap unknown region | `-(δ^γ)·log(1-δ)`, δ = τ·\|α̂-α\|, τ 0.95, γ 1.5 |
| needs a trimap | yes (synthesised from GT alpha) | no |
| evidence | PixelDiT: generated_mse 0.0048 → 0.00032, ~15x | untested on matting |
| magnitude at weight 1.0 | 0.07–0.55, comparable to the flow loss | 0.0004–0.033 |

**They are not interchangeable at the same weight.** Measured in a smoke test,
`band` runs up to 1.8x the flow loss while `focal` is 10–300x smaller; focal
wants `--alpha_loss_weight` around 20 to have comparable influence. The log
prints `flow` and the alpha term separately for exactly this reason.

Applied at **every** sigma with no gating, which is what E2P does
(`flux_image_new.py:184-198`). At high sigma the x0 estimate is a poor matte
rather than a meaningless one, and pushing a poor matte toward the answer is
still a valid gradient; what varies with sigma is the term's magnitude.

**Added** to the flow loss, not substituted for it. E2P substitutes — its
`return flow_loss, cycle_consistency_loss` is commented out in favour of
returning the cycle loss alone — but RevealLayer adds at λ 1.0, and PixelDiT
adds deliberately because a band mask cannot see the interior and something has
to keep it from drifting.

This is cheaper here than in either paper: E2P must invert the velocity
prediction and run a VAE decode to reach pixels, RevealLayer decodes through an
RGBA VAE, while HiDream's head emits x0 in pixel space already.

## Composite augmentation

Unaugmented, the box covers a median **0.69** of the frame on D-646 (0.59 on
AM-2k) and more than 0.80 on 35% of samples. A box over two-thirds of the
picture barely localizes, and with one foreground per image it never selects
either — so the box is both uninformative and redundant.

```yaml
aug_scale: [0.5, 0.75]   # shrink the subject and place it randomly
aug_distractors: 2       # other foregrounds, GT stays the target's alpha alone
aug_prob: 0.5            # the rest keep the original full-frame composite
```

Measured effect on box area: median 0.66 → **0.30**, spanning 0.12 to 0.99.
Keeping half the samples unaugmented matters — an earlier `[0.3, 0.7]` at 100%
produced *no* sample above 0.479, so the model would only ever have seen small
subjects and would likely have degraded on the large ones the eval sets contain.

Distractors are the part that makes the box *load-bearing*: with several
plausible objects present it is the sole cue for which one to matte. D-646 only;
AM-2k ships finished photographs with nothing to recomposite.

Compositing stays at native resolution before the resize, and every choice is
seeded per sample id, so `<fg>_<k>` is byte-identical on every epoch.

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

## Evaluation

One command from an adapter to a metrics table:

```bash
bash run_matting_eval.sh /scratch/.../adapters/step_5000.pth
```

Each dataset runs separately and completely, laid out as:

```
<EVAL_OUT>/
  aim500/
    individual_samples/<id>.png   the predicted matte on its own
    grid_samples/<id>.png         RGB | RGB + bbox | Generated | Ground Truth
    metrics.json  metrics.txt
    shuffled_image/ shuffled_box/ controls, only with EVAL_GAPS=1
  am2k_val/
    ...
```

**Two resolutions, deliberately independent.** Generation resolution is what the
model samples at and defaults to the checkpoint's trained size, since anything
else is off-grid. Metric resolution is what predictions and ground truth are
compared at, and defaults to `native` — E2P's protocol, scoring at the ground
truth's own size with the prediction upsampled to meet it
(`F.interpolate(mode="bilinear", align_corners=True)`, matching their
`resize_tensor`). That is what makes a number comparable to published results.

```bash
EVAL_RESOLUTION=512 EVAL_RES_SCOPE=metric   # generate at trained size, score at 512
EVAL_RESOLUTION=512 EVAL_RES_SCOPE=both     # generate AND score at 512
```

**State the metric resolution with any number you report.** SAD/Grad/Conn are
pixel sums: the same checkpoint scored SAD 19.30 at native and 2.62 at 512.
MSE/MAD are means and barely move. `metrics.txt` records it in a header line for
this reason.

Evaluation uses the complete dataset by default — 500 for AIM-500, 200 for AM-2k
validation. `EVAL_NUM_SAMPLES` is only for quick checks.

It reads resolution, prompt, datasets and **whether the checkpoint was trained
with a bounding box** out of the checkpoint metadata, so a bbox model is never
silently scored without one — that would not fail, it would just measure the
wrong thing. Then it samples the correct condition, samples again with the
reference photo shuffled (and with the box shuffled, for a bbox checkpoint), and
prints:

```
   dataset    n       MSE       MAD       SAD      Grad      Conn  vs all-black
      d646    4   0.00807   0.03090    32.396    34.711    26.740           3%

MAD by ground-truth alpha (the soft buckets are the task):
        background   0.00419      56.08%
  near-transparent   0.11566       6.30%
              half   0.16413       6.09%
       near-opaque   0.15752       3.72%
        foreground   0.09327      27.81%

shuffled image: MSE 0.27124 vs 0.00807   gap: 97.0%
```

Read the buckets, not the headline. The whole-image MSE is dominated by the
~84% of pixels that are flat background or flat foreground; the soft region is
about 16% of the frame and is the entire matting problem. Here it runs **39x
worse** than the background while the headline reads 3% of the all-black
baseline.

Useful env vars: `EVAL_NUM_SAMPLES`, `EVAL_SPLIT`, `EVAL_STEPS`,
`EVAL_GUIDANCE`, `EVAL_DATASETS`, `EVAL_OUT`, `EVAL_SKIP_GAPS=1`. Inference is
skipped when `results.json` already exists, so re-running only recomputes
metrics.

**The metrics are vendored, not imported.** `matting/metrics.py` holds the
P3M-Net reference implementations of SAD / MSE / MAD / Grad / Conn, obtained via
Edit2Perceive and reproduced under their MIT licence. They are copied rather
than rewritten because they are what matting papers report against, and a
from-scratch connectivity error is easy to get subtly wrong and hard to notice.
Only the metric functions came across, which is why the module needs neither
pandas nor cv2, and evaluation now imports nothing from a sibling repo.

Two local changes to that code. `compute_matting_metrics` computed the
trimap-restricted metrics unconditionally and discarded them under `whole=True`,
which made `trimap=None` crash on a path that never uses a trimap; it now skips
that work, so whole-image metrics need no trimap at all. And the non-whole path
raises a clear error instead of failing obscurely when the trimap is missing.

Note on comparability: SAD/Grad/Conn are the standard definitions and units
(SAD in thousands of pixels, alpha in [0, 1]), so the *form* matches published
numbers. But benchmarks evaluate against each dataset's provided trimap, and
neither of ours ships one. These numbers compare our runs to each other; do not
put them in a table beside published D-646 results without saying so.

## Running it

Everything below runs from `HiDream-O1-Image/` in the `hidream` conda env. The
launcher runs in the **foreground** by default so the log is your terminal;
`MATTING_DETACH=1` backgrounds it to `stdout.log` instead.

### Train

```bash
# 1. Reproduce ostris's procedure exactly -- the reference baseline
MATTING_RUN_NAME=ostris_baseline \
MATTING_CONFIG=matting/configs/d646_1024_ostris.yaml \
bash run_matting_hidream.sh

# 2. Our recipe: bbox conditioning, D-646 + AM-2k 50/50
MATTING_RUN_NAME=bbox_mix \
MATTING_CONFIG=matting/configs/mix_1024_bbox.yaml \
bash run_matting_hidream.sh

# 3. D-646 only, no bbox (the launcher's default config)
MATTING_RUN_NAME=d646_run bash run_matting_hidream.sh
```

Any config value can be overridden on the command line, and the override wins:

```bash
MATTING_RUN_NAME=lr_test bash run_matting_hidream.sh --lr 1e-4 --max_steps 6000
```

**Auto-resume.** Re-running the identical command picks up the highest
checkpoint in that run directory and reattaches to the same W&B run. For a clean
start use a new `MATTING_RUN_NAME`, or `MATTING_RESUME=0`.

Launcher variables: `MATTING_RUN_NAME`, `MATTING_CONFIG`, `MATTING_RUN_ROOT`,
`MATTING_RUN_DIR`, `MATTING_RESUME`, `MATTING_DETACH`, `MATTING_NO_WANDB`,
`MATTING_TEE`, `MATTING_SKIP_DATA_SETUP`.

### The four configs

| config | timestep | σ floor | LoRA scope | data | bbox |
| --- | --- | --- | --- | --- | --- |
| `d646_1024_ostris.yaml` | sigmoid | 0.001 | all 374 | d646 | no |
| `d646_1024.yaml` | uniform | 0.05 | decoder 252 | d646 | no |
| `d646_1024_20k.yaml` | uniform | 0.05 | decoder 252 | d646 | no |
| `mix_1024_bbox.yaml` | uniform | 0.05 | decoder 252 | d646+am2k | yes |

`d646_1024_ostris.yaml` exists to measure our deviations against, **not** as a
recommended recipe — its `timestep_type: sigmoid` lost the sampler A/B by 41x.
See "What deviates from ostris" above.

### Evaluate

One command from an adapter to a metrics table:

```bash
bash run_matting_eval.sh /scratch/mridul/runs/matting/hidream_v2/<run>/adapters/step_2000.pth
```

It reads resolution, prompt, datasets and whether the checkpoint was trained
with a box out of the checkpoint metadata, samples the correct condition plus a
shuffled-image control (and a shuffled-box control for a bbox checkpoint), and
prints per-dataset metrics with MAD bucketed by ground-truth alpha.

Variables: `EVAL_NUM_SAMPLES` (16), `EVAL_SPLIT` (train), `EVAL_STEPS` (28),
`EVAL_GUIDANCE` (1.0), `EVAL_DATASETS`, `EVAL_OUT`, `EVAL_SKIP_GAPS=1`.
Inference is skipped when `results.json` already exists, so re-running only
recomputes metrics.

### Gates and diagnostics

Run the gates after touching `sample_builder.py` or the model surgery:

```bash
python -m unittest matting.tests.test_sample_builder -v    # 9 layout parity tests
python -m matting.probe_zero_step_identity [--use_bbox]    # step 0 must be bit-exact
python -m matting.probe_sample_wiring                      # target slice + ref stream live
python -m matting.probe_can_it_learn                       # can the loop fit one sample?
```

Diagnostics, on a checkpoint:

```bash
python -m matting.probe_sigma_sweep --adapter_path <ckpt>  # where along σ it is weak
python -m matting.probe_box_wiring  --adapter_path <ckpt>  # does it read the box?
```

### What to watch

`preview/fixed_generated_mse` is the task metric. `x0_mse_sigma_lo/mid/hi` are
the readable training curves — the pooled `x0_mse` and `loss` are not, because
the loss carries a 1/σ² weight and σ is redrawn every step. And read the
soft-alpha buckets from the evaluator rather than the headline MSE: the
whole-image mean is dominated by flat background and flat foreground, and the
soft region where matting actually happens has measured 39x worse.

## Not done yet

LPIPS and perceptual DINO (paper §3.4 — part
of HiDream's actual objective; cheap here because `x_pred` *is* the image, but
both networks were trained on natural images and a matte is not one). 2048²
training. Flash attention — FA4 is installed and forward-verified through a shim
at `site-packages/flash_attn_interface.py`, but backward is unverified; expect
~10–15% at this sequence length, more at 2048².
