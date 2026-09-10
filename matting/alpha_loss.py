"""Pixel-space losses on the predicted alpha matte.

The flow loss alone is measured to leave the actual matting problem largely
unsolved. On a trained checkpoint the evaluator reports MAD 0.0042 in the
background against 0.164 in the half-alpha bucket -- **39x worse in the ~16% of
pixels that are the task** -- while the headline MSE reads 3% of the all-black
baseline. A loss that acts on the alpha itself is the direct remedy.

Applied at every timestep, with no gating on sigma. That is what Edit2Perceive
does (`pipelines/flux_image_new.py:184-198`): whatever noise level the step drew,
reconstruct the model's estimate of x0 and apply the loss to it. At high sigma
that estimate is a *poor* matte, not a meaningless one, and pushing a poor matte
toward the right answer is still a valid gradient. What changes with sigma is
the term's magnitude, which is what `alpha_loss_weight` is for.

We get this cheaper than either reference. E2P must invert the velocity
prediction and then run a VAE decode to reach pixels; RevealLayer decodes
through an RGBA VAE. HiDream's head emits x0 in pixel space directly, so the
alpha is one channel-mean away and the term costs almost nothing.

Two forms, selected by `--alpha_loss`:

``band``   SAD + MSE + gradient error restricted to the trimap unknown region,
           following E2P's `get_cycle_consistency_matting_loss`
           (`utils/cycle_loss.py:540`). This is the form with evidence on this
           exact task: PixelDiT's MATTING.md reports it moved generated_mse from
           0.0048 to 0.00032, about 15x.
``focal``  RevealLayer's hard-constraint alpha loss (arXiv 2605.11818 eq. 16),
           `-(delta**gamma) * log(1 - delta)` with `delta = tau * |a_hat - a|`,
           tau 0.95, gamma 1.5. Up-weights hard pixels wherever they are, needs
           no trimap and no morphology, and has no band radius to tune.

Both are *added* to the flow loss, not substituted for it. E2P substitutes --
its `return flow_loss, cycle_consistency_loss` is commented out in favour of
returning the cycle loss alone -- but RevealLayer adds (lambda_alpha 1.0), and
PixelDiT adds deliberately, on the grounds that a band mask cannot see the
interior and something has to keep it from drifting. Two of three add, and the
one that substitutes is the one whose flow loss is doing least.
"""

import math

import torch
import torch.nn.functional as F

TAU = 0.95      # RevealLayer eq. 16: caps delta below 1 so the log stays finite
GAMMA = 1.5     # focal exponent, same source


def alpha_from_patches(x, h_patches, w_patches, patch_size=32):
    """(B, N, C*p*p) model output -> (B, 1, H, W) alpha in [0, 1].

    The three channels carry the same matte -- the dataset replicates it because
    the backbone is an RGB pixel model -- so averaging is the inverse of that
    replication and also averages away per-channel noise.
    """
    import einops
    img = einops.rearrange(
        x, "B (H W) (C p1 p2) -> B C (H p1) (W p2)",
        H=h_patches, W=w_patches, p1=patch_size, p2=patch_size)
    return ((img.mean(dim=1, keepdim=True) + 1) / 2).clamp(0, 1)


def _box_dilate(alpha, radius):
    """Grayscale dilation by a (2r+1) square, applied separably.

    Max-pooling rather than `cv2.dilate` so this runs on-device inside the
    training step and stays differentiable-adjacent (it is used only to build a
    mask, under no_grad, but keeping it in torch avoids a host round-trip).
    """
    size = 2 * radius + 1
    d = F.max_pool2d(alpha, (1, size), stride=1, padding=(0, radius))
    return F.max_pool2d(d, (size, 1), stride=1, padding=(radius, 0))


def unknown_band(alpha, radius, fg_threshold=0.98, bg_threshold=0.02):
    """The trimap unknown region: where a dilate/erode pair disagree.

    Built from the ground-truth alpha rather than a shipped trimap, because
    neither dataset provides one -- D-646 has none, and AM-2k's exist only for
    its 200 validation images.
    """
    dilated = _box_dilate(alpha, radius)
    eroded = -_box_dilate(-alpha, radius)
    return ((dilated > bg_threshold) & (eroded < fg_threshold)).to(alpha.dtype)


def _gaussian_derivative_kernels(sigma, device, dtype):
    """Gaussian derivative kernels, matching E2P's gradient term exactly.

    Same construction as `get_cycle_consistency_matting_loss`: half-size from
    the epsilon cutoff, separable outer product, normalised by the root of the
    squared sum.
    """
    eps = 1e-2
    half = math.ceil(sigma * math.sqrt(-2 * math.log(math.sqrt(2 * math.pi) * sigma * eps)))
    coords = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    dg = -coords * g / (sigma ** 2)
    hx = g.unsqueeze(1) * dg.unsqueeze(0)
    hy = hx.t()
    hx = hx / torch.sqrt(torch.sum(hx.abs() * hx.abs()))
    hy = hy / torch.sqrt(torch.sum(hy.abs() * hy.abs()))
    return (hx.to(dtype)[None, None], hy.to(dtype)[None, None])


def _gradient_amplitude(alpha, sigma=1.4):
    kx, ky = _gaussian_derivative_kernels(sigma, alpha.device, alpha.dtype)
    gx = F.conv2d(alpha, kx, padding="same")
    gy = F.conv2d(alpha, ky, padding="same")
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def band_alpha_loss(pred, gt, radius=10):
    """SAD + MSE + gradient error on the trimap unknown band.

    Follows E2P's `get_cycle_consistency_matting_loss`, with one deliberate
    difference carried over from PixelDiT: every term is normalised by band
    size rather than a fixed /1000, so the three stay comparable to each other
    and independent of resolution and band width. E2P divides SAD and gradient
    by 1000 and MSE by the band count, which makes their relative weights depend
    on how many pixels happen to be in the band.

    Args:
        pred, gt: (B, 1, H, W) alpha in [0, 1].
        radius: dilation radius defining the band.
    """
    with torch.no_grad():
        band = unknown_band(gt, radius)
        n = band.sum(dim=(1, 2, 3)).clamp_min(1.0)

    sad = ((pred - gt).abs() * band).sum(dim=(1, 2, 3)) / n
    mse = (((pred - gt) ** 2) * band).sum(dim=(1, 2, 3)) / n
    grad = (((_gradient_amplitude(pred) - _gradient_amplitude(gt)) ** 2) * band
            ).sum(dim=(1, 2, 3)) / n
    return (sad + mse + grad).mean(), {
        "band_sad": sad.mean().item(), "band_mse": mse.mean().item(),
        "band_grad": grad.mean().item(),
        "band_frac": (n / (gt.shape[-1] * gt.shape[-2])).mean().item(),
    }


def focal_alpha_loss(pred, gt, tau=TAU, gamma=GAMMA, eps=1e-6):
    """RevealLayer's hard-constraint alpha loss (arXiv 2605.11818, eq. 16).

    `-(delta**gamma) * log(1 - delta)` with `delta = tau * |pred - gt|`.

    The `-log(1 - delta)` term grows without bound as the error approaches 1,
    so a badly wrong pixel is penalised far more than linearly; `delta**gamma`
    suppresses pixels that are already close, in the manner of focal loss. `tau`
    below 1 is what keeps the log finite at maximum error.

    Needs no trimap and no morphology, so it has no band radius to tune -- but
    it is untested on matting specifically, since RevealLayer does layer
    decomposition.
    """
    delta = (tau * (pred - gt).abs()).clamp(0, 1 - eps)
    loss = -(delta ** gamma) * torch.log1p(-delta)
    return loss.mean(), {"focal_delta_mean": delta.mean().item(),
                         "focal_delta_max": delta.max().item()}


def compute_alpha_loss(kind, pred_patches, gt_patches, h_patches, w_patches,
                       patch_size=32, band_radius=10):
    """Dispatch on `kind`, returning (loss, stats) with alpha decoded from patches."""
    if kind in (None, "none"):
        return None, {}
    pred = alpha_from_patches(pred_patches, h_patches, w_patches, patch_size)
    gt = alpha_from_patches(gt_patches, h_patches, w_patches, patch_size)
    if kind == "band":
        return band_alpha_loss(pred, gt, band_radius)
    if kind == "focal":
        return focal_alpha_loss(pred, gt)
    raise ValueError(f"unknown alpha_loss {kind!r}; expected none, band or focal")
