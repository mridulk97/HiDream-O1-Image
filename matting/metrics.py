"""Standard matting metrics: SAD, MSE, MAD, Grad, Conn.

Vendored so this repo has no cross-repo import for evaluation. These are the
reference implementations from P3M-Net, obtained via Edit2Perceive
(`utils/metric.py`), reproduced under the MIT licence with the original notice
below. They are the implementations matting papers report against, which is why
they are copied rather than rewritten: a from-scratch SAD or connectivity error
is easy to get subtly wrong and hard to notice, and the whole point of a
standard metric is comparability.

Only the metric functions are taken. The dead commented-out variants, the
pandas-backed MetricTracker and the depth/normal metrics that surround them in
the original are left behind, which is also why this module needs neither
pandas nor cv2.

Two things to know before reading a number off this:

* Alpha is expected in **[0, 1]**, not [0, 255]. Under that convention
  `mse_whole` is exactly the `generated_mse` the trainer reports.
* `SAD` is in units of 1000 pixels; the others are per-pixel means.

Local modification: `compute_matting_metrics` computed the trimap-restricted
metrics unconditionally and then discarded them under `whole=True`, which made
`trimap=None` crash on a path that does not use the trimap. It now skips that
work when it is not returned, so whole-image metrics can be computed without a
trimap at all.
"""

"""
Rethinking Portrait Matting with Privacy Preserving

Copyright (c) 2022, Sihan Ma (sima7436@uni.sydney.edu.au) and Jizhizi Li (jili8515@uni.sydney.edu.au)
Licensed under the MIT License (see LICENSE for details)
Github repo: https://github.com/ViTAE-Transformer/P3M-Net
Paper link: https://arxiv.org/abs/2203.16828

"""

import numpy as np
import torch
from skimage.measure import label
import scipy.ndimage.morphology
##############################
### Test loss for matting
##############################

def calculate_sad_mse_mad(predict_old,alpha,trimap):
    predict = np.copy(predict_old)
    pixel = float((trimap == 128).sum())
    predict[trimap == 255] = 1.
    predict[trimap == 0  ] = 0.
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    if pixel==0:
        pixel = trimap.shape[0]*trimap.shape[1]-float((trimap==255).sum())-float((trimap==0).sum())
    mse_diff = np.sum((predict - alpha) ** 2)/pixel
    mad_diff = np.sum(np.abs(predict - alpha))/pixel
    return sad_diff, mse_diff, mad_diff
    
def calculate_sad_mse_mad_whole_img(predict, alpha):
    pixel = predict.shape[0]*predict.shape[1]
    sad_diff = np.sum(np.abs(predict - alpha))/1000
    mse_diff = np.sum((predict - alpha) ** 2)/pixel
    mad_diff = np.sum(np.abs(predict - alpha))/pixel
    return sad_diff, mse_diff, mad_diff	

def calculate_sad_fgbg(predict, alpha, trimap):
    sad_diff = np.abs(predict-alpha)
    weight_fg = np.zeros(predict.shape)
    weight_bg = np.zeros(predict.shape)
    weight_trimap = np.zeros(predict.shape)
    weight_fg[trimap==255] = 1.
    weight_bg[trimap==0  ] = 1.
    weight_trimap[trimap==128  ] = 1.
    sad_fg = np.sum(sad_diff*weight_fg)/1000
    sad_bg = np.sum(sad_diff*weight_bg)/1000
    sad_trimap = np.sum(sad_diff*weight_trimap)/1000
    return sad_fg, sad_bg

# def compute_gradient_whole_image(pd, gt):
#     from scipy.ndimage import gaussian_filter
#     pd_x = gaussian_filter(pd, sigma=1.4, order=[1, 0], output=np.float32)
#     pd_y = gaussian_filter(pd, sigma=1.4, order=[0, 1], output=np.float32)
#     gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
#     gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
#     pd_mag = np.sqrt(pd_x**2 + pd_y**2)
#     gt_mag = np.sqrt(gt_x**2 + gt_y**2)

#     error_map = np.square(pd_mag - gt_mag)
#     loss = np.sum(error_map) / 10
#     return loss

# def compute_connectivity_loss_whole_image(pd, gt, trimap=None, step=0.1):
#     from scipy.ndimage import morphology
#     from skimage.measure import label, regionprops
#     h, w = pd.shape
#     thresh_steps = np.arange(0, 1.1, step)
#     l_map = -1 * np.ones((h, w), dtype=np.float32)
#     lambda_map = np.ones((h, w), dtype=np.float32)
#     for i in range(1, thresh_steps.size):
#         pd_th = pd >= thresh_steps[i]
#         gt_th = gt >= thresh_steps[i]
#         label_image = label(pd_th & gt_th, connectivity=1)
#         cc = regionprops(label_image)
#         size_vec = np.array([c.area for c in cc])
#         if len(size_vec) == 0:
#             continue
#         max_id = np.argmax(size_vec)
#         coords = cc[max_id].coords
#         omega = np.zeros((h, w), dtype=np.float32)
#         omega[coords[:, 0], coords[:, 1]] = 1
#         flag = (l_map == -1) & (omega == 0)
#         l_map[flag == 1] = thresh_steps[i-1]
#         dist_maps = morphology.distance_transform_edt(omega==0)
#         dist_maps = dist_maps / dist_maps.max()
#     l_map[l_map == -1] = 1
#     d_pd = pd - l_map
#     d_gt = gt - l_map
#     phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
#     phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
#     if trimap is not None:
#         loss = np.sum(np.abs(phi_pd - phi_gt) * (trimap == 128)) / 1000
#     else:
#         loss = np.sum(np.abs(phi_pd - phi_gt)) / 1000
#     return loss


def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int_)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy

def compute_gradient_loss(pred, target, trimap=None):

    if pred.dtype == np.uint8:
        pred = pred / 255.0
    if target.dtype == np.uint8:
        target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    if trimap is not None:
        loss = np.sum(error_map[trimap == 128])
    else:
        loss = np.sum(error_map)
    return loss / 1000.


def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def compute_connectivity_error(pred, target, trimap=None, step=0.1):
    if pred.dtype == np.uint8:
        pred = pred / 255.0
    if target.dtype == np.uint8:
        target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=np.float32) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int_)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int_)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(np.int_)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int_)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int_)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int_)
    if trimap is not None:
        loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])
    else:
        loss = np.sum(np.abs(pred_phi - target_phi)) 

    return loss / 1000.

def compute_matting_metrics(pred, alpha, trimap=None, whole=False):
    if isinstance(pred, torch.Tensor):
        pred = pred.squeeze().cpu().numpy()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.squeeze().cpu().numpy()
    if isinstance(trimap, torch.Tensor):
        trimap = trimap.squeeze().cpu().numpy()
    if whole:
        # Whole-image metrics need no trimap. The original computed the
        # trimap-restricted ones here too and discarded them, which is what made
        # trimap=None crash on a path that does not use it; the discarded
        # sad_fg/sad_bg call did the same.
        sad_whole, mse_whole, mad_whole = calculate_sad_mse_mad_whole_img(pred, alpha)
        gradient_loss = compute_gradient_loss(pred, alpha)
        connectivity_loss = compute_connectivity_error(pred, alpha)
        return [mse_whole, mad_whole, sad_whole, gradient_loss, connectivity_loss]

    if trimap is None:
        raise ValueError(
            "trimap-restricted metrics need a trimap; pass whole=True for "
            "whole-image metrics instead"
        )
    sad, mse, mad = calculate_sad_mse_mad(pred, alpha, trimap)
    conn = compute_gradient_loss(pred, alpha, trimap)
    return [sad, mse, mad, conn]
