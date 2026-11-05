from typing import Callable

import numpy as np
import torch
from scipy.ndimage import label as cc_label
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.clip_tp = clip_tp
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp , max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn

        dc = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc

class AdaptiveDiceLoss(SoftDiceLoss):

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False,
                 do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True, clip_tp: float = None,
                 alpha_min: float = 1.0, alpha_max: float = 10.0,
                 ref_fraction: float = 0.01, gamma: float = 0.5):
        super().__init__(apply_nonlin, batch_dice, do_bg, smooth, ddp, clip_tp)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.ref_fraction = ref_fraction
        self.gamma = gamma

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Siempre por muestra: tp, fp, fn quedan con forma (batch, clases)
        axes = list(range(2, x.ndim))
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp, max=None)

        dc = (2 * tp + self.smooth) / torch.clip(2 * tp + fp + fn + self.smooth, 1e-8)

        # === Factor adaptativo ===
        # tp + fn = vóxeles de lesión reales en el parche. Sin gradiente.
        n_voxels = float(np.prod(x.shape[2:]))
        gt_fraction = (tp + fn).detach() / n_voxels

        if not self.do_bg:
            dc = dc[:, 1:]
            gt_fraction = gt_fraction[:, 1:]

        has_fg = gt_fraction > 0
        if not has_fg.any():
            # Ningún parche del batch tiene lesión: pérdida 0 conectada al grafo
            return x.sum() * 0.

        # Entre más pequeña la lesión, mayor el peso alpha
        alpha = (self.ref_fraction / gt_fraction[has_fg]) ** self.gamma
        alpha = torch.clamp(alpha, self.alpha_min, self.alpha_max)

        # Se pondera el error (1 - dice), no el acierto
        return (alpha * (1 - dc[has_fg])).mean()


class AdaptiveDiceLoss_v2(nn.Module):
    """
    Adaptive Dice Loss (ADL) / volume-aware Dice loss de Hu et al. (2019), en la formulación de
    Chartrand et al. (2022, J Magn Reson Imaging 56:1885-1898, ec. 2):

                     C · gᵀ W p + ε
        L_VA = ─────────────────────────,   W_ii = sqrt(λ / vol(i)),   C = 1 + gᵀg / gᵀWg
                   pᵀp + gᵀ W g + ε

    g: referencia binaria, p: probabilidad predicha, vol(i): volumen de la lesión (componente conectado)
    a la que pertenece el vóxel i, λ: número de vóxeles de la lesión más grande del batch.
    Como en el artículo, los vectores abarcan todo el batch (batch_dice se ignora).
    Se devuelve 1 - L_VA para poder minimizarla. Con todos los pesos iguales a 1 equivale a la Dice estándar.

    min_weighted_volume: las lesiones con menos vóxeles reciben peso 1 en vez de sqrt(λ / vol).
    Por defecto 0 (desactivado), es decir, la formulación original del artículo.
    """

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1., ddp: bool = True, clip_tp: float = None, min_weighted_volume: int = 0):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.do_bg = do_bg
        self.eps = smooth
        self.min_weighted_volume = min_weighted_volume

    def _volume_weights(self, g: np.ndarray) -> np.ndarray:
        # g: (b, *spatial) binaria. Cada vóxel de lesión recibe el volumen de su componente conectado
        # (conectividad 26, la misma de evaluate_predictions.py); el fondo queda en 0
        vol = np.zeros(g.shape, dtype=np.float32)
        structure = np.ones((3,) * (g.ndim - 1), dtype=bool)
        for b in range(g.shape[0]):
            labeled, n = cc_label(g[b], structure=structure)
            if n == 0:
                continue
            sizes = np.bincount(labeled.ravel()).astype(np.float32)
            sizes[0] = 0
            vol[b] = sizes[labeled]
        lam = vol.max()  # λ: lesión más grande del batch
        w = np.zeros_like(vol)
        lesion = vol > 0
        w[lesion] = np.sqrt(lam / vol[lesion])
        if self.min_weighted_volume > 0:
            w[lesion & (vol < self.min_weighted_volume)] = 1.0
        return w

    def forward(self, x, y, loss_mask=None):
        x = x.float()
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # Referencia en one-hot (b, c, ...), igual que get_tp_fp_fn_tn
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y.float()
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
                y_onehot.scatter_(1, y.long(), 1)

        if loss_mask is not None:
            x = x * loss_mask
            y_onehot = y_onehot * loss_mask

        first_class = 0 if self.do_bg else 1
        losses = []
        for c in range(first_class, x.shape[1]):
            p, g = x[:, c], y_onehot[:, c]
            gsum = g.sum()
            if gsum == 0:
                continue  # sin lesión de esta clase en el batch: de esos vóxeles se encarga la CE

            w = torch.from_numpy(self._volume_weights(g.detach().cpu().numpy() > 0)).to(x.device)
            gWg = (g * w * g).sum()
            C = 1 + gsum / gWg  # gᵀg = Σg porque g es binaria
            l_va = (C * (g * w * p).sum() + self.eps) / ((p * p).sum() + gWg + self.eps)
            losses.append(1 - l_va)

        if len(losses) == 0:
            # Ningún parche del batch tiene lesión: pérdida 0 conectada al grafo
            return x.sum() * 0.
        return torch.stack(losses).mean()


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            y_onehot = torch.zeros(net_output.shape, device=net_output.device, dtype=torch.bool)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (~y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (~y_onehot)

    if mask is not None:
        with torch.no_grad():
            mask_here = torch.tile(mask, (1, tp.shape[1], *[1 for _ in range(2, tp.ndim)]))
        tp *= mask_here
        fp *= mask_here
        fn *= mask_here
        tn *= mask_here
        # benchmark whether tiling the mask would be faster (torch.tile). It probably is for large batch sizes
        # OK it barely makes a difference but the implementation above is a tiny bit faster + uses less vram
        # (using nnUNetv2_train 998 3d_fullres 0)
        # tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        # fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        # fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        # tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, keepdim=False)
        fp = fp.sum(dim=axes, keepdim=False)
        fn = fn.sum(dim=axes, keepdim=False)
        tn = tn.sum(dim=axes, keepdim=False)

    return tp, fp, fn, tn


if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_soft = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_mem = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_adl = AdaptiveDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)

    res_soft = dl_soft(pred, ref)
    res_mem = dl_mem(pred, ref)
    res_adl = dl_adl(pred, ref)

    print(f"SoftDice:        {res_soft}")
    print(f"MemEfficient:    {res_mem}")
    print(f"AdaptiveDice:    {res_adl}")
