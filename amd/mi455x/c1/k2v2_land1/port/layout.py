"""gfx1250 K2 expert weight layout (vendored from ``/data/muse455/k2/port/layout_gfx1250.py``; tests there).

Weights: AITER (16,16) preshuffle (the MI355X single layout, unchanged). Scales: [E, N/16, K/256, 2 (step parity),
16 (row), 4 (k block)], one dword a lane feeds two WMMAs (opsel = step parity). See INTERFACE.md §1.
"""

import torch


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    """[E, N, K/2] native packed e2m1 -> (16,16) preshuffle."""
    e, n, k = w.shape
    return w.view(e, n // 16, 16, k // 16, 16).permute(0, 1, 3, 2, 4).contiguous().view(w.shape)


def shuffle_scale_gfx1250(s: torch.Tensor) -> torch.Tensor:
    """[E, N, K/32] native e8m0 -> [E, N/16, K/256, 2, 16, 4]."""
    e, n, ks = s.shape
    return s.view(e, n // 16, 16, ks // 8, 2, 4).permute(0, 1, 3, 4, 2, 5).contiguous().view(s.shape)


def w_dword(e, rg, st, lane, half, n, k):
    return e * (n * k // 8) + rg * (2 * k) + st * 256 + half * 128 + (lane // 16) * 64 + (lane % 16) * 4


def s_dword(e, rg, st, lane, n, k):
    return ((e * (n // 16) + rg) * (k // 256) + st // 2) * 32 + lane


def gfx1250_weights(w13, s13, w2, s2):
    return shuffle_weight(w13), shuffle_scale_gfx1250(s13), shuffle_weight(w2), shuffle_scale_gfx1250(s2)
