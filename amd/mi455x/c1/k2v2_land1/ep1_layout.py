"""TP4/EP1 expert slicing for the gfx1250 K2 (MI455X), from the Quark MXFP4 checkpoint.

Every rank keeps all 256 experts and 1/TP of each expert's intermediate dimension (the C1
``_tp_expert_stack_slice`` rule, ported to the per-expert checkpoint tensors):

* w13 rows  = [gate rows r*I/TP .. (r+1)*I/TP ; up rows r*I/TP .. (r+1)*I/TP]       -> [2*I/TP, H/2] e2m1x2
* s13 rows  = the same rows of the gate / up e8m0 scales                              -> [2*I/TP, H/32]
* w2  cols  = down packed-K bytes r*I/TP/2 .. (r+1)*I/TP/2 (intermediate is w2's K)   -> [H, I/TP/2]
* s2  cols  = down scale blocks r*I/TP/32 .. (r+1)*I/TP/32                            -> [H, I/TP/32]

The slice is taken on the native (checkpoint) layout, then the gfx1250 layout is applied
(``layout_gfx1250``: weight bytes in the (16,16) preshuffle, scales in the gfx1250 WMMA order). An MXFP4 scale
block is 32 k wide and I/TP = 1024 is a multiple of 32, so slicing never splits a block: the sliced weights hold
exactly the bytes of the full expert.

Here H is the pooled expert space (``moe_hidden_size`` 4096) and I = ``moe_intermediate_size`` 4096.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache

import torch

sys.path.insert(0, "/data/muse455/k2/port")
import layout_gfx1250 as L  # noqa: E402

CKPT = "/data/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
H = 4096  # pooled expert hidden
I = 4096  # expert intermediate
E = 256
TP = 4
PREFIX = "model.language_model.layers.{layer}.mlp"


@lru_cache(maxsize=1)
def _index():
    return json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]


class _Reader:
    def __init__(self):
        self.fs = {}

    def get(self, key):
        from safetensors import safe_open

        fn = _index()[key]
        if fn not in self.fs:
            self.fs[fn] = safe_open(f"{CKPT}/{fn}", "pt")
        return self.fs[fn].get_tensor(key)


def slice_expert(gate, up, down, gate_s, up_s, down_s, rank, tp=TP):
    """Native per-expert tensors -> this rank's EP1 slice (native layout)."""
    ip = gate.shape[0] // tp
    w13 = torch.cat([gate.narrow(0, rank * ip, ip), up.narrow(0, rank * ip, ip)])
    s13 = torch.cat([gate_s.narrow(0, rank * ip, ip), up_s.narrow(0, rank * ip, ip)])
    w2 = down.narrow(1, rank * ip // 2, ip // 2)
    s2 = down_s.narrow(1, rank * ip // 32, ip // 32)
    return w13.contiguous(), s13.contiguous(), w2.contiguous(), s2.contiguous()


def load_native_expert(rd, layer, e):
    p = f"{PREFIX.format(layer=layer)}.experts.{e}"
    return (rd.get(f"{p}.gate_proj.weight"), rd.get(f"{p}.up_proj.weight"), rd.get(f"{p}.down_proj.weight"),
            rd.get(f"{p}.gate_proj.weight_scale"), rd.get(f"{p}.up_proj.weight_scale"),
            rd.get(f"{p}.down_proj.weight_scale"))


def load_layer_ep1(layer, rank, experts=None, tp=TP, gfx1250=True, rd=None):
    """Stacked EP1 slice of one layer for one rank: (w13, s13, w2, s2) as uint8 CPU tensors,
    in the gfx1250 layout (``gfx1250=True``) or native."""
    rd = rd or _Reader()
    experts = range(E) if experts is None else experts
    parts = [slice_expert(*load_native_expert(rd, layer, e), rank, tp) for e in experts]
    w13, s13, w2, s2 = (torch.stack([p[i] for p in parts]) for i in range(4))
    if gfx1250:
        w13, s13, w2, s2 = L.gfx1250_weights(w13, s13, w2, s2)
    return w13, s13, w2, s2


def unslice(parts_by_rank):
    """Inverse: per-rank native slices [(w13, s13, w2, s2)] -> full native (gate, up, down, gs, us, ds)."""
    ip = parts_by_rank[0][0].shape[-2] // 2
    gate = torch.cat([p[0][..., :ip, :] for p in parts_by_rank], -2)
    up = torch.cat([p[0][..., ip:, :] for p in parts_by_rank], -2)
    gs = torch.cat([p[1][..., :ip, :] for p in parts_by_rank], -2)
    us = torch.cat([p[1][..., ip:, :] for p in parts_by_rank], -2)
    down = torch.cat([p[2] for p in parts_by_rank], -1)
    ds = torch.cat([p[3] for p in parts_by_rank], -1)
    return gate, up, down, gs, us, ds


def ep1_bytes_per_gpu(n_layers=60, tp=TP):
    """MXFP4 expert bytes one rank holds in the EP1 copy (layers 1..60)."""
    per_expert = 3 * H * I // 2 + 3 * H * I // 32
    return n_layers * E * per_expert // tp


LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def dequant(w, s):
    """Native [.., N, K/2] e2m1x2 + [.., N, K/32] e8m0 -> fp32 [.., N, K]."""
    codes = torch.stack([w & 0xF, w >> 4], -1).reshape(*w.shape[:-1], w.shape[-1] * 2)
    lut = LUT.to(w.device)
    return lut[codes.long()] * torch.exp2(s.float() - 127).repeat_interleave(32, -1)
