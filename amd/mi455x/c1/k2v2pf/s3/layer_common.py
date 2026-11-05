"""Shared fixtures of the k2lm harnesses (copied from k2v3/s3/layer_ab.py): checkpoint reads, the served QKV weight
of a rank, the RoPE table, K1's per-layer / per-step operands (``Tail``) and the served Triton ops."""

import json
import os

import torch

CKPT = "/data/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
ROWS = 8
HQ, HK, D = 32, 4, 64
NQKV = 2 * (HQ + HK) * D  # 4608
EPS_POST_FFN = 1e-8  # config.post_norm_eps (post_feedforward_layernorm)
EPS_RMS = 1e-5  # config.rms_norm_eps (input_layernorm of L + 1, QK norm)
ROPE_MAX = 393216  # cos_sin rows built (positions stay below; the ~340K-context cases need > 262144)
SLOTS = 262144  # full KV pool slots
SWA_SLOTS = 65536


def config():
    with open(os.path.join(CKPT, "config.json")) as f:
        c = json.load(f)
    return c.get("text_config", c)


def ckpt_rows(name, start, length):
    from safetensors import safe_open

    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        fn = json.load(f)["weight_map"][name]
    with safe_open(os.path.join(CKPT, fn), "pt") as f:
        return f.get_slice(name)[start:start + length]


def qkv_weight(layer, rank, dev):
    """The served ``self_attn.qkv_proj.weight`` of ``layer`` on ``rank`` (QKVOGateParallelLinear.weight_loader:
    [q rank shard, k rank shard, v rank shard, gate rank shard], BF16 [4608, 8192])."""
    cache = f"/tmp/k2tail_wqkv_l{layer}_r{rank}.pt"
    if os.path.exists(cache):
        return torch.load(cache).to(dev)
    p = f"model.language_model.layers.{layer}.self_attn."
    parts = [ckpt_rows(p + "q_proj.weight", rank * HQ * D, HQ * D), ckpt_rows(p + "k_proj.weight", rank * HK * D, HK * D),
             ckpt_rows(p + "v_proj.weight", rank * HK * D, HK * D), ckpt_rows(p + "gate_proj.weight", rank * HQ * D, HQ * D)]
    w = torch.cat([x.to(torch.bfloat16) for x in parts], 0).contiguous()
    assert w.shape == (NQKV, 8192)
    torch.save(w, cache)
    return w.to(dev)


def cos_sin_cache(base, dev):
    """sglang RotaryEmbedding._compute_cos_sin_cache (rotary_dim 64, fp32, position_start 0)."""
    inv = 1.0 / (base ** (torch.arange(0, D, 2, dtype=torch.float, device=dev) / D))
    t = torch.arange(0, ROPE_MAX, dtype=torch.float, device=dev)
    f = torch.einsum("i,j -> ij", t, inv)
    return torch.cat((f.cos(), f.sin()), dim=-1).contiguous()


class Tail:
    """The tail's operands: per layer (boundary weights, L + 1's QKV) and per step (positions .. caches)."""

    def __init__(self, lay, layer, rank, dev, cfg, seed, varied, pos_base=131000):
        nxt = layer + 1
        w = lay.w
        bf = torch.bfloat16
        # served _ensure_cached_weights: (weight + gain_center 1.0) in the parameter dtype (bf16), alpha / beta fp32
        self.w_post = (w["w_pf"].to(bf) + 1.0).contiguous()
        self.w_pre = (w["w_qn"].to(bf) + 1.0).contiguous()
        self.alpha = w["alpha_f"].reshape(-1).float().contiguous()
        self.beta = w["beta_f"].reshape(-1).float().contiguous()
        self.w_qkv = qkv_weight(nxt, rank, dev)
        theta = float(cfg["layer_rope_theta"][nxt])
        self.rope = theta != 0.0
        self.sliding = cfg["layer_types"][nxt] == "sliding_attention"
        self.scale_q = float(cfg["qk_scale_factor"])
        self.cos_sin = cos_sin_cache(int(theta), dev) if self.rope else None
        g = torch.Generator().manual_seed(1000 + seed)
        base = pos_base if not varied else int(torch.randint(2048, ROPE_MAX - 64, (1,), generator=g))
        self.positions = (base + torch.arange(ROWS)).to(torch.int64).to(dev)
        loc = torch.randperm(SLOTS, generator=g)[:ROWS].to(torch.int64)
        if varied and seed % 3 == 2:
            loc[3] = -1  # a padded row: no cache store
        self.locations = loc.to(dev)
        if self.sliding:
            m = torch.randint(0, SWA_SLOTS, (SLOTS + 16 + 1,), generator=g, dtype=torch.int64)
            m[-1] = -1
            if varied and seed % 3 == 1:
                m[int(loc[5])] = -1  # an evicted SWA slot: no cache store
            self.mapping = m.to(dev)
            self.cache_slots = SWA_SLOTS
        else:
            self.mapping = None
            self.cache_slots = SLOTS
        ks, vs = (1.0, 1.0) if not varied else (float(0.25 + torch.rand(1, generator=g)), float(0.5 + torch.rand(1, generator=g)))
        self.k_scale = torch.tensor([ks], dtype=torch.float32, device=dev)
        self.v_scale = torch.tensor([vs], dtype=torch.float32, device=dev)
        self.cache_init = torch.randint(0, 256, (2, self.cache_slots, HK, D), generator=g, dtype=torch.uint8).to(dev)

    def new_caches(self):
        c = self.cache_init.clone()
        return c[0].view(torch.float8_e4m3fn), c[1].view(torch.float8_e4m3fn)


def served_ops():
    from sglang.kernels.ops.attention.qk_norm_rope_cache import qkv_proj_qk_norm_rope_cache, projection_tile
    from sglang.srt.rocm_muse_spark.fused_norm_residual import (
        _fused_norm_residual_config, _fused_norm_residual_triton_kernel)

    assert projection_tile(ROWS, HQ) == (8, 64, 16, 4), projection_tile(ROWS, HQ)
    nw, ns, extra = _fused_norm_residual_config(8192, 0, ROWS, True)
    return qkv_proj_qk_norm_rope_cache, _fused_norm_residual_triton_kernel, (nw, ns, extra)


