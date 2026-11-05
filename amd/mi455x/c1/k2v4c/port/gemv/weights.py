"""Real layer weights for the stage tests (Muse Spark Open MXFP4 checkpoint; the dense tensors are bf16 / fp32)."""

from __future__ import annotations

import json
import os

import torch
from safetensors import safe_open

CKPT = "/data/Muse-Spark-Open-816B-A42B-Quark-MXFP4"
PFX = "model.language_model.layers.{}."
TEMPERATURE = 0.3  # config residual_gate_temperature
ROUTER_MULT = 0.17677669529663687  # config output_multiplier


def _index():
    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def load(layer=31, dev="cuda"):
    """Layer ``layer``'s dense tensors (+ layer ``layer + 1``'s pre-QKV norm for seam A), cached in /tmp."""
    cache = f"/tmp/k2g_weights_l{layer}.pt"
    if os.path.exists(cache):
        return {k: v.to(dev) for k, v in torch.load(cache).items()}
    out = _load(layer, dev)
    torch.save({k: v.cpu() for k, v in out.items()}, cache)
    return out


def _load(layer, dev):
    idx = _index()
    names = {
        "w_o": "self_attn.o_proj.weight",
        "w_router": "mlp.gate.weight",
        "bias": "mlp.gate.e_score_correction_bias",
        "w_pool": "mlp.pre_expert_proj.weight",
        "w_unpool": "mlp.post_expert_proj.weight",
        "w_pre": "mlp.pre_expert_norm.weight",
        "gate_attn": "post_attention_residual_gate.gate",
        "gate_ffn": "post_feedforward_residual_gate.gate",
        "w_fn": "pre_feedforward_layernorm.weight",
        "w_pf": "post_feedforward_layernorm.weight",
    }
    out = {}
    for k, n in names.items():
        key = PFX.format(layer) + n
        with safe_open(os.path.join(CKPT, idx[key]), "pt") as f:
            out[k] = f.get_tensor(key).to(dev)
    key = PFX.format(layer + 1) + "input_layernorm.weight"
    with safe_open(os.path.join(CKPT, idx[key]), "pt") as f:
        out["w_qn"] = f.get_tensor(key).to(dev)
    for g, (a, b) in {"gate_attn": ("alpha_a", "beta_a"), "gate_ffn": ("alpha_f", "beta_f")}.items():
        out[a], out[b] = residual_coefficients(out[g])
    return out


def residual_coefficients(gate, eps=1e-3):
    """``MuseSpark1xNormPreservingResidualConnection._coefficients`` (channel-wise, stream gate on)."""
    w = gate.view(-1).float() / TEMPERATURE
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=eps))
    return alpha.contiguous(), beta.contiguous()
