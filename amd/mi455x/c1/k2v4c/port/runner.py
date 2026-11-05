"""Host side of the gfx1250 K2 (lead): per-rank peer runtime, per-layer weight bundles, launch.

    runner = K2Runner(group_cpu, rank, device)              # collective over the TP group (IPC peer buffers)
    runner.add_layer(layer_idx, decoder_layer_module)      # dense views + the EP1 expert copy (loaded here)
    runner.begin_step()                                     # once a step, inside the graph: advance the epoch
    stream_out, branch = runner.run(layer_idx, core, stream) # one K2 launch

The EP1 expert copy (each rank: all 256 experts, intermediate quarter ``rank``) is read from the checkpoint with
safetensors slices (only this rank's bytes) and permuted to the gfx1250 scale order once. It sits beside the served
EP4 experts, so every step K2 does not take (prefill, other widths, layers 0 / 61) runs the served path unchanged.
"""

import json
import os

import torch

from port import consts as K
from port import k2
from port.sync.epoch import advance_epoch_
from port.sync.peer import PeerRuntime

CKPT = os.environ.get("SGLANG_MUSE_MONO_GFX1250_CKPT", "/data/Muse-Spark-Open-816B-A42B-Quark-MXFP4")
PREFIX = "model.language_model.layers.{}.mlp.experts.{}."
BF16_EXPERT_LAYERS = (0, 61)


# ---------------------------------------------------------------- EP1 expert copy
class _Ckpt:
    def __init__(self, root=CKPT):
        from safetensors import safe_open

        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self._open = safe_open
        self._files = {}

    def slice(self, name, dim, start, length):
        fn = self.map[name]
        if fn not in self._files:
            self._files[fn] = self._open(os.path.join(self.root, fn), "pt")
        sl = self._files[fn].get_slice(name)
        idx = [slice(None)] * len(sl.get_shape())
        idx[dim] = slice(start, start + length)
        return sl[tuple(idx)]


def load_ep1_experts(layer, rank, device, ckpt=None):
    """(w13, s13, w2, s2) of ``layer`` for ``rank``: [256, 2048, 2048] u8, [256, 2048, 128] u8, [256, 4096, 512] u8,
    [256, 4096, 32] u8, in the gfx1250 K2 layout (weights (16,16)-shuffled, scales [E, N/16, K/256, 2, 16, 4])."""
    from port import layout as L

    ck = ckpt or _Ckpt()
    ri = K.RI
    w13 = torch.empty(K.E, 2 * ri, K.MH // 2, dtype=torch.uint8)
    s13 = torch.empty(K.E, 2 * ri, K.MH // 32, dtype=torch.uint8)
    w2 = torch.empty(K.E, K.MH, ri // 2, dtype=torch.uint8)
    s2 = torch.empty(K.E, K.MH, ri // 32, dtype=torch.uint8)
    for e in range(K.E):
        p = PREFIX.format(layer, e)
        w13[e, :ri] = ck.slice(p + "gate_proj.weight", 0, rank * ri, ri)
        w13[e, ri:] = ck.slice(p + "up_proj.weight", 0, rank * ri, ri)
        s13[e, :ri] = ck.slice(p + "gate_proj.weight_scale", 0, rank * ri, ri)
        s13[e, ri:] = ck.slice(p + "up_proj.weight_scale", 0, rank * ri, ri)
        w2[e] = ck.slice(p + "down_proj.weight", 1, rank * ri // 2, ri // 2)
        s2[e] = ck.slice(p + "down_proj.weight_scale", 1, rank * ri // 32, ri // 32)
    out = L.gfx1250_weights(w13, s13, w2, s2)
    return tuple(t.to(device).contiguous() for t in out)


# ---------------------------------------------------------------- one layer's K2 operands
class K2Layer:
    def __init__(self, layer_idx, mod, rank, device, ckpt=None):
        """``mod``: the served ``MuseSpark1xDecoderLayer`` (weights loaded)."""
        from port.gemv import gemv

        shuf = gemv.shuffle_rows if gemv.SHUFFLED else (lambda t: t)
        attn, moe = mod.self_attn, mod.mlp
        gate_a = mod.post_attention_residual_gate
        self.layer = layer_idx
        w = {}
        w["w_o"] = shuf(attn.o_proj.weight.data.contiguous())  # [8192, 2048] this rank's heads
        alpha, beta = _residual_coefficients(gate_a.gate)
        w["alpha_a"], w["beta_a"] = alpha, beta
        w["w_fn"] = mod.pre_feedforward_layernorm.weight.data.contiguous()
        w["w_router"] = moe.gate.weight.data.float().contiguous()
        w["expert_bias"] = moe.gate.e_score_correction_bias.data.float().contiguous()
        w["w_pool"] = shuf(moe.pre_expert_proj.weight.data.contiguous())
        w["w_pre"] = moe.pre_expert_norm.weight.data.contiguous()
        w["w_post"] = moe.post_expert_norm.weight.data.contiguous()
        w["w_unpool"] = shuf(moe.post_expert_proj.weight.data.contiguous())
        w["w13"], w["w13s"], w["w2"], w["w2s"] = load_ep1_experts(layer_idx, rank, device, ckpt)
        self.w = w


def _residual_coefficients(gate, temp=K.RESIDUAL_GATE_TEMP, eps=1e-3):
    g = gate.data.view(-1).float() / temp
    beta = torch.sigmoid(g)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-g) * (1.0 + beta), min=eps))
    return alpha.contiguous(), beta.contiguous()


# ---------------------------------------------------------------- the runner
class K2Runner:
    def __init__(self, group_cpu, rank, device, key=None):
        self.key = key or k2.K2Build()
        self.rank, self.device = rank, device
        self.rt = PeerRuntime(k2.peer_bytes(), group_cpu, rank, K.TP, device, blocks=K.BLOCKS)
        self.scratch = torch.zeros(k2.scratch_bytes() // 4, dtype=torch.int32, device=device)
        self.layers = {}
        self.ckpt = _Ckpt()
        self.stream_out = torch.empty(K.MAX_TOKENS, K.H, dtype=torch.float32, device=device)
        self.branch = torch.empty(K.MAX_TOKENS, K.H, dtype=torch.bfloat16, device=device)
        k2.build(self.key)

    def add_layer(self, layer_idx, mod):
        if layer_idx in BF16_EXPERT_LAYERS:
            return
        self.layers[layer_idx] = K2Layer(layer_idx, mod, self.rank, self.device, self.ckpt)

    def takes_step(self, rows):
        return bool(self.layers) and k2.step_disable_reason(rows) is None

    def takes(self, layer_idx, rows):
        return layer_idx in self.layers and k2.step_disable_reason(rows) is None

    def begin_step(self):
        advance_epoch_(self.rt.epoch)

    def run(self, layer_idx, core, stream, stream_out=None, branch=None):
        """core [8, 2048] bf16 (pre-o_proj attention output), stream [8, 8192] fp32 -> (stream_out, branch)."""
        lw = self.layers[layer_idx].w
        so = self.stream_out if stream_out is None else stream_out
        br = self.branch if branch is None else branch
        t = dict(lw, core=core.contiguous(), stream_a=stream.contiguous(), stream_out=so, branch_out=br)
        k2.k2_launch(self.key, t, self.rank, layer_idx, self.rt.epoch, self.rt.peers, self.scratch,
                     self.rt.health, self.rt.alarm, torch.cuda.current_stream().cuda_stream)
        return so, br

    def check(self):
        self.rt.check()

    def close(self):
        self.rt.close()
