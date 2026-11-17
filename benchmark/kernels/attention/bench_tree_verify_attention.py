"""Per-step target-verify attention on ROCm (aiter backend shapes, page size 1, bf16 KV).

--mode tree:
  tree     - tree_verify_attention at Q tree nodes
  fallback - the masked tree path: extend_attention_fwd with the FULL custom mask
  chain    - the chain's verify at its block of 16: aiter unified_attention for
             Muse-Glimmer (SWA KV pool), causal extend_attention_fwd for Qwen3.8-27B
--mode chain, at Q chain tokens:
  old      - the chain path above
  unified  - aiter unified_attention (Qwen with SGLANG_USE_AITER_UNIFIED_ATTN=1)
  new      - tree_verify_attention with the causal chain mask

Each method's per-layer calls for every attention layer of the model run in one
CUDA graph, every layer with its own KV pool; L2 and MALL are flushed before each
replay, and only the replay is timed. Reports ms per step over all layers.

Usage (one GPU): python3 bench_tree_verify_attention.py [--mode chain] [--model glimmer qwen]
"""

import argparse
import itertools

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.tree_verify_splitkv import (
    tree_ancestor_bits,
    tree_verify_attention,
)

MODELS = {
    # heads, kv_heads, head_dim, sliding layers, full layers, window, context_len
    "glimmer": (32, 2, 128, 39, 13, 2047, 131072),
    "qwen": (24, 4, 256, 0, 16, 0, 262144),
}
CHAIN_BLOCK = 16
DEVICE = "cuda"


def random_tree(nodes, gen):
    mask = torch.eye(nodes, dtype=torch.bool)
    for j in range(1, nodes):
        mask[j] |= mask[int(torch.randint(max(0, j - 4), j, (1,), generator=gen))]
    return mask


class Batch:
    """One request table and per-layer KV pools shared by every method."""

    def __init__(self, *, model, bs, ctx, max_nodes, gen):
        heads, kv_heads, head_dim, n_swa, n_full, window, context_len = MODELS[model]
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.window, self.context_len = window, context_len
        self.bs, self.ctx = bs, ctx
        self.layer_windows = [window] * n_swa + [0] * n_full
        kv_len = ctx + max_nodes
        num_slots = bs * kv_len + 1
        perm = torch.randperm(num_slots - 1, generator=gen) + 1
        self.req_to_token = torch.zeros(bs + 1, context_len, dtype=torch.int32)
        self.req_to_token[1:, :kv_len] = perm.view(bs, kv_len).to(torch.int32)
        self.req_to_token = self.req_to_token.to(DEVICE)
        self.req_pool_indices = torch.arange(1, bs + 1, device=DEVICE)
        self.seq_lens = torch.full((bs,), ctx, dtype=torch.int64, device=DEVICE)
        # The window pool's own slot order, as the SWA KV pool keeps it.
        self.slot_map = torch.randperm(num_slots, generator=gen).to(DEVICE)
        self.pools = [
            (
                torch.randn(num_slots, kv_heads, head_dim, device=DEVICE).bfloat16(),
                torch.randn(num_slots, kv_heads, head_dim, device=DEVICE).bfloat16(),
            )
            for _ in self.layer_windows
        ]
        self.sm_scale = head_dim**-0.5

    def table(self, cols, window):
        rows = self.req_to_token[self.req_pool_indices, :cols].long()
        return self.slot_map[rows] if window > 0 else rows

    def q(self, nodes):
        return torch.randn(
            self.bs * nodes, self.heads, self.head_dim, device=DEVICE
        ).bfloat16()


def tree_step(batch, nodes, gen, **config):
    masks = torch.stack([random_tree(nodes, gen) for _ in range(batch.bs)])
    depth = masks.sum(-1).view(-1) - 1
    positions = (batch.ctx + depth).to(DEVICE)
    ancestors = tree_ancestor_bits(
        tree_mask=masks.view(-1).to(DEVICE),
        out=torch.empty(batch.bs * nodes, dtype=torch.int64, device=DEVICE),
        num_nodes=nodes,
        num_masked_reqs=batch.bs,
    )
    q, out = batch.q(nodes), batch.q(nodes)

    def layer(i):
        window = batch.layer_windows[i]
        k, v = batch.pools[i]
        tree_verify_attention(
            q=q, k_cache=k, v_cache=v, out=out, req_to_token=batch.req_to_token,
            req_pool_indices=batch.req_pool_indices, seq_lens=batch.seq_lens,
            positions=positions, ancestors=ancestors, sm_scale=batch.sm_scale,
            window=window, slot_map=batch.slot_map if window else None, **config,
        )  # fmt: skip

    return layer, masks


def extend_step(batch, nodes, masks):
    """extend_attention_fwd as the backend's verify fallback (masks given) or the
    Qwen chain (masks None) calls it."""
    bs, ctx = batch.bs, batch.ctx
    q, out = batch.q(nodes), batch.q(nodes)
    k_new = torch.randn(bs * nodes, batch.kv_heads, batch.head_dim, device=DEVICE)
    v_new = torch.randn_like(k_new)
    k_new, v_new = k_new.bfloat16(), v_new.bfloat16()
    qo_indptr = torch.arange(0, (bs + 1) * nodes, nodes, dtype=torch.int32)
    kv_indptr = torch.arange(0, (bs + 1) * ctx, ctx, dtype=torch.int32)
    indices = {w: batch.table(ctx, w).reshape(-1).int() for w in (0, batch.window)}
    custom_mask = mask_indptr = None
    if masks is not None:
        full = torch.ones(bs, nodes, ctx + nodes, dtype=torch.uint8)
        full[:, :, ctx:] = masks
        custom_mask = full.view(-1).to(DEVICE)
        mask_indptr = torch.arange(
            0, (bs + 1) * nodes * (ctx + nodes), nodes * (ctx + nodes)
        ).to(DEVICE)
    zero_offsets = torch.zeros(bs, dtype=torch.int32, device=DEVICE)
    qo_indptr, kv_indptr = qo_indptr.to(DEVICE), kv_indptr.to(DEVICE)

    def layer(i):
        window = batch.layer_windows[i]
        k, v = batch.pools[i]
        extra = {}
        if window:
            extra = dict(sliding_window_size=window - 1, window_kv_offsets=zero_offsets)
        extend_attention_fwd(
            q, k_new, v_new, out, k, v, qo_indptr, kv_indptr, indices[window],
            custom_mask, True, mask_indptr, nodes, 1.0, 1.0, batch.sm_scale, **extra,
        )  # fmt: skip

    return layer


def chain_step(batch, nodes):
    q, out = batch.q(nodes), batch.q(nodes)

    def layer(i):
        window = batch.layer_windows[i]
        k, v = batch.pools[i]
        tree_verify_attention(
            q=q, k_cache=k, v_cache=v, out=out, req_to_token=batch.req_to_token,
            req_pool_indices=batch.req_pool_indices, seq_lens=batch.seq_lens,
            positions=None, ancestors=None, sm_scale=batch.sm_scale, window=window,
            slot_map=batch.slot_map if window else None,
        )  # fmt: skip

    return layer


def unified_chain_step(batch, nodes=CHAIN_BLOCK):
    from aiter.ops.triton.attention.unified_attention import unified_attention

    q, out = batch.q(nodes), batch.q(nodes)
    cu = torch.arange(0, (batch.bs + 1) * nodes, nodes, dtype=torch.int32)
    tables = {
        w: batch.table(batch.context_len, w).int().contiguous()
        for w in (0, batch.window)
    }
    seqused_k = (batch.seq_lens + nodes).int()
    cu = cu.to(DEVICE)

    def layer(i):
        window = batch.layer_windows[i]
        k, v = batch.pools[i]
        unified_attention(
            q=q, k=k.view(-1, 1, batch.kv_heads, batch.head_dim),
            v=v.view(-1, 1, batch.kv_heads, batch.head_dim), out=out,
            cu_seqlens_q=cu, max_seqlen_q=nodes, seqused_k=seqused_k,
            max_seqlen_k=batch.context_len, softmax_scale=batch.sm_scale, causal=True,
            window_size=(window - 1, 0) if window else (-1, -1),
            block_table=tables[window], softcap=0, q_descale=None, k_descale=None,
            v_descale=None,
        )  # fmt: skip

    return layer


def step_ms(layer, num_layers, flush, reps=15):
    """Median GPU time of one graph replay of every layer's call, after a flush."""
    for i in range(num_layers):
        layer(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(num_layers):
            layer(i)
    times = []
    for _ in range(reps):
        flush.zero_()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    del graph
    return sorted(times)[len(times) // 2]


def chain_main(args, gen, flush):
    nodes_per_model = {"glimmer": [16], "qwen": [8, 16]}
    print("model    bs   ctx  Q |    old  unified    new   (ms/step)")
    for model, bs, ctx in itertools.product(args.model, args.bs, args.ctx):
        nodes_list = args.nodes or nodes_per_model[model]
        batch = Batch(model=model, bs=bs, ctx=ctx, max_nodes=max(nodes_list), gen=gen)
        layers = len(batch.layer_windows)
        for nodes in nodes_list:
            unified = step_ms(unified_chain_step(batch, nodes), layers, flush)
            if model == "glimmer":
                old = unified
            else:
                old = step_ms(extend_step(batch, nodes, None), layers, flush)
            new = step_ms(chain_step(batch, nodes), layers, flush)
            print(
                f"{model:8s} {bs:2d} {ctx:5d} {nodes:2d} | {old:6.3f} {unified:8.3f} "
                f"{new:6.3f}",
                flush=True,
            )
        del batch
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tree", "chain"], default="tree")
    parser.add_argument("--model", nargs="+", default=["glimmer", "qwen"])
    parser.add_argument("--bs", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--ctx", type=int, nargs="+", default=[1024, 2048, 8192])
    parser.add_argument("--nodes", type=int, nargs="+", default=None)
    parser.add_argument("--skip-fallback", action="store_true")
    args = parser.parse_args()
    gen = torch.Generator().manual_seed(0)
    flush = torch.empty(1 << 29, dtype=torch.uint8, device=DEVICE)
    if args.mode == "chain":
        chain_main(args, gen, flush)
        return
    args.nodes = args.nodes or [16, 32]
    print("model    bs   ctx | chain Q16 | " + " | ".join(
        f"tree Q{n:<2d} fallback Q{n:<2d}" for n in args.nodes) + "   (ms/step)")  # fmt: skip
    for model, bs, ctx in itertools.product(args.model, args.bs, args.ctx):
        batch = Batch(model=model, bs=bs, ctx=ctx, max_nodes=max(args.nodes), gen=gen)
        layers = len(batch.layer_windows)
        if model == "glimmer":
            chain = unified_chain_step(batch)
        else:
            chain = extend_step(batch, CHAIN_BLOCK, None)
        row = [f"{model:8s} {bs:2d} {ctx:5d} | {step_ms(chain, layers, flush):9.3f}"]
        for nodes in args.nodes:
            tree, masks = tree_step(batch, nodes, gen)
            new = step_ms(tree, layers, flush)
            old = float("nan")
            if not args.skip_fallback:
                old = step_ms(extend_step(batch, nodes, masks), layers, flush)
            row.append(f"{new:12.3f} {old:12.3f}")
        print(" | ".join(row), flush=True)
        del batch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
