"""Retrieval extension of DFLASH trees past the draft block, on the device.

A request's own context (prompt and generated tokens) often repeats long spans, which the
drafter copies well but only up to its block. Retrieval looks up the longest earlier
occurrence of the context's suffix and proposes the tokens that followed it. When the
drafted path (each tree node's first child, from the root down to the block's last depth)
equals those tokens, the tree's last nodes are replaced by a chain of the tokens that
follow, below the path's leaf. Verification treats the chain as drafts whose q is a point
mass, so it stays lossless; the verify width, and so the step's shape, does not change.

Token ids live in a ``[request slots, context + 1]`` history indexed like ``req_to_token``;
its last column is a sink for masked writes.
"""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

# A match is packed as ``length << MATCH_POS_BITS | end position``, so the largest packed
# value is the longest match, the latest one among equals.
MATCH_POS_BITS = 20
MAX_CONTEXT = 1 << MATCH_POS_BITS
# Longest suffix match measured (explore/sysloop/retr.py) and positions searched before the
# request's last token: 8 blocks of 1024 cover every prompt and output the projection used.
MAX_MATCH = 48
SEARCH_BLOCK = 1024
SEARCH_BLOCKS = 8


@triton.jit
def _suffix_match_kernel(
    history,
    req_pool_indices,
    seq_lens,
    best,
    context,
    HIST_STRIDE: tl.constexpr,
    MATCH_LEN: tl.constexpr,
    POS_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """``best[b]``: the longest earlier match of request ``b``'s suffix that ends at its
    last token, position ``seq_lens[b]``; packed (length, end), -1 when none."""
    b = tl.program_id(0)
    block = tl.program_id(1)
    row = history + tl.load(req_pool_indices + b).to(tl.int64) * HIST_STRIDE
    n = tl.minimum(tl.load(seq_lens + b).to(tl.int32), context - 1)
    start = tl.maximum(n - NUM_BLOCKS * BLOCK, 0) + block * BLOCK
    if start < n:
        ends = start + tl.arange(0, BLOCK)
        alive = ends < n
        length = tl.zeros((BLOCK,), dtype=tl.int32)
        for j in range(MATCH_LEN):
            tail = tl.load(row + n - j, mask=n - j >= 0, other=-1)
            alive = alive & (ends - j >= 0) & (n - j >= 0)
            token = tl.load(row + ends - j, mask=alive, other=-2)
            alive = alive & (token == tail)
            length += alive.to(tl.int32)
        packed = tl.where(length > 0, (length << POS_BITS) | ends, -1)
        tl.atomic_max(best + b, tl.max(packed, axis=0))


@triton.jit
def _extend_tree_kernel(
    history,
    req_pool_indices,
    seq_lens,
    best,
    edges_in,
    tokens_in,
    node_count,
    edges_out,
    tokens_out,
    parents_out,
    row_tokens,
    row_probs,
    ext_len,
    context,
    HIST_STRIDE: tl.constexpr,
    EDGE_STRIDE: tl.constexpr,
    PARENT_STRIDE: tl.constexpr,
    POS_BITS: tl.constexpr,
    WIDTH: tl.constexpr,
    K: tl.constexpr,
    PATH: tl.constexpr,
    MAX_EXT: tl.constexpr,
    MIN_MATCH: tl.constexpr,
    BLOCK_W: tl.constexpr,
    HAS_COUNT: tl.constexpr,
    HAS_ROWS: tl.constexpr,
):
    b = tl.program_id(0)
    row = history + tl.load(req_pool_indices + b).to(tl.int64) * HIST_STRIDE
    n = tl.minimum(tl.load(seq_lens + b).to(tl.int32), context - 1)
    packed = tl.load(best + b)
    length = packed >> POS_BITS
    end = packed & ((1 << POS_BITS) - 1)
    # The proposal is history[end + 1 : n + 1]; it must reach past the block.
    ok = (packed >= 0) & (length >= MIN_MATCH) & (n - end > PATH)

    slots = tl.arange(0, BLOCK_W)
    valid = slots < WIDTH - 1
    if HAS_COUNT:
        valid = valid & (slots + 1 < tl.load(node_count + b))
    edges = tl.load(edges_in + b * EDGE_STRIDE + slots, mask=valid, other=0)
    parents = tl.where(valid, edges // K, -1)
    # Walk the first children from the root; every path token must equal the proposal.
    node = tl.zeros((), dtype=tl.int32)
    for d in range(PATH):
        child = tl.min(tl.where(parents == node, slots + 1, WIDTH), axis=0)
        found = child < WIDTH
        token = tl.load(tokens_in + b * EDGE_STRIDE + child - 1, mask=found, other=-1)
        want = tl.load(row + end + 1 + d, mask=ok, other=-2)
        ok = ok & found & (token == want)
        node = tl.where(found, child, node)
    leaf = node
    extend = tl.minimum(tl.minimum(n - end - PATH, WIDTH - 1 - leaf), MAX_EXT)
    extend = tl.where(ok, extend, 0)
    tl.store(ext_len + b, extend)

    # Nodes first..WIDTH-1 become the chain: the first one's parent is the leaf.
    first = WIDTH - extend
    steps = tl.arange(0, BLOCK_W)
    in_chain = steps < extend
    nodes = first + steps
    edge = tl.where(steps == 0, leaf, nodes - 1) * K
    chain = tl.load(row + end + 1 + PATH + steps, mask=in_chain, other=0)
    tl.store(edges_out + b * EDGE_STRIDE + nodes - 1, edge, mask=in_chain)
    tl.store(tokens_out + b * EDGE_STRIDE + nodes - 1, chain, mask=in_chain)
    tl.store(parents_out + b * PARENT_STRIDE + nodes, edge, mask=in_chain)
    if HAS_ROWS:
        # Each chain parent's children: its one chain token with q = 1, put at rank 0 of
        # the leaf's candidates so that a row never lists a token twice.
        ranks = tl.arange(0, K)
        base = tl.load(row_tokens + (b * WIDTH + leaf) * K + ranks)
        lead = tl.sum(tl.where(ranks == 0, base, 0), axis=0)
        for i in range(0, extend):
            parent = tl.where(i == 0, leaf, first + i - 1)
            token = tl.load(row + end + 1 + PATH + i)
            tokens = tl.where(ranks == 0, token, tl.where(base == token, lead, base))
            out = (b * WIDTH + parent) * K + ranks
            tl.store(row_tokens + out, tokens)
            tl.store(row_probs + out, tl.where(ranks == 0, 1.0, 0.0))
        if extend > 0:
            out = (b * WIDTH + WIDTH - 1) * K + ranks
            tl.store(row_tokens + out, base)
            tl.store(row_probs + out, tl.zeros((K,), dtype=tl.float32))


def history_context(history: torch.Tensor) -> int:
    """Positions a history row holds; the column after them is the sink."""
    return history.shape[1] - 1


class ExtendedTree(msgspec.Struct, frozen=True):
    """A verify tree with its last ``ext_len[b]`` nodes replaced by the retrieval chain,
    in ``DFlashTreeProposal``'s ABI at ``depth``; ``node_draft_*`` as the verifier reads
    them (None for greedy trees)."""

    draft_tokens: torch.Tensor
    parent_list: torch.Tensor
    top_scores_index: torch.Tensor
    depth: int
    ext_len: torch.Tensor
    node_draft_tokens: torch.Tensor | None
    node_draft_probs: torch.Tensor | None


def find_suffix_matches(
    *,
    history: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """``[bs]`` int32: each request's longest earlier match of its context's suffix,
    the context ending at position ``seq_lens`` (the bonus token), packed as
    ``length << MATCH_POS_BITS | end``; -1 when the last token never occurred before."""
    bs = req_pool_indices.shape[0]
    best = torch.full((bs,), -1, dtype=torch.int32, device=history.device)
    if bs == 0:
        return best
    _suffix_match_kernel[(bs, SEARCH_BLOCKS)](
        history,
        req_pool_indices,
        seq_lens,
        best,
        history_context(history),
        HIST_STRIDE=history.stride(0),
        MATCH_LEN=MAX_MATCH,
        POS_BITS=MATCH_POS_BITS,
        BLOCK=SEARCH_BLOCK,
        NUM_BLOCKS=SEARCH_BLOCKS,
    )
    return best


def extend_tree_with_retrieval(
    *,
    history: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    matches: torch.Tensor,
    draft_tokens: torch.Tensor,
    parent_list: torch.Tensor,
    top_scores_index: torch.Tensor,
    node_count: torch.Tensor | None,
    node_draft_tokens: torch.Tensor | None,
    node_draft_probs: torch.Tensor | None,
    width: int,
    topk: int,
    path_depth: int,
    max_extension: int,
    min_match: int,
) -> ExtendedTree:
    """The tree of ``width`` nodes (best-first, so its first nodes are a tree too) with the
    retrieval chain in its last nodes where the gate holds: the match is at least
    ``min_match`` long, the first-child path from the root reaches ``path_depth`` within
    the tree's real nodes, and its tokens equal the proposal's first ``path_depth``. The
    chain is ``min(max_extension, proposal tokens left, width - 1 - leaf)`` long."""
    bs = draft_tokens.shape[0]
    depth = min(width - 1, path_depth + max_extension)
    edges = top_scores_index[:, : width - 1].contiguous().clone()
    tokens = draft_tokens[:, : width - 1].contiguous().clone()
    parents = torch.full(
        (bs, topk * (depth - 1) + 1), -1, dtype=parent_list.dtype, device=edges.device
    )
    keep = min(parent_list.shape[1], parents.shape[1])
    parents[:, :keep] = parent_list[:, :keep]
    has_rows = node_draft_tokens is not None
    if has_rows:
        node_draft_tokens = node_draft_tokens[:, :width].contiguous().clone()
        node_draft_probs = node_draft_probs[:, :width].float().contiguous().clone()
    ext_len = torch.zeros((bs,), dtype=torch.int32, device=edges.device)
    if bs > 0:
        _extend_tree_kernel[(bs,)](
            history,
            req_pool_indices,
            seq_lens,
            matches,
            edges,
            tokens,
            node_count if node_count is not None else ext_len,
            edges,
            tokens,
            parents,
            node_draft_tokens if has_rows else tokens,
            node_draft_probs if has_rows else tokens,
            ext_len,
            history_context(history),
            HIST_STRIDE=history.stride(0),
            EDGE_STRIDE=edges.stride(0),
            PARENT_STRIDE=parents.stride(0),
            POS_BITS=MATCH_POS_BITS,
            WIDTH=width,
            K=topk,
            PATH=path_depth,
            MAX_EXT=max_extension,
            MIN_MATCH=min_match,
            BLOCK_W=triton.next_power_of_2(width),
            HAS_COUNT=node_count is not None,
            HAS_ROWS=has_rows,
        )
    return ExtendedTree(
        draft_tokens=tokens,
        parent_list=parents,
        top_scores_index=edges,
        depth=depth,
        ext_len=ext_len,
        node_draft_tokens=node_draft_tokens,
        node_draft_probs=node_draft_probs,
    )


def suffix_match_reference(
    history: list[int], n: int, *, max_match: int = MAX_MATCH
) -> tuple[int, int]:
    """(length, end) of the longest match of ``history[: n + 1]``'s suffix ending before
    ``n`` (the latest among equals, within the searched window), or (0, -1)."""
    best = (0, -1)
    for end in range(max(n - SEARCH_BLOCKS * SEARCH_BLOCK, 0), n):
        length = 0
        while (
            length < max_match
            and end - length >= 0
            and history[end - length] == history[n - length]
        ):
            length += 1
        if length > 0 and (length, end) > best:
            best = (length, end)
    return best


def extend_tree_reference(
    *,
    proposal: list[int],
    parents: list[int],
    tokens: list[int],
    num_real: int,
    width: int,
    path_depth: int,
    max_extension: int,
    match_len: int,
    min_match: int,
) -> tuple[list[int], list[int], int, int]:
    """The rule ``_extend_tree_kernel`` implements, for one request: ``parents`` and
    ``tokens`` of nodes 1..width-1 and the proposal after the match. Returns the new
    parents and tokens, the chain length and the path's leaf (-1 when the gate fails)."""
    parents, tokens = list(parents), list(tokens)
    if match_len < min_match or len(proposal) <= path_depth:
        return parents, tokens, 0, -1
    node = 0
    for d in range(path_depth):
        children = [i for i in range(1, num_real) if parents[i - 1] == node]
        if not children or tokens[children[0] - 1] != proposal[d]:
            return parents, tokens, 0, -1
        node = children[0]
    extend = min(max_extension, len(proposal) - path_depth, width - 1 - node)
    first = width - extend
    for i in range(extend):
        parents[first + i - 1] = node if i == 0 else first + i - 1
        tokens[first + i - 1] = proposal[path_depth + i]
    return parents, tokens, extend, node
