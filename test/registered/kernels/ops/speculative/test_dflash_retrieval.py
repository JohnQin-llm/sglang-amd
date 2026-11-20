"""DFLASH retrieval extension: suffix matches over the token history, and verify trees whose
last nodes become the retrieval chain where the drafted path agrees with it."""

import math
import os
import sys

import pytest
import torch
from sglang.kernels.ops.speculative.dflash_retrieval import (
    MATCH_POS_BITS,
    extend_tree_reference,
    extend_tree_with_retrieval,
    find_suffix_matches,
    suffix_match_reference,
)
from sglang.kernels.ops.speculative.dflash_tree import (
    build_dflash_sampled_tree_proposal,
    build_dflash_tree_proposal,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
needs_triton = pytest.mark.skipif(
    DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
    reason="Triton kernels need a GPU or TRITON_INTERPRET=1",
)
needs_gpu = pytest.mark.skipif(DEVICE == "cpu", reason="EAGLE tree kernels need a GPU")


def _history(seqs, context):
    history = torch.full((len(seqs), context + 1), -1, dtype=torch.int32)
    for row, seq in enumerate(seqs):
        history[row, : len(seq)] = torch.tensor(seq, dtype=torch.int32)
    return history.to(DEVICE)


def _unpack(packed):
    return (packed >> MATCH_POS_BITS, packed & ((1 << MATCH_POS_BITS) - 1))


@needs_triton
def test_suffix_match_is_the_longest_then_latest():
    """Packed (length, end) maxima pick the longest match and, among equals, the latest,
    capped at MAX_MATCH; a last token that never occurred before has none."""
    gen = torch.Generator().manual_seed(1)
    seqs = []
    for row in range(12):
        base = torch.randint(0, 6 if row % 2 else 50, (300,), generator=gen).tolist()
        span = base[40:110]
        # A long copy (longer than MAX_MATCH on some rows) ends the context.
        seqs.append(base + span[: 20 + 6 * row])
    seqs.append(list(range(100)))  # every token new: no match
    seqs.append([7] * 80)  # a constant run: the latest end before the last token
    seq_lens = torch.tensor([len(s) - 1 for s in seqs], dtype=torch.int64)
    history = _history(seqs, context=512)
    packed = find_suffix_matches(
        history=history,
        req_pool_indices=torch.arange(len(seqs), device=DEVICE),
        seq_lens=seq_lens.to(DEVICE),
    ).cpu()
    for row, seq in enumerate(seqs):
        length, end = suffix_match_reference(seq, len(seq) - 1)
        if length == 0:
            assert packed[row].item() == -1, f"row {row}"
        else:
            assert _unpack(packed[row].item()) == (length, end), f"row {row}"
    assert packed[-2].item() == -1
    assert _unpack(packed[-1].item()) == (48, 78)


def _first_child_path(edges, k, path_depth, num_real):
    parents = [e // k for e in edges]
    node, path = 0, []
    for _ in range(path_depth):
        kids = [i for i in range(1, num_real) if parents[i - 1] == node]
        if not kids:
            return None
        node = kids[0]
        path.append(node)
    return path


def _planted(tokens, path, chain_len, agree, gen, vocab):
    """A context whose suffix occurred before, followed by the path tokens (changed at
    one depth unless ``agree``) and ``chain_len`` more tokens: (sequence, proposal)."""
    filler = lambda n: (vocab + torch.randint(0, 10**6, (n,), generator=gen)).tolist()  # noqa: E731
    suffix = filler(6)
    want = [tokens[v - 1] for v in path]
    if not agree:
        want[int(torch.randint(0, len(want), (1,), generator=gen))] += 1
    head = filler(30) + suffix
    seq = head + want + filler(chain_len) + filler(20) + suffix
    return seq, seq[len(head) :]


@needs_triton
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("width, max_extension", [(20, 12), (20, 3), (9, 12)])
def test_extended_tree_follows_the_gate(sampled, width, max_extension):
    """Kernel = extend_tree_reference: the chain replaces the last nodes only where the
    whole drafted path agrees; parent_list keeps EAGLE's ABI (a node's entry is its own
    edge); a chain parent's row is a point mass on its child, rank 0, with no token
    listed twice, and the chain's last node has no q."""
    rows, num_depths, k, vocab = 24, 5, 8, 64
    gen = torch.Generator().manual_seed(width * 100 + max_extension + sampled)
    candidate_ids = torch.stack(
        [torch.randperm(vocab, generator=gen)[: num_depths * k] for _ in range(rows)]
    ).view(rows, num_depths, k)
    # Peaked q, as on copy spans: first draws reach the last depth within the width.
    scores = torch.randn(rows, num_depths, k, k, generator=gen) * 4
    scores[:, 0] = scores[:, 0, :1]
    log_q = torch.log_softmax(scores, dim=-1).to(DEVICE)
    candidate_ids = candidate_ids.to(DEVICE)
    node_tokens = node_probs = None
    if sampled:
        proposal, node_tokens, node_probs = build_dflash_sampled_tree_proposal(
            candidate_ids, log_q, max_nodes=width
        )
    else:
        proposal = build_dflash_tree_proposal(candidate_ids, log_q, max_nodes=width)
    edges = proposal.top_scores_index.cpu().tolist()
    tokens = proposal.draft_tokens.cpu().tolist()
    seqs, proposals, matches = [], [], []
    for row in range(rows):
        path = _first_child_path(edges[row], k, num_depths, width)
        agree = row % 3 != 0
        if path is None:
            path = [1] * num_depths
        seq, prop = _planted(tokens[row], path, row % 7, agree, gen, vocab)
        seqs.append(seq)
        proposals.append(prop)
        matches.append(row % 5 != 4)  # every fifth row: match too short
    context = max(len(s) for s in seqs) + 8
    history = _history(seqs, context)
    seq_lens = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
    req = torch.arange(rows, device=DEVICE)
    packed = find_suffix_matches(
        history=history, req_pool_indices=req, seq_lens=seq_lens
    )
    min_match = torch.where(
        torch.tensor(matches, device=DEVICE), 4, 7
    )  # the planted suffix is 6 long
    out = {}
    for need in (4, 7):
        out[need] = extend_tree_with_retrieval(
            history=history,
            req_pool_indices=req,
            seq_lens=seq_lens,
            matches=packed,
            draft_tokens=proposal.draft_tokens,
            parent_list=proposal.parent_list,
            top_scores_index=proposal.top_scores_index,
            node_count=proposal.node_count,
            node_draft_tokens=node_tokens,
            node_draft_probs=node_probs,
            width=width,
            topk=k,
            path_depth=num_depths,
            max_extension=max_extension,
            min_match=need,
        )
    fired = 0
    for row in range(rows):
        need = int(min_match[row])
        ext = out[need]
        length, _ = suffix_match_reference(seqs[row], len(seqs[row]) - 1)
        assert length == 6
        want_parents, want_tokens, want_len, leaf = extend_tree_reference(
            proposal=proposals[row],
            parents=[e // k for e in edges[row]],
            tokens=tokens[row],
            num_real=width,
            width=width,
            path_depth=num_depths,
            max_extension=max_extension,
            match_len=length,
            min_match=need,
        )
        got_len = int(ext.ext_len[row])
        assert got_len == want_len, f"row {row}"
        got_edges = ext.top_scores_index[row].cpu().tolist()
        assert [e // k for e in got_edges] == want_parents, f"row {row}"
        assert ext.draft_tokens[row].cpu().tolist() == want_tokens, f"row {row}"
        parent_list = ext.parent_list[row].cpu().tolist()
        assert len(parent_list) == k * (ext.depth - 1) + 1
        first = width - got_len
        for node in range(1, width):
            if node < first:
                assert got_edges[node - 1] == edges[row][node - 1]
            else:
                assert got_edges[node - 1] % k == 0
                assert parent_list[node] == got_edges[node - 1]
        fired += got_len > 0
        if got_len == 0 or not sampled:
            continue
        row_tokens = ext.node_draft_tokens[row].cpu()
        row_probs = ext.node_draft_probs[row].cpu()
        chain_parents = [leaf] + list(range(first, width - 1))
        for i, parent in enumerate(chain_parents):
            assert row_tokens[parent, 0].item() == want_tokens[first + i - 1]
            assert row_probs[parent].tolist() == [1.0] + [0.0] * (k - 1)
            assert len(set(row_tokens[parent].tolist())) == k
        assert row_probs[width - 1].abs().sum().item() == 0
    # Rows agree two times in three; a 9-node tree often has its leaf last.
    assert fired >= (4 if width == 20 else 1)


@needs_triton
def test_history_holds_prompt_and_committed_tokens():
    """Prefill tokens, each step's bonus token and its committed tokens (the tree's
    compacted verify rows, of which only the first accept_lens count) land at their
    positions; the rest goes to the sink column, so a row reads prompt + output."""
    from sglang.srt.speculative.dflash_retrieval_extension import RetrievalExtension

    gen = torch.Generator().manual_seed(3)
    context, width, slots = 64, 6, 4
    req_to_token = torch.zeros(slots, context, dtype=torch.int32, device=DEVICE)
    retrieval = RetrievalExtension(
        max_extension=4, min_match=2, path_depth=3, topk=4, device=torch.device(DEVICE)
    )
    req = torch.tensor([2, 0], device=DEVICE)
    prompts = [torch.randint(0, 50, (n,), generator=gen).tolist() for n in (7, 3)]
    retrieval.record_extend(
        req_to_token=req_to_token,
        req_pool_indices=req,
        input_ids=torch.tensor(prompts[0] + prompts[1], device=DEVICE),
        positions=torch.tensor(list(range(7)) + list(range(3)), device=DEVICE),
        extend_lens=torch.tensor([7, 3], dtype=torch.int32, device=DEVICE),
        num_tokens=10,
    )
    # A radix-cached prefix comes from the host.
    retrieval.record_cached_prefixes(
        req_to_token=req_to_token, rows=[3], prefixes=[[11, 12, 13]]
    )
    streams = [list(p) for p in prompts]
    bonus = [int(torch.randint(0, 50, (1,), generator=gen)) for _ in streams]
    for _ in range(12):
        seq_lens = torch.tensor([len(s) for s in streams], device=DEVICE)
        retrieval.match(
            req_to_token=req_to_token,
            req_pool_indices=req,
            seq_lens=seq_lens,
            bonus_tokens=torch.tensor(bonus, device=DEVICE),
        )
        tokens = torch.randint(0, 50, (2, width), generator=gen)
        accept = torch.randint(1, width + 1, (2,), generator=gen)
        retrieval.record_commit(
            req_to_token=req_to_token,
            req_pool_indices=req,
            seq_lens=seq_lens,
            tokens=tokens.to(DEVICE),
            accept_lens=accept.to(torch.int32).to(DEVICE),
        )
        for i in range(2):
            streams[i] += [bonus[i]] + tokens[i, : accept[i] - 1].tolist()
            bonus[i] = int(tokens[i, accept[i] - 1])
    history = retrieval._history.cpu()
    for i, slot in enumerate((2, 0)):
        stream = streams[i] + [bonus[i]]
        n = min(len(stream), context)
        assert history[slot, :n].tolist() == stream[:n]
    assert (history[1, :context] == -1).all()
    assert history[3, :4].tolist() == [11, 12, 13, -1]


@pytest.mark.parametrize("topk, extension", [(16, 0), (1, 31), (16, 4), (16, 31)])
def test_decode_reserve_covers_the_longest_commit(monkeypatch, topk, extension):
    """Regression: DFLASH's decode KV reserve assumed a step commits at most the draft
    block F, but a step that accepts into the retrieval chain commits up to F + L. The
    next step's verify window then ran past the allocated req_to_token slots, whose
    stale ids pointed into radix-cached prefixes, and the verify overwrote their KV
    (served T=1 lossless failure). The reserve's commit term must cover the longest
    path an extended tree of any verify width can commit."""
    from types import SimpleNamespace

    from sglang.srt.speculative import dflash_retrieval_extension as ext

    block, num_nodes = 16, 32
    monkeypatch.setenv("SGLANG_DFLASH_RETRIEVAL_EXTENSION", str(extension))
    monkeypatch.setattr(
        ext, "get_spec", lambda: SimpleNamespace(speculative_eagle_topk=topk)
    )
    commit = ext.max_step_commit(block_size=block, num_nodes=num_nodes)
    assert block <= commit <= num_nodes
    if extension == 0 or topk == 1:
        assert commit == block  # unchanged without the extension
        return
    retrieval = ext.RetrievalExtension(
        max_extension=extension,
        min_match=4,
        path_depth=block - 1,
        topk=topk,
        device=torch.device("cpu"),
    )
    for width in range(2, num_nodes + 1):
        assert retrieval.path_width(width) <= commit


@needs_gpu
def test_extended_tree_verifies_through_the_chain():
    """EAGLE's tree build places the chain below the leaf (positions one deeper per node),
    and a greedy verify whose target follows the path and the proposal accepts all of it;
    Traversal Verification on extended sampled trees equals its float64 reference."""
    from sglang.kernels.ops.speculative.spec_tree import verify_tree_greedy_func
    from sglang.kernels.ops.speculative.traversal_verify import (
        tree_speculative_sampling_traversal,
        tree_speculative_sampling_traversal_reference,
    )
    from sglang.srt.speculative.eagle_utils import (
        TreeMaskMode,
        build_tree_kernel_efficient,
    )

    rows, num_depths, k, vocab, width, max_extension = 32, 5, 4, 97, 16, 8
    gen = torch.Generator().manual_seed(5)
    candidate_ids = torch.stack(
        [torch.randperm(vocab, generator=gen)[: num_depths * k] for _ in range(rows)]
    ).view(rows, num_depths, k)
    # Peaked q, as on copy spans, so that most first-draw paths reach the last depth.
    scores = torch.randn(rows, num_depths, k, k, generator=gen) * 4
    scores[:, 0] = scores[:, 0, :1]
    log_q = torch.log_softmax(scores, dim=-1).cuda()
    proposal, node_tokens, node_probs = build_dflash_sampled_tree_proposal(
        candidate_ids.cuda(), log_q, max_nodes=width
    )
    edges = proposal.top_scores_index.cpu().tolist()
    tokens = proposal.draft_tokens.cpu().tolist()
    seqs, paths = [], []
    for row in range(rows):
        path = _first_child_path(edges[row], k, num_depths, width)
        seq, _ = _planted(tokens[row], path or [1] * num_depths, 12, True, gen, vocab)
        seqs.append([t % vocab for t in seq])
        paths.append(path or [])
    history = _history(seqs, max(len(s) for s in seqs) + 8)
    seq_lens = torch.tensor([len(s) - 1 for s in seqs], device="cuda")
    req = torch.arange(rows, device="cuda")
    packed = find_suffix_matches(
        history=history, req_pool_indices=req, seq_lens=seq_lens
    )
    ext = extend_tree_with_retrieval(
        history=history,
        req_pool_indices=req,
        seq_lens=seq_lens,
        matches=packed,
        draft_tokens=proposal.draft_tokens,
        parent_list=proposal.parent_list,
        top_scores_index=proposal.top_scores_index,
        node_count=None,
        node_draft_tokens=node_tokens,
        node_draft_probs=node_probs,
        width=width,
        topk=k,
        path_depth=num_depths,
        max_extension=max_extension,
        min_match=4,
    )
    ext_len = ext.ext_len.cpu()
    assert (ext_len > 0).sum() >= rows // 4
    tree_seq_lens = torch.full((rows,), 50, dtype=torch.long, device="cuda")
    _, positions, retrive_index, next_token, next_sibling, candidates = (
        build_tree_kernel_efficient(
            bonus_tokens=torch.zeros(rows, dtype=torch.long, device="cuda"),
            parent_list=ext.parent_list,
            top_scores_index=ext.top_scores_index,
            draft_tokens=ext.draft_tokens,
            seq_lens=tree_seq_lens,
            seq_lens_sum=int(tree_seq_lens.sum()),
            topk=k,
            spec_steps=ext.depth,
            num_verify_tokens=width,
            tree_mask_mode=TreeMaskMode.FULL_MASK,
        )
    )
    positions = positions.view(rows, width).cpu()
    candidates = candidates.view(rows, width)
    # Greedy: the target predicts each accepted node's successor along path + chain.
    target = torch.zeros(rows, width, dtype=torch.long)
    for row in range(rows):
        e = int(ext_len[row])
        first = width - e
        nodes = [0] + paths[row] + list(range(first, width))[:e]
        for depth, node in enumerate(nodes):
            assert positions[row, node].item() == 50 + depth
        for a, b in zip(nodes, nodes[1:]):
            target[row, a] = candidates[row, b].item()
    accept_index = torch.full(
        (rows, ext.depth + 1), -1, dtype=torch.int32, device="cuda"
    )
    accept_num = torch.zeros(rows, dtype=torch.int32, device="cuda")
    predicts = torch.zeros(rows * width, dtype=torch.int32, device="cuda")
    verify_tree_greedy_func(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_num,
        candidates=candidates,
        retrieve_index=retrive_index.view(rows, width),
        retrieve_next_token=next_token.view(rows, width),
        retrieve_next_sibling=next_sibling.view(rows, width),
        target_predict=target.cuda(),
        topk=k,
    )
    want = torch.tensor(
        [len(path) + int(e) for path, e in zip(paths, ext_len)], dtype=torch.int32
    )
    assert torch.equal(accept_num.cpu(), want)

    # Traversal Verification: kernel vs reference on the extended rows.
    logits = torch.randn(rows, width, vocab, device="cuda")
    boost = math.log(vocab) + 1 + ext.node_draft_probs.log().clamp(min=-30) * 0.7
    logits.scatter_add_(2, ext.node_draft_tokens, boost)
    target_probs = torch.softmax(logits, dim=-1)
    coins = torch.rand(rows, width, device="cuda")
    final_coins = torch.rand(rows, device="cuda")
    tree = dict(
        candidates=candidates,
        retrive_index=retrive_index.view(rows, width),
        retrive_next_token=next_token.view(rows, width),
        retrive_next_sibling=next_sibling.view(rows, width),
        node_draft_tokens=ext.node_draft_tokens,
        node_draft_probs=ext.node_draft_probs,
    )
    results = []
    for fn, dev in (
        (tree_speculative_sampling_traversal, "cuda"),
        (tree_speculative_sampling_traversal_reference, "cpu"),
    ):
        args = {name: value.to(dev) for name, value in tree.items()}
        out = (
            torch.full((rows * width,), -1, dtype=torch.int32, device=dev),
            torch.full((rows, ext.depth + 1), -1, dtype=torch.int32, device=dev),
            torch.zeros(rows, dtype=torch.int32, device=dev),
        )
        margins = fn(
            predicts=out[0],
            accept_index=out[1],
            accept_token_num=out[2],
            uniform_samples=coins.to(dev),
            uniform_samples_for_final_sampling=final_coins.to(dev),
            target_probs=target_probs.to(dev),
            **args,
        )
        results.append((out, margins))
    (_, got_index, got_num), _ = results[0]
    (_, ref_index, ref_num), margins = results[1]
    clear = margins[:, 0] > 1e-5
    assert clear.sum() >= rows - 2
    assert torch.equal(got_num.cpu()[clear], ref_num[clear])
    assert torch.equal(got_index.cpu()[clear], ref_index[clear])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
