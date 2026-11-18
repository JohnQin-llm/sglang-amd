"""The post-verify KV move takes its targets from the verify window, not from
req_to_token: the scheduler may rewrite a finished request's row once verify's
read-done event fires, and a reread would move KV into the new owner's slots.
The DFLASH tree verify's window itself is read at the start of the step."""

import contextlib
import unittest
import unittest.mock
from types import SimpleNamespace

import torch

from sglang.srt.speculative.spec_utils import move_accept_tokens_to_target_kvcache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _RecordingKVCache:
    def move_kv_cache(self, tgt_loc, src_loc):
        self.tgt_loc, self.src_loc = tgt_loc.clone(), src_loc.clone()


@unittest.skipUnless(torch.cuda.is_available(), "the move kernels are Triton")
class TestMoveAcceptTokensTargets(CustomTestCase):
    def _run(self, window, accept_rows):
        device = "cuda"
        rows = torch.tensor([5, 2, 7], device=device)
        seq_lens = torch.tensor([10, 20, 30], device=device)
        req_to_token = torch.arange(512, dtype=torch.int32, device=device).view(8, 64)
        out_cache_loc = torch.cat(
            [req_to_token[r, s : s + window] for r, s in zip(rows, seq_lens)]
        ).to(torch.int64)
        accept_index = torch.tensor(accept_rows, dtype=torch.int32, device=device)
        num_accepted = (accept_index != -1).sum(dim=1)

        expected_tgt = torch.zeros(accept_index.numel(), dtype=torch.int64)
        packed = torch.cat(
            [
                req_to_token[r, s : s + n]
                for r, s, n in zip(rows, seq_lens, num_accepted)
            ]
        )
        expected_tgt[: packed.numel()] = packed.cpu()
        valid = accept_index[accept_index != -1].long()
        expected_src = torch.zeros(accept_index.numel(), dtype=torch.int64)
        expected_src[: valid.numel()] = out_cache_loc[valid].cpu()

        # The finished requests' rows go to new requests after read-done.
        req_to_token[rows] = -7

        kv_cache = _RecordingKVCache()
        move_accept_tokens_to_target_kvcache(
            SimpleNamespace(
                seq_lens=seq_lens,
                req_pool_indices=rows,
                req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
                out_cache_loc=out_cache_loc,
            ),
            accept_index,
            num_accepted - 1,
            SimpleNamespace(get_kvcache=lambda: kv_cache),
        )
        self.assertTrue(torch.equal(kv_cache.tgt_loc.cpu(), expected_tgt))
        self.assertTrue(torch.equal(kv_cache.src_loc.cpu(), expected_src))

    def test_chain_targets_ignore_rewritten_rows(self):
        # NGRAM: accept_index spans the whole window.
        self._run(
            window=6,
            accept_rows=[
                [0, 1, 2, -1, -1, -1],
                [6, -1, -1, -1, -1, -1],
                [12, 13, 14, 15, 16, 17],
            ],
        )

    def test_tree_targets_ignore_rewritten_rows(self):
        # topk > 1: the accepted path is shorter than the window and not a prefix of it.
        self._run(
            window=8,
            accept_rows=[
                [0, 2, 5, -1],
                [8, -1, -1, -1],
                [16, 19, 20, 23],
            ],
        )


@unittest.skipUnless(torch.cuda.is_available(), "the tree kernels are Triton")
class TestDFlashTreeVerifyWindow(CustomTestCase):
    def test_tree_verify_writes_the_window_read_at_step_start(self):
        """The DFLASH tree verify plans its KV writes after the draft forward. On HIP
        (no WAR barrier) a finished request's row may by then list a new request's
        radix-cached prefix, so a reread wrote tree nodes over cached prompt KV."""
        from sglang.kernels.ops.speculative.dflash_tree import (
            build_dflash_sampled_tree_proposal,
            tree_depth,
            tree_prefix,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.speculative import dflash_tree_verify

        device = "cuda"
        rows = torch.tensor([5, 2, 7], device=device)
        seq_lens = torch.tensor([10, 20, 30], device=device)
        req_to_token = torch.arange(512, dtype=torch.int32, device=device).view(8, 64)
        max_nodes, width, num_depths, topk = 8, 6, 3, 4
        expected = torch.cat(
            [req_to_token[r, s : s + width] for r, s in zip(rows, seq_lens)]
        ).to(torch.int64)
        window = dflash_tree_verify.read_tree_verify_window(
            req_pool_indices=rows,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            num_nodes=max_nodes,
        )
        # The rows go to new requests while the draft forward runs.
        req_to_token[rows] = -7

        bs = rows.numel()
        candidate_ids = torch.arange(num_depths * topk, device=device).view(
            1, num_depths, topk
        )
        log_q = torch.log_softmax(
            torch.randn(bs, num_depths, topk, topk, device=device), dim=-1
        )
        proposal, node_tokens, node_probs = build_dflash_sampled_tree_proposal(
            candidate_ids.repeat(bs, 1, 1), log_q, max_nodes=max_nodes
        )
        verified = {}

        def capture(batch, **kwargs):
            verified["spec_info"] = batch.spec_info

        model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(verify_mask=None, max_context_len=64),
            model_config=SimpleNamespace(model_is_mrope=False),
            req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
            token_to_kv_pool_allocator=None,
        )
        with unittest.mock.patch.object(
            dflash_tree_verify, "run_eagle_verify", capture
        ):
            dflash_tree_verify.verify_dflash_tree(
                batch=SimpleNamespace(
                    forward_mode=ForwardMode.DECODE,
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens.cpu(),
                    seq_lens_sum=int(seq_lens.sum()),
                    req_pool_indices=rows,
                    req_to_token_pool=model_runner.req_to_token_pool,
                    multimodal_inputs=[None] * bs,
                ),
                bonus_tokens=torch.zeros(bs, dtype=torch.long, device=device),
                proposal=tree_prefix(proposal, width, topk=topk),
                target_worker=SimpleNamespace(model_runner=model_runner),
                plan_stream=None,
                plan_stream_ctx=contextlib.nullcontext(),
                topk=topk,
                depth=tree_depth(proposal, width, topk=topk),
                num_nodes=width,
                out_cache_loc=window[:, :width].reshape(-1),
                device=device,
                node_draft_tokens=node_tokens[:, :width].contiguous(),
                node_draft_probs=node_probs[:, :width].contiguous(),
            )
        # eagle_prepare_for_verify writes the verify KV here and rereads
        # req_to_token only when it is unset.
        prepared = verified["spec_info"].prepared_out_cache_loc
        self.assertIsNotNone(prepared)
        self.assertTrue(torch.equal(prepared.cpu(), expected.cpu()))


if __name__ == "__main__":
    unittest.main()
