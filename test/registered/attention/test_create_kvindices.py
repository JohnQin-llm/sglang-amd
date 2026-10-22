import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Triton kernel unit test for KV indices creation
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestCreateKvIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_device(get_device())

    def test_pack_read_stream_graph_replay(self):
        for page_size in (1, 16):
            for with_start in (False, True):
                for with_mapping in (False, True):
                    with self.subTest(
                        page_size=page_size, start=with_start, mapping=with_mapping
                    ):
                        lengths = [0, 513, 32769]
                        width = 65536 // page_size
                        table = torch.randint(0, 1024, (3, width), dtype=torch.int32)
                        rows = torch.tensor([2, 0, 1], dtype=torch.int32)
                        lens = torch.tensor(lengths, dtype=torch.int32)
                        indptr = torch.zeros(4, dtype=torch.int32)
                        indptr[1:] = lens.cumsum(0)
                        starts = [0, 7, 15] if with_start else [0, 0, 0]
                        offsets = (
                            torch.tensor(starts, dtype=torch.int32)
                            if with_start
                            else None
                        )
                        mapping = (
                            torch.arange(1024 * page_size, dtype=torch.int64)
                            .flip(0)
                            .contiguous()
                            if with_mapping
                            else None
                        )
                        if mapping is not None:
                            mapping[::13] = -1
                        out = torch.full((sum(lengths) + 17,), -777, dtype=torch.int64)
                        view = SimpleNamespace(
                            ids=table,
                            row_ids=rows,
                            row_stride=table.stride(0),
                            entry_page_size=page_size,
                            v2p=None,
                            is_translated=False,
                        )
                        reader = SimpleNamespace(
                            _reads_translated=lambda kind: False,
                            read_source=lambda *args, **kwargs: view,
                        )

                        def run():
                            return KVIndexTranslator.pack_read_stream(
                                reader,
                                None,
                                req_pool_indices=rows,
                                seq_lens=lens,
                                indptr=indptr,
                                out=out,
                                kv_start_idx=offsets,
                                token_mapping=mapping,
                            )

                        self.assertEqual(run(), with_mapping)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            run()
                        saved = table.clone()
                        for state in ("initial", "changed", "restored"):
                            if state == "changed":
                                table.add_(1).remainder_(1024)
                            elif state == "restored":
                                table.copy_(saved)
                            graph.replay()
                            expected = []
                            for row, start, length in zip((2, 0, 1), starts, lengths):
                                pos = torch.arange(start, start + length)
                                values = table[row, pos // page_size].to(torch.int64)
                                values = values * page_size + pos % page_size
                                if mapping is not None:
                                    values = mapping[values]
                                expected.append(values)
                            torch.cuda.synchronize()
                            self.assertTrue(
                                torch.equal(out[: sum(lengths)], torch.cat(expected))
                            )
                            self.assertTrue(bool((out[sum(lengths) :] == -777).all()))
                        graph.reset()

    def _run_test(self, batch, max_batch, max_context_len):
        req_to_token = torch.arange(
            max_batch * max_context_len, dtype=torch.int32, device=get_device()
        ).reshape((max_batch, max_context_len))
        req_pool_indices = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_batch), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device=get_device(),
        )
        paged_kernel_lens = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_context_len), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device=get_device(),
        )

        kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device=get_device())
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # ref
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
        kv_indices_ref = torch.cat(
            [
                req_to_token[req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]]
                for i in range(batch)
            ],
            dim=0,
        ).contiguous()

        # triton
        kv_indices_triton = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device=get_device()
        )
        create_flashinfer_kv_indices_triton[(batch,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            None,
            kv_indices_triton,
            req_to_token.size(1),
        )

        # Check
        self.assertTrue(torch.equal(kv_indices_ref, kv_indices_triton))

    def test_create_kvindices(self):
        BATCH = [1, 37, 1786]
        MAX_BATCH = 4096
        MAX_CONTEXT_LEN = 4096
        for batch in BATCH:
            self._run_test(batch, MAX_BATCH, MAX_CONTEXT_LEN)

    def _run_page_table_test(self, batch, ps, with_window_start):
        """ENTRY_PAGE_SIZE > 1: the source is a PAGE-granular table (the unified
        pool's read table); the kernel must reconstruct token ids by the affine
        rule token = entry * ps + pos % ps -- including the kv_start_idx
        (sliding-window) offset path, whose pos is an absolute token position."""
        max_batch, max_pages = 64, 128
        page_table = torch.randint(
            0, 1 << 20, (max_batch, max_pages), dtype=torch.int32
        )
        req_pool_indices = torch.tensor(
            np.random.choice(range(max_batch), size=batch, replace=False),
            dtype=torch.int32,
        )
        lens = torch.tensor(
            np.random.randint(1, max_pages * ps, size=batch), dtype=torch.int32
        )
        if with_window_start:
            start = torch.clamp(
                lens - torch.randint(1, ps * 3, (batch,), dtype=torch.int32), min=0
            )
            gather_lens = lens - start
        else:
            start, gather_lens = None, lens
        kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(gather_lens, dim=0)

        # ref: absolute positions [start, start+len) through the affine rule
        refs = []
        for i in range(batch):
            s = int(start[i]) if start is not None else 0
            pos = torch.arange(s, s + int(gather_lens[i]), dtype=torch.int64)
            entry = page_table[int(req_pool_indices[i])][pos // ps].to(torch.int64)
            refs.append(entry * ps + pos % ps)
        ref = torch.cat(refs).contiguous()

        out = torch.empty(int(kv_indptr[-1]), dtype=torch.int64)
        create_flashinfer_kv_indices_triton[(batch,)](
            page_table,
            req_pool_indices,
            gather_lens,
            kv_indptr,
            start,
            out,
            page_table.size(1),
            ENTRY_PAGE_SIZE=ps,
        )
        self.assertTrue(torch.equal(ref, out))

    def test_page_table_source_reconstruction(self):
        for batch in (1, 37):
            for ps in (4, 64, 256):
                self._run_page_table_test(batch, ps, with_window_start=False)
                self._run_page_table_test(batch, ps, with_window_start=True)


if __name__ == "__main__":
    unittest.main()
