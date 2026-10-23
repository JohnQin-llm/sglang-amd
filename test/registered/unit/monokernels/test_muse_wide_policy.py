# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for opt-in wide verification and its compact routing layout."""

import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.rocm_muse_spark.mono import spec
from sglang.srt.rocm_muse_spark.mono.ops import k2_dispatch, wide_contract
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestMuseWidePolicy(unittest.TestCase):
    def test_wide_opt_in_preserves_default_widths(self):
        try:
            for enabled in (False, True):
                with (
                    self.subTest(enabled=enabled),
                    patch.dict(
                        os.environ, {"SGLANG_MUSE_MONO_WIDE_STAGED": str(int(enabled))}
                    ),
                ):
                    importlib.reload(spec)
                    self.assertEqual(
                        spec.SERVING_WIDTHS, (1, 8, 16, 24) if enabled else (1, 8)
                    )
                    self.assertEqual(spec.MAX_TOKENS, 8)
                    self.assertEqual(
                        spec.ALLOCATION_WIDTHS,
                        tuple(range(1, 9)) + ((16, 24) if enabled else ()),
                    )
                    mode = SimpleNamespace(
                        is_decode=lambda: False, is_target_verify=lambda: True
                    )
                    for rows in (8, 15, 16, 17, 23, 24, 25):
                        batch = SimpleNamespace(forward_mode=mode, batch_size=1)
                        accepted = spec.step_disable_reason(batch, rows) is None
                        self.assertEqual(
                            accepted, rows == 8 or (enabled and rows in (16, 24))
                        )
                    batch = SimpleNamespace(forward_mode=mode, batch_size=3)
                    self.assertIsNotNone(spec.step_disable_reason(batch, 16))
        finally:
            importlib.reload(spec)

    def test_dispatch_keeps_narrow_family_and_rejects_padding_gaps(self):
        for rows in range(1, 9):
            self.assertEqual(k2_dispatch.family_for_width(rows), "narrow")
        for rows in (16, 24):
            self.assertEqual(k2_dispatch.family_for_width(rows), "wide")
        for rows in (0, 9, 15, 17, 23, 25, 32):
            with self.assertRaises(ValueError):
                k2_dispatch.family_for_width(rows)

    def test_compact_picks_preserve_every_token_expert_pair(self):
        for rows in (16, 24):
            for spread in (0, 7, 8):
                routes = [
                    [(token * spread + k) % 256 for k in range(8)]
                    for token in range(rows)
                ]
                experts, picks, fans = wide_contract.deterministic_union(routes, rows)
                decoded = set()
                for slot, expert in enumerate(experts):
                    seen = 0
                    for offset in range(rows):
                        word = picks[slot * rows + offset]
                        if word < 0:
                            continue
                        token, k = (
                            (word >> 4, word & 15) if rows == 24 else (offset, word)
                        )
                        self.assertEqual(routes[token][k], expert)
                        self.assertNotIn((token, k), decoded)
                        decoded.add((token, k))
                        seen += 1
                    self.assertEqual(seen, fans[slot])
                self.assertEqual(
                    decoded, {(token, k) for token in range(rows) for k in range(8)}
                )

    def test_m24_live_tail_starts_above_sixteen_picks(self):
        for fan in (1, 8, 16):
            self.assertEqual(wide_contract.expert_tiles(24, fan), ((0, 16),))
        for fan in (17, 24):
            self.assertEqual(wide_contract.expert_tiles(24, fan), ((0, 16), (16, 8)))
        self.assertEqual(wide_contract.expert_tiles(24, 0), ())
        with self.assertRaises(ValueError):
            wide_contract.expert_tiles(24, 25)


if __name__ == "__main__":
    unittest.main()
