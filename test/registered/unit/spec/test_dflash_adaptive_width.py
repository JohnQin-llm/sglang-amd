"""DFLASH adaptive tree width: the host policy and the sync-free stats hand-off."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative.dflash_adaptive_width import (
    StaleDeviceStats,
    VerifyWidthPolicy,
    interpolate_cost,
    read_adaptive_width_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Shaped like the measured Glimmer TP1 table: width is nearly free at bs 1 and costly at 16.
_COST_MS = {
    16: {1: 15.0, 4: 17.0, 16: 24.0},
    32: {1: 15.6, 4: 18.5, 16: 33.0},
}


class _FakeEvent:
    done = False

    def record(self):
        self.done = False

    def query(self):
        return self.done


class TestInterpolateCost(CustomTestCase):
    def test_measured_points_and_linear_between(self):
        row = _COST_MS[16]
        self.assertEqual(interpolate_cost(row, 4), 17.0)
        self.assertAlmostEqual(interpolate_cost(row, 10), 20.5)

    def test_flat_outside_the_measured_range(self):
        self.assertEqual(interpolate_cost(_COST_MS[32], 0), 15.6)
        self.assertEqual(interpolate_cost(_COST_MS[32], 64), 33.0)


class TestVerifyWidthPolicy(CustomTestCase):
    def setUp(self):
        self.policy = VerifyWidthPolicy(widths=(16, 32), cost_ms=_COST_MS)

    def test_wide_tree_at_low_load_narrow_at_high_load(self):
        accept_lens = [6.5, 7.3]
        self.assertEqual(self.policy.choose(bs=1, expected_accept_lens=accept_lens), 32)
        self.assertEqual(
            self.policy.choose(bs=16, expected_accept_lens=accept_lens), 16
        )

    def test_follows_the_predicted_gain(self):
        # At bs 4 the wider tree costs 8.8% more, so it needs at least that much AL.
        self.assertEqual(self.policy.choose(bs=4, expected_accept_lens=[6.0, 6.6]), 32)
        self.assertEqual(self.policy.choose(bs=4, expected_accept_lens=[6.0, 6.4]), 16)

    def test_no_gain_takes_the_narrowest_width(self):
        # The narrowest width matches the chain's verify rows: without predicted
        # gain the policy never pays for extra rows.
        for bs in (1, 4, 16):
            self.assertEqual(
                self.policy.choose(bs=bs, expected_accept_lens=[5.0, 5.0]), 16
            )


class TestReadConfig(CustomTestCase):
    def _env(self, **values):
        keys = (
            "SGLANG_DFLASH_TREE_WIDTHS",
            "SGLANG_DFLASH_TREE_COST_TABLE",
            "SGLANG_DFLASH_TREE_FORCE_WIDTH",
            "SGLANG_DFLASH_TREE_CALIBRATION_DUMP",
        )
        env = {k: v for k, v in os.environ.items() if k not in keys}
        env.update(values)
        return patch.dict(os.environ, env, clear=True)

    def _table(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(
            {
                "cost_ms": {
                    str(q): {str(b): ms for b, ms in row.items()}
                    for q, row in _COST_MS.items()
                }
            },
            f,
        )
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_unset_is_off(self):
        with self._env():
            self.assertIsNone(read_adaptive_width_config(32))

    def test_widths_and_table(self):
        with self._env(
            SGLANG_DFLASH_TREE_WIDTHS="32,16",
            SGLANG_DFLASH_TREE_COST_TABLE=self._table(),
        ):
            config = read_adaptive_width_config(32)
        self.assertEqual(config.widths, (16, 32))
        self.assertEqual(config.cost_ms[32][16], 33.0)

    def test_largest_width_must_be_the_server_width(self):
        with self._env(SGLANG_DFLASH_TREE_WIDTHS="16,32"):
            with self.assertRaises(ValueError):
                read_adaptive_width_config(48)

    def test_policy_needs_costs_for_every_width(self):
        with self._env(SGLANG_DFLASH_TREE_WIDTHS="16,32,48"):
            with self.assertRaises(ValueError):
                read_adaptive_width_config(48)

    def test_forced_width_needs_no_table(self):
        with self._env(
            SGLANG_DFLASH_TREE_WIDTHS="16,32", SGLANG_DFLASH_TREE_FORCE_WIDTH="16"
        ):
            self.assertEqual(read_adaptive_width_config(32).force_width, 16)


class TestStaleDeviceStats(CustomTestCase):
    def test_reads_only_finished_copies_newest_last(self):
        stats = StaleDeviceStats(
            initial=[1.0, 1.0],
            device=torch.device("cpu"),
            slots=3,
            event_factory=_FakeEvent,
        )
        stats.push(torch.tensor([2.0, 3.0]))
        stats.push(torch.tensor([4.0, 5.0]))
        self.assertEqual(stats.latest(), [1.0, 1.0])
        stats._events[0].done = True
        self.assertEqual(stats.latest(), [2.0, 3.0])
        stats._events[1].done = True
        self.assertEqual(stats.latest(), [4.0, 5.0])

    def test_overwritten_slot_is_dropped(self):
        stats = StaleDeviceStats(
            initial=[0.0],
            device=torch.device("cpu"),
            slots=2,
            event_factory=_FakeEvent,
        )
        for value in (1.0, 2.0, 3.0):
            stats.push(torch.tensor([value]))
        for event in stats._events:
            event.done = True
        self.assertEqual(stats.latest(), [3.0])


if __name__ == "__main__":
    unittest.main()
