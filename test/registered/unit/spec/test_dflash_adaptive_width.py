"""DFLASH adaptive tree width: the host policy and the sync-free stats hand-off."""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dflash_adaptive_width import (
    CHAIN,
    AdaptiveVerifyWidth,
    AdaptiveWidthConfig,
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
            "SGLANG_DFLASH_TREE_SAMPLED_WIDTHS",
            "SGLANG_DFLASH_TREE_COST_TABLE",
            "SGLANG_DFLASH_TREE_FORCE_WIDTH",
            "SGLANG_DFLASH_TREE_CALIBRATION_DUMP",
        )
        env = {k: v for k, v in os.environ.items() if k not in keys}
        env.update(values)
        return patch.dict(os.environ, env, clear=True)

    def _table(self, sampled=None):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        table = {
            "cost_ms": {
                str(q): {str(b): ms for b, ms in row.items()}
                for q, row in _COST_MS.items()
            }
        }
        if sampled is not None:
            table["sampled_cost_ms"] = sampled
        json.dump(table, f)
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

    def test_sampled_members_with_the_chain(self):
        sampled = {"chain": {"1": 14.0, "16": 22.0}, "32": {"1": 16.0, "16": 36.0}}
        with self._env(
            SGLANG_DFLASH_TREE_WIDTHS="16,32",
            SGLANG_DFLASH_TREE_SAMPLED_WIDTHS="32,chain",
            SGLANG_DFLASH_TREE_COST_TABLE=self._table(sampled),
        ):
            config = read_adaptive_width_config(32)
        self.assertEqual(config.widths, (16, 32))
        self.assertEqual(config.sampled_members, (CHAIN, 32))
        self.assertEqual(config.sampled_cost_ms[CHAIN][16], 22.0)
        self.assertEqual(config.cost_ms[16][16], 24.0)

    def test_sampled_members_need_sampled_costs(self):
        with self._env(
            SGLANG_DFLASH_TREE_SAMPLED_WIDTHS="chain,32",
            SGLANG_DFLASH_TREE_COST_TABLE=self._table(),
        ):
            with self.assertRaises(ValueError):
                read_adaptive_width_config(32)

    def test_sampled_batches_reuse_the_greedy_table(self):
        with self._env(
            SGLANG_DFLASH_TREE_WIDTHS="16,32",
            SGLANG_DFLASH_TREE_COST_TABLE=self._table(),
        ):
            config = read_adaptive_width_config(32)
        self.assertEqual(config.sampled_members, (16, 32))
        self.assertEqual(config.sampled_cost_ms, config.cost_ms)

    def test_greedy_widths_hold_no_chain(self):
        with self._env(SGLANG_DFLASH_TREE_WIDTHS="chain,32"):
            with self.assertRaises(ValueError):
                read_adaptive_width_config(32)

    def test_forced_chain(self):
        with self._env(
            SGLANG_DFLASH_TREE_SAMPLED_WIDTHS="chain,16,32",
            SGLANG_DFLASH_TREE_FORCE_WIDTH="chain",
        ):
            config = read_adaptive_width_config(32)
        self.assertEqual(config.force_width, CHAIN)
        width = AdaptiveVerifyWidth(
            config=config, num_depths=7, chain_rows=8, device=torch.device("cpu")
        )
        self.assertEqual(width.choose_sampled_member(bs=4), CHAIN)
        self.assertEqual(width.member_rows(CHAIN), 8)
        # Greedy batches keep the server width; only the chain's 8 rows need graphs.
        self.assertEqual(width.select_width(survival=None, bs=4), 32)
        self.assertEqual(width._captured_rows(), [8])


class TestChainMember(CustomTestCase):
    def test_policy_falls_back_to_the_chain_at_high_load(self):
        # Shaped like Qwen at T=1: the chain verifies 8 rows, the tree 16.
        cost = {CHAIN: {1: 20.0, 16: 28.0}, 16: {1: 20.5, 16: 37.0}}
        policy = VerifyWidthPolicy(widths=(CHAIN, 16), cost_ms=cost)
        accept_lens = [3.95, 4.35]
        self.assertEqual(policy.choose(bs=1, expected_accept_lens=accept_lens), 16)
        self.assertEqual(policy.choose(bs=16, expected_accept_lens=accept_lens), CHAIN)

    def test_member_choice_reads_only_earlier_steps(self):
        config = AdaptiveWidthConfig(
            widths=(16,),
            sampled_members=(CHAIN, 16),
            cost_ms={},
            sampled_cost_ms={CHAIN: {1: 10.0}, 16: {1: 10.0}},
            force_width=None,
            calibration_dump=None,
        )
        width = AdaptiveVerifyWidth(
            config=config, num_depths=15, chain_rows=16, device=torch.device("cpu")
        )
        width._sampled_stats._events = [_FakeEvent() for _ in range(4)]
        # Equal cost: before any copy lands the tie goes to the chain.
        self.assertEqual(width.choose_sampled_member(bs=1), CHAIN)
        tree = SimpleNamespace(prefix_accept_len=torch.tensor([[1.0] * 15 + [6.0]]))
        chain = SimpleNamespace(prefix_accept_len=torch.tensor([[1.0] * 15 + [5.0]]))
        width.predict_sampled(tree_survival=tree, chain_survival=chain, bs=1)
        # The copy has not landed yet, so this step's draws cannot steer the choice.
        self.assertEqual(width.choose_sampled_member(bs=1), CHAIN)
        for event in width._sampled_stats._events:
            event.done = True
        self.assertEqual(width.choose_sampled_member(bs=1), 16)


class TestSampledPolicy(CustomTestCase):
    def _width(self):
        config = AdaptiveWidthConfig(
            widths=(32,),
            sampled_members=(CHAIN, 16, 32),
            cost_ms={},
            sampled_cost_ms={CHAIN: {1: 10.0}, 16: {1: 11.0}, 32: {1: 12.0}},
            force_width=None,
            calibration_dump=None,
        )
        return AdaptiveVerifyWidth(
            config=config, num_depths=15, chain_rows=16, device=torch.device("cpu")
        )

    def test_exploration_visits_the_runner_up_then_every_member(self):
        width = self._width()
        width._sampled_stats._events = [_FakeEvent() for _ in range(4)]
        # Before any prediction lands the ranking is by cost: chain, 16, 32.
        picks = [width.choose_sampled_member(bs=1) for _ in range(512)]
        explored = picks[31::32]
        self.assertEqual(explored[:8], [16] * 7 + [32])
        self.assertEqual(explored[15], 16)
        self.assertTrue(all(p == CHAIN for i, p in enumerate(picks) if i % 32 != 31))

    def test_member_scale_rescales_the_prediction(self):
        width = self._width()
        width._sampled_stats._events = [_FakeEvent() for _ in range(4)]
        # 16 accepted for every 10 predicted at member 16, after plenty of verifies.
        width._member_totals[:, 1] = torch.tensor([1.6e9, 1.0e9])
        tree = SimpleNamespace(prefix_accept_len=torch.arange(1.0, 33.0).view(1, 32))
        chain = SimpleNamespace(prefix_accept_len=torch.full((1, 16), 4.0))
        out = width.predict_sampled(tree_survival=tree, chain_survival=chain, bs=1)
        torch.testing.assert_close(out[0, 3:], torch.tensor([4.0, 16.0, 32.0]))
        torch.testing.assert_close(
            out[0, :3], torch.tensor([4.0, 16.0 * 1.6, 32.0]), rtol=1e-4, atol=1e-4
        )


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
