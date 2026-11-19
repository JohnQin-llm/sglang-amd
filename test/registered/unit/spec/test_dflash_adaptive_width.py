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
    RAGGED,
    AdaptiveVerifyWidth,
    AdaptiveWidthConfig,
    StaleDeviceStats,
    StepPhases,
    VerifyWidthPolicy,
    interpolate_cost,
    load_cost_table,
    member_label,
    read_adaptive_width_config,
)
from sglang.srt.speculative.dflash_ragged_chain import (
    RaggedChainPolicy,
    RaggedChainTable,
    read_ragged_chain_table,
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

    def synchronize(self):
        self.done = True


class _TimedEvent(_FakeEvent):
    def __init__(self, enable_timing=False):
        pass

    def elapsed_time(self, other):
        return 1.5


class TestStepPhases(CustomTestCase):
    def test_completed_steps_reach_the_file_without_a_batch(self):
        """A step is written once its events complete, so a killed server loses none."""
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("torch.cuda.Event", _TimedEvent),
        ):
            path = os.path.join(tmp, "phases.jsonl")
            phases = StepPhases(path)
            phases.begin(4)
            phases.mark("verify")
            phases.end()
            with open(path) as f:
                self.assertEqual(f.read(), "")
            phases._done[0][1][-1][1].synchronize()
            phases.begin(2)
            phases.end()
            with open(path) as f:
                lines = [json.loads(x) for x in f.read().splitlines()]
            self.assertEqual([x["bs"] for x in lines], [4])
            self.assertEqual(lines[0]["verify"], 1.5)


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
        # Greedy batches keep the server width; only the chain needs its own graphs.
        self.assertEqual(width.choose_width(bs=4), 32)
        self.assertEqual(width._captured_keys(), [CHAIN])

    def test_shorter_chain_members(self):
        sampled = {"chain10": {"1": 13.0}, "chain": {"1": 14.0}, "32": {"1": 16.0}}
        with self._env(
            SGLANG_DFLASH_TREE_SAMPLED_WIDTHS="32,chain,chain10",
            SGLANG_DFLASH_TREE_COST_TABLE=self._table(sampled),
        ):
            config = read_adaptive_width_config(32)
        self.assertEqual(config.sampled_members, (-10, CHAIN, 32))
        self.assertEqual(config.sampled_cost_ms[-10][1], 13.0)
        self.assertEqual(member_label(-10), "chain10")
        width = AdaptiveVerifyWidth(
            config=config, num_depths=15, chain_rows=16, device=torch.device("cpu")
        )
        self.assertEqual(width.member_rows(-10), 10)
        self.assertEqual(width._captured_keys(), [-10, CHAIN])
        # The chain columns lead, in member order: 10 rows, then the full 16.
        tree = SimpleNamespace(prefix_accept_len=torch.arange(1.0, 33.0).view(1, 32))
        chain = SimpleNamespace(prefix_accept_len=torch.arange(1.0, 17.0).view(1, 16))
        out = width.predict_sampled(tree_survival=tree, chain_survival=chain, bs=1)
        torch.testing.assert_close(out[0, 3:], torch.tensor([10.0, 16.0, 32.0]))

    def test_chain_member_shorter_than_the_block(self):
        with self._env(
            SGLANG_DFLASH_TREE_SAMPLED_WIDTHS="chain16,32",
            SGLANG_DFLASH_TREE_FORCE_WIDTH="32",
        ):
            config = read_adaptive_width_config(32)
        with self.assertRaises(ValueError):
            AdaptiveVerifyWidth(
                config=config, num_depths=15, chain_rows=16, device=torch.device("cpu")
            )


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


class TestMeasuredMemberLevels(CustomTestCase):
    def _width(self):
        config = AdaptiveWidthConfig(
            widths=(16,),
            sampled_members=(CHAIN, 8),
            cost_ms={},
            # Glimmer at bs 16: the chain verifies 16 rows, the 8-node tree 8.
            sampled_cost_ms={CHAIN: {16: 25.64}, 8: {16: 20.53}},
            force_width=None,
            calibration_dump=None,
            sampled_accept_len={CHAIN: {1: 5.35}, 8: {1: 4.62}},
        )
        width = AdaptiveVerifyWidth(
            config=config, num_depths=15, chain_rows=16, device=torch.device("cpu")
        )
        width._sampled_stats._events = [_FakeEvent() for _ in range(4)]
        return width

    def _predict(self, width, chain_al, tree_al):
        tree = SimpleNamespace(prefix_accept_len=torch.full((16, 16), tree_al))
        chain = SimpleNamespace(prefix_accept_len=torch.full((16, 16), chain_al))
        width.predict_sampled(tree_survival=tree, chain_survival=chain, bs=16)
        for event in width._sampled_stats._events:
            event.done = True

    def test_levels_come_from_the_measured_lengths(self):
        # The per-node model favors the chain by 30%; its level is ignored.
        width = self._width()
        self._predict(width, chain_al=6.5, tree_al=5.0)
        self.assertEqual(width.choose_sampled_member(bs=16), 8)

    def test_content_moves_the_choice(self):
        width = self._width()
        for _ in range(200):
            self._predict(width, chain_al=5.0, tree_al=5.0)
        # Content that lifts the chain's estimate well over its long-run average (and
        # not the tree's) makes the chain the faster member.
        for _ in range(40):
            self._predict(width, chain_al=9.0, tree_al=5.0)
        self.assertEqual(width.choose_sampled_member(bs=16), CHAIN)
        # No exploration steps: every pick is the policy's.
        picks = {width.choose_sampled_member(bs=16) for _ in range(64)}
        self.assertEqual(picks, {CHAIN})


def _ragged_policy(rows_per_request):
    """Block 16, one collision-mass bin, and the given [largest bs, rows] pairs."""
    table = RaggedChainTable(
        block=16,
        bin_edges=[],
        accept_lens=[[1.0 + 0.3 * k for k in range(16)]],
        rows_per_request=rows_per_request,
    )
    return RaggedChainPolicy(table=table, device=torch.device("cpu"))


class TestRaggedMember(CustomTestCase):
    def _config(self, **overrides):
        values = dict(
            widths=(16,),
            sampled_members=(RAGGED, CHAIN, 8),
            cost_ms={},
            # Shaped like Glimmer T=1: the ragged chain verifies 8 rows per request at
            # bs 16 and the whole block at bs 1.
            sampled_cost_ms={
                RAGGED: {1: 15.8, 16: 20.6},
                CHAIN: {1: 15.8, 16: 25.6},
                8: {1: 15.4, 16: 20.5},
            },
            force_width=None,
            calibration_dump=None,
            sampled_accept_len={
                RAGGED: {1: 5.36, 16: 4.95},
                CHAIN: {1: 5.36},
                8: {1: 4.62},
            },
        )
        values.update(overrides)
        return AdaptiveWidthConfig(**values)

    def _width(self, **overrides):
        width = AdaptiveVerifyWidth(
            config=self._config(**overrides),
            num_depths=15,
            chain_rows=16,
            device=torch.device("cpu"),
            ragged=_ragged_policy([[2, 16], [12, 10], [4096, 8]]),
        )
        if width._sampled_stats is not None:
            width._sampled_stats._events = [_FakeEvent() for _ in range(4)]
        return width

    def test_parse_and_label(self):
        self.assertEqual(member_label(RAGGED), "ragged")
        with patch.dict(
            os.environ,
            {
                "SGLANG_DFLASH_TREE_SAMPLED_WIDTHS": "32,chain,ragged",
                "SGLANG_DFLASH_TREE_FORCE_WIDTH": "ragged",
            },
        ):
            config = read_adaptive_width_config(32)
        self.assertEqual(config.sampled_members, (RAGGED, CHAIN, 32))
        self.assertEqual(config.force_width, RAGGED)
        with patch.dict(
            os.environ,
            {
                "SGLANG_DFLASH_TREE_WIDTHS": "ragged,32",
                "SGLANG_DFLASH_TREE_FORCE_WIDTH": "ragged",
            },
        ):
            self.assertEqual(read_adaptive_width_config(32).widths, (RAGGED, 32))

    def test_greedy_only_member_reads_the_ragged_table(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(
            {
                "block": 16,
                "bin_edges": [],
                "accept_lens": [[1.0 + 0.3 * k for k in range(16)]],
                "rows_per_request": [[4096, 8]],
            },
            f,
        )
        f.close()
        self.addCleanup(os.unlink, f.name)
        read_ragged_chain_table.cache_clear()
        self.addCleanup(read_ragged_chain_table.cache_clear)
        with patch.dict(
            os.environ,
            {
                "SGLANG_DFLASH_TREE_WIDTHS": "ragged,32",
                "SGLANG_DFLASH_TREE_FORCE_WIDTH": "ragged",
                "SGLANG_DFLASH_RAGGED_CHAIN_TABLE": f.name,
            },
        ):
            width = AdaptiveVerifyWidth.maybe_create(
                max_width=32, num_depths=15, chain_rows=16, device=torch.device("cpu")
            )
        self.assertEqual(width.ragged.widths, [8])
        self.assertEqual(width.choose_width(bs=4), RAGGED)

    def test_table_reads_per_batch_size_levels(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(
            {
                "cost_ms": {"32": {"1": 16.0}},
                "sampled_member_accept_len": {
                    "ragged": {"1": 5.4, "16": 4.9},
                    "chain": 5.36,
                },
            },
            f,
        )
        f.close()
        self.addCleanup(os.unlink, f.name)
        _, _, greedy_levels, accept_len = load_cost_table(f.name)
        self.assertEqual(greedy_levels, {})
        self.assertEqual(accept_len, {RAGGED: {1: 5.4, 16: 4.9}, CHAIN: {1: 5.36}})

    def test_needs_the_table_exactly_with_the_member(self):
        with self.assertRaises(ValueError):
            AdaptiveVerifyWidth(
                config=self._config(),
                num_depths=15,
                chain_rows=16,
                device=torch.device("cpu"),
            )
        with self.assertRaises(ValueError):
            AdaptiveVerifyWidth(
                config=self._config(sampled_members=(CHAIN, 8)),
                num_depths=15,
                chain_rows=16,
                device=torch.device("cpu"),
                ragged=_ragged_policy([[4096, 8]]),
            )

    def test_ragged_rows_share_the_chain_verify_states(self):
        width = self._width()
        # Rows per request 16 (the block: the chain's state), 10 and 8; then tree 8.
        self.assertEqual(width._captured_keys(), [-10, -8, CHAIN, 8])
        self.assertEqual(width.ragged_key(16), CHAIN)
        self.assertEqual(width.ragged_key(8), -8)
        forced = self._width(force_width=RAGGED)
        self.assertEqual(forced._captured_keys(), [-10, -8, CHAIN])

    def test_ragged_column_is_the_full_chain(self):
        width = self._width()
        tree = SimpleNamespace(prefix_accept_len=torch.arange(1.0, 33.0).view(1, 32))
        chain = SimpleNamespace(prefix_accept_len=torch.arange(1.0, 17.0).view(1, 16))
        out = width.predict_sampled(tree_survival=tree, chain_survival=chain, bs=1)
        # Raw columns in member order: ragged (the chain's 16), chain, tree 8.
        torch.testing.assert_close(out[0, 3:], torch.tensor([16.0, 16.0, 8.0]))

    def test_policy_takes_the_ragged_chain_at_high_load(self):
        width = self._width()
        tree = SimpleNamespace(prefix_accept_len=torch.full((16, 32), 5.0))
        chain = SimpleNamespace(prefix_accept_len=torch.full((16, 16), 5.0))
        width.predict_sampled(tree_survival=tree, chain_survival=chain, bs=16)
        for event in width._sampled_stats._events:
            event.done = True
        # bs 1: equal levels and costs, the tie goes to the ragged chain (first);
        # bs 16: ragged 4.95 / 20.6 beats tree 8's 4.62 / 20.5 and the chain's
        # 5.36 / 25.6.
        self.assertEqual(width.choose_sampled_member(bs=16), RAGGED)
        cheap_tree = self._width(
            sampled_cost_ms={
                RAGGED: {1: 15.8, 16: 20.6},
                CHAIN: {1: 15.8, 16: 25.6},
                8: {1: 15.4, 16: 18.0},
            }
        )
        self.assertEqual(cheap_tree.choose_sampled_member(bs=16), 8)


class TestGreedyPolicy(CustomTestCase):
    def _width(self, accept_len=None, widths=(8, 16), cost8=20.0, cost16=24.2):
        config = AdaptiveWidthConfig(
            widths=widths,
            sampled_members=(16,),
            # Shaped like Qwen at bs 16: 16 nodes cost 21% more than 8.
            cost_ms={RAGGED: {16: 19.0}, 8: {16: cost8}, 16: {16: cost16}},
            sampled_cost_ms={},
            force_width=None,
            calibration_dump=None,
            accept_len=accept_len or {},
        )
        width = AdaptiveVerifyWidth(
            config=config,
            num_depths=15,
            chain_rows=16,
            device=torch.device("cpu"),
            ragged=_ragged_policy([[4096, 8]]) if RAGGED in widths else None,
        )
        width._stats._events = [_FakeEvent() for _ in range(4)]
        return width

    def _push(self, width, al8, al16):
        survival = SimpleNamespace(
            prefix_accept_len=torch.tensor([[1.0] * 7 + [al8] + [1.0] * 7 + [al16]])
        )
        width.push_greedy(survival=survival, bs=1)
        for event in width._stats._events:
            event.done = True

    def test_one_noisy_step_does_not_switch(self):
        width = self._width()
        for _ in range(100):
            self._push(width, 4.4, 4.9)
        self.assertEqual(width.choose_width(bs=16), 8)
        # One step predicting 16 a third better than 8: the average barely moves.
        self._push(width, 4.0, 5.4)
        self.assertEqual(width.choose_width(bs=16), 8)
        # A lasting change does switch.
        for _ in range(100):
            self._push(width, 4.0, 5.4)
        self.assertEqual(width.choose_width(bs=16), 16)

    def test_measured_levels(self):
        # The per-node model favors 16 by 30%; the measured levels by 10%, under the
        # cost ratio.
        width = self._width(accept_len={8: {1: 4.6}, 16: {1: 5.06}})
        self._push(width, 4.0, 5.2)
        self.assertEqual(width.choose_width(bs=16), 8)

    def test_ragged_chain_for_greedy_batches(self):
        with self.assertRaises(ValueError):
            self._width(widths=(RAGGED, 8, 16))
        width = self._width(
            accept_len={RAGGED: {16: 4.5}, 8: {1: 4.6}, 16: {1: 5.06}},
            widths=(RAGGED, 8, 16),
        )
        self.assertEqual(width.choose_width(bs=16), RAGGED)
        self.assertEqual(width._captured_keys(), [-8, 8])
        # The ragged chain shares tree 8's content factor: content that lowers both
        # leaves the levels' choice.
        self._push(width, 4.4, 4.9)
        self._push(width, 3.5, 3.9)
        self.assertEqual(width.choose_width(bs=16), RAGGED)
        # A step where 16 looks weak against 8 picks the ragged chain; ragged steps
        # build no tree, and their relaxing content factors hand the choice back.
        cheap_tree = self._width(
            accept_len={RAGGED: {16: 4.55}, 8: {1: 4.6}, 16: {1: 5.06}},
            widths=(RAGGED, 8, 16),
            cost8=19.3,
            cost16=21.0,
        )
        self._push(cheap_tree, 4.4, 4.9)
        self.assertEqual(cheap_tree.choose_width(bs=16), 16)
        self._push(cheap_tree, 4.4, 4.0)
        self.assertEqual(cheap_tree.choose_width(bs=16), RAGGED)
        for _ in range(8):
            cheap_tree.after_ragged_verify(
                accept_lens=torch.ones(16), verify_lens=torch.ones(16), greedy=True
            )
            for event in cheap_tree._stats._events:
                event.done = True
        self.assertEqual(cheap_tree.choose_width(bs=16), 16)


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

    def test_lag_reads_the_same_step_whatever_has_landed(self):
        stats = StaleDeviceStats(
            initial=[0.0],
            device=torch.device("cpu"),
            slots=4,
            event_factory=_FakeEvent,
            lag=2,
        )
        stats.push(torch.tensor([1.0]))
        stats.push(torch.tensor([2.0]))
        self.assertEqual(stats.latest(), [0.0])
        stats.push(torch.tensor([3.0]))
        # Push 1 is two before the newest, whether or not later copies have landed.
        self.assertEqual(stats.latest(), [1.0])
        for event in stats._events:
            event.done = True
        self.assertEqual(stats.latest(), [1.0])
        stats.push(torch.tensor([4.0]))
        self.assertEqual(stats.latest(), [2.0])

    def test_share_runs_before_the_copy(self):
        # Stands in for a broadcast from TP rank 0, in place.
        stats = StaleDeviceStats(
            initial=[0.0],
            device=torch.device("cpu"),
            slots=4,
            event_factory=_FakeEvent,
            lag=1,
            share=lambda values: values.fill_(7.0),
        )
        pushed = torch.tensor([1.0])
        stats.push(pushed)
        stats.push(torch.tensor([2.0]))
        self.assertEqual(stats.latest(), [7.0])
        self.assertEqual(pushed.tolist(), [1.0])


if __name__ == "__main__":
    unittest.main()
