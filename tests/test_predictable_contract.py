"""PredictableBettingProcess 契约测试。

覆盖：prepare/observe 封存式轮次协议（下注先于观察、幂等、未封存拒绝、
重复观察拒绝）、半开区间反馈分箱（含 x=1 边界）、策略表校验、
log 域财富/归零永久保持、极小财富、状态恢复/指纹锁定/修改策略拒绝。
"""

from __future__ import annotations

import json
import math
import unittest
import copy

from mean_evidence import (
    FiniteBettingStrategy,
    PredictableBettingProcess,
)


def two_state_strategy(m=0.5):
    """二状态策略：上一轮 x < m 封存 lambda=0，否则封存 lambda=1/m。"""
    return FiniteBettingStrategy(
        states=["low", "high"],
        initial_state="low",
        bet_lambdas={"low": 0.0, "high": 1.0 / m},
        edges={
            "low": {0: "low", 1: "high"},
            "high": {0: "low", 1: "high"},
        },
        cutpoints=[0.0, m, 1.0],
    )


def fixed_strategy(m, lam):
    """固定 lambda 退化：单状态、反馈不改变状态。"""
    return FiniteBettingStrategy(
        states=["s"],
        initial_state="s",
        bet_lambdas={"s": lam},
        edges={"s": {0: "s"}},
        cutpoints=[0.0, 1.0],
    )


class StrategyConstructionTests(unittest.TestCase):
    def test_valid_two_state(self):
        s = two_state_strategy()
        self.assertEqual(s.states, ("low", "high"))
        self.assertEqual(s.num_bins, 2)
        self.assertEqual(s.lambda_for("low"), 0.0)
        self.assertEqual(s.lambda_for("high"), 2.0)

    def test_states_must_be_nonempty_strs(self):
        with self.assertRaises(ValueError):
            FiniteBettingStrategy([], "s", {"s": 1.0},
                                  {"s": {0: "s"}}, [0.0, 1.0])
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(["a", "a"], "a", {"a": 1.0},
                                  {"a": {0: "a"}}, [0.0, 1.0])
        with self.assertRaises(TypeError):
            FiniteBettingStrategy([1], 1, {1: 1.0},
                                  {1: {0: 1}}, [0.0, 1.0])

    def test_initial_state_must_exist(self):
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(["a"], "b", {"a": 1.0},
                                  {"a": {0: "a"}}, [0.0, 1.0])

    def test_cutpoints_must_partition_unit_interval(self):
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(["a"], "a", {"a": 1.0},
                                  {"a": {0: "a"}}, [0.0, 0.5])
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(["a"], "a", {"a": 1.0},
                                  {"a": {0: "a"}}, [0.0, 1.0, 0.5])
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(["a"], "a", {"a": 1.0},
                                  {"a": {0: "a"}}, [0.0])

    def test_every_bin_needs_an_edge(self):
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(
                ["a"], "a", {"a": 1.0}, {"a": {0: "a"}}, [0.0, 0.5, 1.0]
            )
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(
                ["a"], "a", {"a": 1.0}, {"a": {0: "a", 2: "a"}},
                [0.0, 0.5, 1.0],
            )

    def test_unknown_edge_target_rejected(self):
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(["a"], "a", {"a": 1.0},
                                  {"a": {0: "zzz"}}, [0.0, 1.0])

    def test_negative_lambda_rejected(self):
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(["a"], "a", {"a": -0.1},
                                  {"a": {0: "a"}}, [0.0, 1.0])

    def test_missing_state_row_or_lambda_rejected(self):
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(["a", "b"], "a",
                                  {"a": 1.0, "b": 1.0},
                                  {"a": {0: "a"}}, [0.0, 1.0])
        with self.assertRaises(ValueError):
            FiniteBettingStrategy(["a", "b"], "a", {"a": 1.0},
                                  {"a": {0: "a"}, "b": {0: "b"}},
                                  [0.0, 1.0])


class HalfOpenBinTests(unittest.TestCase):
    def test_half_open_classification_and_x_one_boundary(self):
        s = FiniteBettingStrategy(
            ["a"], "a", {"a": 1.0},
            {"a": {0: "a", 1: "a", 2: "a"}},
            [0.0, 0.25, 0.75, 1.0],
        )
        # 显式半开区间：[0,.25) [.25,.75) [.75,1]
        self.assertEqual(s.classify(0.0), 0)
        self.assertEqual(s.classify(0.2499999999), 0)
        self.assertEqual(s.classify(0.25), 1)      # 左闭
        self.assertEqual(s.classify(0.7499999999), 1)
        self.assertEqual(s.classify(0.75), 2)      # 左闭
        self.assertEqual(s.classify(1.0), 2)       # x=1 归入最后一档

    def test_x_equal_cutpoint_uses_right_bin_in_transition(self):
        s = two_state_strategy(0.5)
        self.assertEqual(s.next_state_for("low", s.classify(0.5)), "high")
        self.assertEqual(s.next_state_for("high", s.classify(1.0)), "high")


class SealedRoundProtocolTests(unittest.TestCase):
    def test_observe_without_prepare_rejected(self):
        p = PredictableBettingProcess(0.5, two_state_strategy(), 0.25)
        with self.assertRaises(RuntimeError):
            p.observe(0.9)
        self.assertEqual(p.t, 0)

    def test_repeated_prepare_is_idempotent_same_lambda(self):
        p = PredictableBettingProcess(0.5, two_state_strategy(), 0.25)
        b1 = p.prepare()
        b2 = p.prepare()
        b3 = p.prepare()
        self.assertIs(b1, b2)
        self.assertEqual(b1, b3)
        self.assertEqual(p.sealed_round, 1)
        snap = p.observe(0.9)
        self.assertEqual(snap.t, 1)
        self.assertIsNone(p.sealed_round)

    def test_double_observe_after_one_prepare_rejected(self):
        p = PredictableBettingProcess(0.5, two_state_strategy(), 0.25)
        p.prepare()
        p.observe(0.9)
        with self.assertRaises(RuntimeError):
            p.observe(0.1)  # 第二轮未封存
        self.assertEqual(p.t, 1)

    def test_bet_sealed_before_x_is_seen(self):
        # 逐事件核对：封存记录只依赖历史，状态与封存 lambda 在 observe 前确定
        p = PredictableBettingProcess(0.5, two_state_strategy(), 0.25)
        b_low = p.prepare()
        self.assertEqual((b_low.round_index, b_low.state, b_low.lam),
                         (1, "low", 0.0))
        p.observe(0.9)  # 高反馈 -> high
        b_high = p.prepare()
        self.assertEqual((b_high.round_index, b_high.state, b_high.lam),
                         (2, "high", 2.0))
        # high 状态下 x=0：因子 1+2(0-.5)=0，财富归零；状态回 low
        snap = p.observe(0.0)
        self.assertEqual(snap.evidence, 0.0)
        self.assertEqual(p.log_wealth(), -math.inf)
        self.assertEqual(p.state, "low")
        b_next = p.prepare()
        self.assertEqual((b_next.round_index, b_next.state, b_next.lam),
                         (3, "low", 0.0))

    def test_invalid_x_keeps_seal_and_state(self):
        p = PredictableBettingProcess(0.5, two_state_strategy(), 0.25)
        bet = p.prepare()
        for bad in (1.5, -0.1, float("nan"), float("inf"), True):
            with self.assertRaises((ValueError, TypeError)):
                p.observe(bad)
            self.assertEqual(p.t, 0)
            self.assertEqual(p.state, "low")
            self.assertEqual(p.sealed_bet(), bet)
        snap = p.observe(0.2)  # 封存仍有效
        self.assertEqual(snap.t, 1)

    def test_extend_requires_no_pending_seal_and_is_atomic(self):
        p = PredictableBettingProcess(0.5, fixed_strategy(0.5, 1.0), 0.25)
        p.prepare()
        with self.assertRaises(RuntimeError):
            p.extend([0.1, 0.2])
        p.observe(0.1)
        before = p.to_dict()
        with self.assertRaises(ValueError):
            p.extend([0.2, 1.5])
        self.assertEqual(p.to_dict(), before)
        p.extend([0.2, 0.8])
        self.assertEqual(p.t, 3)


class LambdaBoundTests(unittest.TestCase):
    def test_lambda_above_one_over_m_rejected_at_construction(self):
        s = FiniteBettingStrategy(
            ["a"], "a", {"a": 2.5},
            {"a": {0: "a"}}, [0.0, 1.0],
        )
        with self.assertRaises(ValueError):
            PredictableBettingProcess(0.5, s, 0.25)  # 1/m = 2

    def test_endpoint_lambda_allowed_and_zero_factor(self):
        p = PredictableBettingProcess(0.5, fixed_strategy(0.5, 2.0), 0.25)
        p.prepare()
        snap = p.observe(0)
        self.assertEqual(snap.evidence, 0.0)
        self.assertEqual(p.log_wealth(), -math.inf)


class WealthAndEvidenceTests(unittest.TestCase):
    def test_dead_wealth_stays_zero_but_state_keeps_moving(self):
        p = PredictableBettingProcess(0.5, two_state_strategy(), 0.25)
        seq = [0.9, 0.0, 1.0, 1.0, 0.1]
        # t1 low(lambda=0), x=.9 W=1 进 high；t2 high lambda=2 x=0 归零回 low；
        # 之后 W 恒 0，状态仍随反馈照常转移
        states = []
        for x in seq:
            p.prepare()
            p.observe(x)
            states.append(p.state)
        self.assertEqual(p.log_wealth(), -math.inf)
        self.assertEqual(states, ["high", "low", "high", "high", "low"])
        self.assertEqual(p.snapshot().evidence, 0.0)

    def test_max_evidence_persists_after_wealth_drops(self):
        # 固定 lambda=1, m=.5：x=1 因子 1.5；x=0 因子 .5
        p = PredictableBettingProcess(0.5, fixed_strategy(0.5, 1.0), 0.4)
        for x in (1.0, 1.0, 0.0):
            p.run_round(x)
        s = p.snapshot()
        self.assertAlmostEqual(s.evidence, 1.125)
        self.assertAlmostEqual(s.max_evidence, 2.25)
        self.assertAlmostEqual(s.p_value, 1.0 / 2.25)

    def test_tiny_wealth_underflows_to_zero_gracefully(self):
        # m=.5, lambda=1：连续 1200 个 x=0，W=0.5^1200 在 double 下溢为 0.0，
        # 但 log 域仍保留有限值，历史最大值与 p 值不受污染。
        p = PredictableBettingProcess(0.5, fixed_strategy(0.5, 1.0), 0.05)
        p.extend([0.0] * 1200)
        s = p.snapshot()
        self.assertEqual(s.evidence, 0.0)
        self.assertTrue(math.isfinite(s.log_evidence))
        self.assertEqual(s.max_evidence, 1.0)
        self.assertEqual(s.p_value, 1.0)
        self.assertIsNone(s.stop_time)

    def test_stop_time_first_crossing_only(self):
        p = PredictableBettingProcess(0.5, fixed_strategy(0.5, 1.0), 0.5)  # 阈值 2
        stops = []
        for x in (1.0, 1.0, 0.0, 0.0):
            snap = p.run_round(x)
            stops.append(snap.stop_time)
        self.assertEqual(stops, [None, 2, 2, 2])
        # lambda=1：因子 1.5,1.5,.5,.5 → 末值 .5625，峰值 2.25
        self.assertAlmostEqual(p.snapshot().evidence, 0.5625)
        self.assertAlmostEqual(p.snapshot().max_evidence, 2.25)


class SerializationTests(unittest.TestCase):
    def _played(self):
        p = PredictableBettingProcess(0.5, two_state_strategy(), 0.25)
        p.extend([0.9, 0.2, 1.0, 0.1])
        return p

    def test_export_contains_fingerprint_and_sealed_round(self):
        p = PredictableBettingProcess(0.5, two_state_strategy(), 0.25)
        p.extend([0.9, 0.2])
        p.prepare()  # 封存第 3 轮后再导出
        d = p.to_dict()
        self.assertEqual(d["kind"], "predictable_betting")
        self.assertEqual(d["version"], 1)
        self.assertEqual(d["fingerprint"], p.fingerprint)
        self.assertEqual(len(d["fingerprint"]), 64)
        self.assertEqual(d["sealed"]["round_index"], 3)
        self.assertEqual(d["sealed"]["state"], p.state)
        json.dumps(d)  # 可 JSON 序列化

    def test_round_trip_mid_round_and_observe_after_restore(self):
        p = self._played()
        p.prepare()
        d = p.to_dict()
        r = PredictableBettingProcess.from_dict(
            d, strategy=two_state_strategy(), m=0.5, alpha=0.25
        )
        self.assertEqual(r.state, p.state)
        self.assertEqual(r.t, p.t)
        self.assertEqual(r.sealed_round, p.sealed_round)
        self.assertEqual(r.sealed_bet(), p.sealed_bet())
        r.observe(0.7)
        p.observe(0.7)
        self.assertEqual(r.snapshot(), p.snapshot())
        self.assertEqual(r.log_wealth(), p.log_wealth())

    def test_restore_allows_repeated_observe_only_after_reprepare(self):
        p = self._played()
        r = PredictableBettingProcess.from_json(p.to_json())
        self.assertEqual(r.snapshot(), p.snapshot())
        with self.assertRaises(RuntimeError):
            r.observe(0.5)
        r.prepare()
        r.observe(0.5)
        self.assertEqual(r.t, p.t + 1)

    def test_modified_strategy_fingerprint_rejected(self):
        d = self._played().to_dict()
        # 改策略表里的一个 lambda（试图恢复后换策略）
        tampered = copy.deepcopy(d)
        tampered["strategy"]["bet_lambdas"]["high"] = 1.5
        with self.assertRaises(ValueError):
            PredictableBettingProcess.from_dict(tampered)
        # 传入不同策略对象也拒绝
        other = FiniteBettingStrategy(
            ["low", "high"], "low", {"low": 0.0, "high": 1.5},
            {"low": {0: "low", 1: "high"},
             "high": {0: "low", 1: "high"}},
            [0.0, 0.5, 1.0],
        )
        with self.assertRaises(ValueError):
            PredictableBettingProcess.from_dict(d, strategy=other)

    def test_restore_rejects_bad_m_alpha_and_corruption(self):
        d = self._played().to_dict()
        with self.assertRaises(ValueError):
            PredictableBettingProcess.from_dict(d, m=0.51)
        with self.assertRaises(ValueError):
            PredictableBettingProcess.from_dict(d, alpha=0.1)
        bad_kind = copy.deepcopy(d)
        bad_kind["kind"] = "other"
        with self.assertRaises(ValueError):
            PredictableBettingProcess.from_dict(bad_kind)
        bad_fp = copy.deepcopy(d)
        bad_fp["fingerprint"] = "0" * 64
        with self.assertRaises(ValueError):
            PredictableBettingProcess.from_dict(bad_fp)
        bad_seal = copy.deepcopy(d)
        bad_seal["sealed"] = {"round_index": 99, "state": "low", "lam": 0.0}
        with self.assertRaises(ValueError):
            PredictableBettingProcess.from_dict(bad_seal)

    def test_round_trip_with_dead_wealth(self):
        p = PredictableBettingProcess(0.5, two_state_strategy(), 0.25)
        p.extend([0.9, 0.0])  # 第 2 轮归零
        r = PredictableBettingProcess.from_json(p.to_json())
        self.assertEqual(r.log_wealth(), -math.inf)
        self.assertEqual(r.snapshot(), p.snapshot())
        r.extend([1.0, 1.0])  # 归零不复活
        self.assertEqual(r.log_wealth(), -math.inf)

if __name__ == "__main__":
    unittest.main()
