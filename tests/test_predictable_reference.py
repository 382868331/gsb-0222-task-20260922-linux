"""PredictableBettingProcess 的独立 Fraction 参考测试。

期望值（手算二状态序列、固定 lambda 退化对照、自适应策略 Bernoulli 路径
枚举）全部由本文件内的精确有理数实现给出，不调用被测核心产生期望值。
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

from mean_evidence import (
    FiniteBettingStrategy,
    MeanEvidenceProcess,
    PredictableBettingProcess,
)


def fr(x):
    return Fraction(float(x))


# ---------------------------------------------------------------------------
# 手算：二状态策略（m=1/2，切点 1/2，low->lambda=0，high->lambda=2）
# ---------------------------------------------------------------------------

TWO_STATE = {
    "m": Fraction(1, 2),
    "cut": Fraction(1, 2),
    "lam": {"low": Fraction(0), "high": Fraction(2)},
}


def hand_two_state(xs):
    """逐事件手算：每轮先按当前状态封存 lambda，再吃 x、按下半开区间换状态。

    返回 (每轮封存lambda列表, 每轮财富Fraction, 最终状态)。
    """
    m, cut, lams = TWO_STATE["m"], TWO_STATE["cut"], TWO_STATE["lam"]
    state = "low"
    wealth = Fraction(1)
    sealed = []
    wealths = []
    for x in xs:
        lam = lams[state]          # 封存：只依赖历史
        sealed.append(lam)
        fx = Fraction(x)
        wealth *= Fraction(1) + lam * (fx - m)
        b = 1 if fx >= cut else 0  # 半开区间 [cut,1]：x=1/2 归高档
        state = "high" if b == 1 else "low"
        wealths.append(wealth)
    return sealed, wealths, state


class HandComputedTwoStateTests(unittest.TestCase):
    def _proc(self):
        strat = FiniteBettingStrategy(
            ["low", "high"], "low", {"low": 0.0, "high": 2.0},
            {"low": {0: "low", 1: "high"},
             "high": {0: "low", 1: "high"}},
            [0.0, 0.5, 1.0],
        )
        return PredictableBettingProcess(0.5, strat, 0.25)

    def test_event_by_event_sealing_matches_hand(self):
        xs = [0, 1, 0, 1, 1]
        sealed, wealths, final_state = hand_two_state(xs)
        # 手算封存序列：low 起 -> 0；x0=0 仍 low -> 0；x1=1 进 high -> 2；
        # x2=0 回 low -> 0；x3=1 进 high -> 2
        self.assertEqual(sealed, [Fraction(0), Fraction(0), Fraction(2),
                                  Fraction(0), Fraction(2)])
        proc = self._proc()
        for i, x in enumerate(xs):
            bet = proc.prepare()
            # 封存值与手算一致，且封存发生在 observe 前
            self.assertEqual(Fraction(bet.lam), sealed[i])
            self.assertEqual(bet.round_index, i + 1)
            snap = proc.observe(x)
            self.assertAlmostEqual(snap.log_evidence,
                                   math.log(float(wealths[i])) if wealths[i] > 0
                                   else -math.inf, places=12)
        self.assertEqual(proc.state, final_state)
        # 财富：t3 在 high 以 lambda=2 吃 x=0 归零，之后恒 0
        self.assertEqual(proc.log_wealth(), -math.inf)

    def test_x_one_goes_to_high_bin(self):
        proc = self._proc()
        proc.prepare()
        proc.observe(1)  # x=1 必须落入 [1/2,1] 高档
        self.assertEqual(proc.state, "high")
        proc.prepare()
        snap = proc.observe(1)  # high 封存 lambda=2，因子 1+2(1-.5)=2，W=2
        self.assertAlmostEqual(snap.evidence, 2.0)

    def test_cutpoint_half_is_high_bin(self):
        proc = self._proc()
        proc.prepare()
        proc.observe(0.5)  # 恰为切点，左闭 -> 高档
        self.assertEqual(proc.state, "high")
        self.assertEqual(proc.prepare().lam, 2.0)


# ---------------------------------------------------------------------------
# 固定策略退化：单状态策略必须与固定 lambda 单轨道完全一致；
# Fraction 乘积核对小序列
# ---------------------------------------------------------------------------

class FixedDegenerateTests(unittest.TestCase):
    def test_matches_fixed_grid_single_track_fraction(self):
        m, lam, alpha = 0.3, 2.0, 0.2
        xs = [1, 0, 0, 0.8, 0.2, 1]
        strat = FiniteBettingStrategy(
            ["s"], "s", {"s": lam}, {"s": {0: "s"}}, [0.0, 1.0]
        )
        pred = PredictableBettingProcess(m, strat, alpha)
        fixed = MeanEvidenceProcess(m, [lam], alpha)
        w = Fraction(1)
        mf = Fraction(float(m))
        lf = Fraction(float(lam))
        for x in xs:
            pred.prepare()
            snap = pred.observe(x)
            fsnap = fixed.update(float(x))
            w *= Fraction(1) + lf * (Fraction(float(x)) - mf)
            self.assertAlmostEqual(snap.log_evidence, math.log(float(w)),
                                   places=12)
            self.assertEqual(snap.evidence, fsnap.evidence)
            self.assertEqual(snap.stop_time, fsnap.stop_time)
            self.assertEqual(snap.max_evidence, fsnap.max_evidence)

    def test_small_fraction_product_exact(self):
        # m=1/2，lambda=1（固定退化），x = 1,0,1：因子 3/2, 1/2, 3/2 → 9/8
        strat = FiniteBettingStrategy(
            ["s"], "s", {"s": 1.0}, {"s": {0: "s"}}, [0.0, 1.0]
        )
        proc = PredictableBettingProcess(0.5, strat, 0.1)
        for x in (1, 0, 1):
            proc.run_round(x)
        self.assertAlmostEqual(proc.snapshot().evidence, float(Fraction(9, 8)))
        # 峰值出现在 t=1 与 t=3 均为 3/2（t2 回落到 3/4）
        self.assertAlmostEqual(proc.snapshot().max_evidence, float(Fraction(3, 2)))
        self.assertAlmostEqual(proc.snapshot().p_value, float(Fraction(2, 3)))


# ---------------------------------------------------------------------------
# 自适应策略 Bernoulli 路径精确枚举：H0（p=m=1/2）下 Ville 命中质量 <= alpha
# 策略与 TWO_STATE 相同：low 封存 0，high 封存 2，切点 1/2
# ---------------------------------------------------------------------------


def adaptive_hit_mass(n: int, alpha: Fraction) -> Fraction:
    """精确枚举 2^n 路径，按封存协议推进状态机与财富，统计首次命中质量。"""
    m = Fraction(1, 2)
    threshold = Fraction(1) / alpha
    hit = Fraction(0)

    def walk(t, state, wealth, prob, already_hit):
        nonlocal hit
        if t == n:
            if already_hit:
                hit += prob
            return
        lam = Fraction(0) if state == "low" else Fraction(2)
        half = prob / 2  # p=1/2 Bernoulli
        for bit, nxt_prob in ((1, half), (0, half)):
            w2 = wealth * (Fraction(1) + lam * (Fraction(bit) - m))
            s2 = "high" if bit == 1 else "low"
            walk(t + 1, s2, w2, nxt_prob,
                 already_hit or (w2 >= threshold))

    walk(0, "low", Fraction(1), Fraction(1), False)
    return hit


class AdaptiveVilleEnumerationTests(unittest.TestCase):
    def test_hit_mass_le_alpha(self):
        for n in (1, 2, 3, 4, 6, 8):
            for alpha in (Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)):
                with self.subTest(n=n, alpha=alpha):
                    self.assertLessEqual(adaptive_hit_mass(n, alpha), alpha)

    def test_tight_bound_small_case(self):
        # alpha=1/4，阈值 4：只有封存了 lambda=2（即上一轮为 1）且连续两个
        # 因子 2 的路径前缀 11（在 t=2,t=3）才达到 4……逐一枚举核对命中质量。
        # 直接断言若干 n 的精确有理数，且 <= alpha。
        self.assertEqual(adaptive_hit_mass(1, Fraction(1, 4)), Fraction(0))
        self.assertEqual(adaptive_hit_mass(2, Fraction(1, 4)), Fraction(0))
        # n=3：唯一命中路径 111（t3 财富 4），质量 1/8 <= 1/4
        self.assertEqual(adaptive_hit_mass(3, Fraction(1, 4)), Fraction(1, 8))

    def test_core_stop_events_match_exact_enumeration(self):
        # 同一策略、alpha=1/4，小确定性序列上核心 stop_time 与精确枚举一致
        strat = FiniteBettingStrategy(
            ["low", "high"], "low", {"low": 0.0, "high": 2.0},
            {"low": {0: "low", 1: "high"},
             "high": {0: "low", 1: "high"}},
            [0.0, 0.5, 1.0],
        )
        threshold = Fraction(4)
        cases = [
            ([1, 1, 1], 3),   # W: 1,2,4
            ([1, 0, 1, 1], None),  # t2 归零，永不命中
            ([0, 1, 1, 1], 4),    # W: 1,1,2,4 首次 t4
        ]
        for xs, want in cases:
            proc = PredictableBettingProcess(0.5, strat, 0.25)
            w = Fraction(1)
            state = "low"
            stop = None
            for i, x in enumerate(xs, start=1):
                lam = Fraction(0) if state == "low" else Fraction(2)
                proc.prepare()
                snap = proc.observe(x)
                w *= Fraction(1) + lam * (Fraction(x) - Fraction(1, 2))
                state = "high" if x >= 0.5 else "low"
                if stop is None and w >= threshold:
                    stop = i
                self.assertEqual(snap.stop_time, stop, msg=f"xs={xs} t={i}")
            self.assertEqual(stop, want)


if __name__ == "__main__":
    unittest.main()
