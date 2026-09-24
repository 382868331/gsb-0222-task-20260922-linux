"""双侧候选均值网格（TwoSidedMeanEvidence）的独立参考与契约测试。

期望值（精确上下尾财富、混合证据、拒绝时刻、存活集合、双侧命中质量）
全部由本文件内独立的 Fraction 有理实现给出；mean_evidence 仅在对照测试中
被导入，期望值不来自被测代码。
"""

from __future__ import annotations

import copy
import math
import unittest
from fractions import Fraction

import mean_evidence as me
from mean_evidence import TwoSidedMeanEvidence


# ---------------------------------------------------------------------------
# 独立精确参考：全程 Fraction，不调用被测核心
# ---------------------------------------------------------------------------

def _fr(x) -> Fraction:
    return Fraction(float(x))


class RefTwoSided:
    """与产品代码算法路径无关的朴素精确双侧参考。"""

    def __init__(self, mus, lambdas, alpha):
        self.mus = [_fr(v) for v in mus]
        self.lams = [_fr(v) for v in lambdas]
        self.alpha = _fr(alpha)
        self.half = self.alpha / 2
        self.k = len(self.lams)
        # 每个候选两套财富（上尾、下尾）与各自历史峰值。
        self.w_up = [[Fraction(1)] * self.k for _ in self.mus]
        self.w_lo = [[Fraction(1)] * self.k for _ in self.mus]
        self.max_up = [Fraction(1) for _ in self.mus]
        self.max_lo = [Fraction(1) for _ in self.mus]
        self.t = 0

    def _mix(self, w):
        return sum(w, Fraction(0)) / self.k

    def update(self, x):
        fx = _fr(x)
        self.t += 1
        thr = Fraction(1) / self.half
        out = []
        for j, mu in enumerate(self.mus):
            for i, lam in enumerate(self.lams):
                self.w_up[j][i] *= Fraction(1) + lam * (fx - mu)
                self.w_lo[j][i] *= Fraction(1) - lam * (fx - mu)
            e_up, e_lo = self._mix(self.w_up[j]), self._mix(self.w_lo[j])
            self.max_up[j] = max(self.max_up[j], e_up)
            self.max_lo[j] = max(self.max_lo[j], e_lo)
            rejected = self.max_up[j] >= thr or self.max_lo[j] >= thr
            out.append((e_up, e_lo, rejected))
        return out

    def surviving(self):
        thr = Fraction(1) / self.half
        return [
            float(mu) for j, mu in enumerate(self.mus)
            if self.max_up[j] < thr and self.max_lo[j] < thr
        ]


def two_sided_hit_mass(mu: Fraction, lams: list[Fraction],
                       alpha: Fraction, n: int) -> Fraction:
    """H0：Bernoulli(p=mu) 下，n 步内任一侧首次越过 2/alpha 的精确概率。"""
    thr = Fraction(2) / alpha
    k = len(lams)
    p, q = mu, Fraction(1) - mu
    hit = Fraction(0)

    def walk(t, w_up, w_lo, prob, already):
        nonlocal hit
        if t == n:
            if already:
                hit += prob
            return
        if already:
            walk(t + 1, w_up, w_lo, prob * p, True)
            walk(t + 1, w_up, w_lo, prob * q, True)
            return

        def step_up(w, x):
            return [w[i] * (Fraction(1) + lams[i] * (x - mu)) for i in range(k)]

        def step_lo(w, x):
            return [w[i] * (Fraction(1) - lams[i] * (x - mu)) for i in range(k)]

        up1, lo1 = step_up(w_up, Fraction(1)), step_lo(w_lo, Fraction(1))
        up0, lo0 = step_up(w_up, Fraction(0)), step_lo(w_lo, Fraction(0))
        cross1 = (sum(up1) / k >= thr) or (sum(lo1) / k >= thr)
        cross0 = (sum(up0) / k >= thr) or (sum(lo0) / k >= thr)
        walk(t + 1, up1, lo1, prob * p, cross1)
        walk(t + 1, up0, lo0, prob * q, cross0)

    one = [Fraction(1)] * k
    walk(0, one, [Fraction(1) for _ in lams], Fraction(1), False)
    return hit


def lcg_values(n: int, seed: int = 20260924) -> list[float]:
    out = []
    state = seed
    for _ in range(n):
        state = (1103515245 * state + 12345) % (2 ** 31)
        out.append(0.05 + 0.9 * (state / (2 ** 31)))  # 落在 (0.05, 0.95)
    return out


# ---------------------------------------------------------------------------
# 手算小例子（期望值字面量手算，不查 oracle）
# ---------------------------------------------------------------------------

class HandComputedTests(unittest.TestCase):
    def test_threshold_is_two_over_alpha(self):
        p = TwoSidedMeanEvidence([0.5], [0.0, 2.0], 0.5)
        self.assertEqual(p.alpha_per_side, 0.25)
        self.assertEqual(p.threshold, 4.0)
        s = p.snapshot()
        self.assertEqual(s.threshold, 4.0)
        self.assertEqual(s.alpha_per_side, 0.25)
        self.assertEqual(s.t, 0)
        self.assertEqual(s.surviving, (0.5,))
        self.assertEqual(s.rejected, ())

    def test_all_ones_hand_computed(self):
        # mu=1/2, lams={0,2}, alpha=1/2, 每侧阈值 4。
        # 全 1 流：上尾 W=(1, 2^t)，E+ = (1+2^t)/2；下尾 λ=2 轨道首步因子
        # 1-2(1-1/2)=0 永久归零，E-=1/2。
        # t=1: E+=3/2；t=2: 5/2；t=3: 9/2 ≥ 4 → 上尾首次拒绝于 t=3。
        p = TwoSidedMeanEvidence([0.5], [0.0, 2.0], 0.5)
        s1 = p.update(1)
        c = s1.candidates[0]
        self.assertAlmostEqual(c.upper_evidence, 1.5)
        self.assertAlmostEqual(c.lower_evidence, 0.5)
        self.assertFalse(c.rejected)
        s2 = p.update(1)
        self.assertAlmostEqual(s2.candidates[0].upper_evidence, 2.5)
        s3 = p.update(1)
        c3 = s3.candidates[0]
        self.assertAlmostEqual(c3.upper_evidence, 4.5)
        self.assertTrue(c3.rejected)
        self.assertEqual(c3.reject_time, 3)
        self.assertEqual(c3.reject_side, "upper")
        self.assertEqual(s3.surviving, ())
        self.assertEqual(s3.rejected, (0.5,))
        # 下尾轨道首步即归零且不复活
        up_lw, lo_lw = p.candidate_log_wealth(0)
        self.assertEqual(up_lw, (0.0, 3 * math.log(2)))
        self.assertEqual(lo_lw, (0.0, -math.inf))

    def test_all_zeros_hand_computed_lower_reject(self):
        # 镜像：全 0 流，下尾 W-=(1,2^t)，t=3 时下尾拒绝。
        p = TwoSidedMeanEvidence([0.5], [0.0, 2.0], 0.5)
        s = p.extend([0, 0, 0])
        c = s.candidates[0]
        self.assertAlmostEqual(c.lower_evidence, 4.5)
        self.assertAlmostEqual(c.upper_evidence, 0.5)
        self.assertEqual(c.reject_time, 3)
        self.assertEqual(c.reject_side, "lower")

    def test_two_sided_p_value_relation(self):
        # p_双侧 = min(1, 2 min(p+, p-))；拒绝当且仅当 p_双侧 ≤ alpha。
        p = TwoSidedMeanEvidence([0.5], [0.0, 2.0], 0.5)
        p.extend([1, 1, 1])
        c = p.snapshot().candidates[0]
        # max E+ = 4.5 → p+ = 2/9，2 p+ = 4/9
        self.assertAlmostEqual(c.upper_p_value, 2.0 / 9.0)
        self.assertAlmostEqual(c.p_value_two_sided, 4.0 / 9.0)
        self.assertLessEqual(c.p_value_two_sided, 0.5)

    def test_both_tails_bankrupt_never_rejected(self):
        # 序列 [0,1]：x=0 杀死上尾 λ=2 轨道（因子 1-2·1/2=0），
        # x=1 杀死下尾 λ=2 轨道（因子 1-2·1/2=0）。
        # 两侧 E 都只剩 λ=0 轨道的一半 → E+=E-=1/2，永不越过 4。
        p = TwoSidedMeanEvidence([0.5], [0.0, 2.0], 0.5)
        s = p.extend([0, 1])
        c = s.candidates[0]
        self.assertAlmostEqual(c.upper_evidence, 0.5)
        self.assertAlmostEqual(c.lower_evidence, 0.5)
        self.assertFalse(c.rejected)
        self.assertIsNone(c.reject_time)
        self.assertEqual(s.surviving, (0.5,))
        up_lw, lo_lw = p.candidate_log_wealth(0)
        self.assertEqual(up_lw, (0.0, -math.inf))
        self.assertEqual(lo_lw, (0.0, -math.inf))
        # 之后任意观测财富不再复活
        s = p.extend([1, 1, 0])
        up_lw, lo_lw = p.candidate_log_wealth(0)
        self.assertEqual(up_lw, (0.0, -math.inf))
        self.assertEqual(lo_lw, (0.0, -math.inf))
        self.assertEqual(s.surviving, (0.5,))

    def test_rejection_is_record_only_keeps_first_time(self):
        # 5 个 1 使上尾越过（grid {0,1}：E+=(1+1.5^t)/2，t=4 时≈3.03<4，
        # t=5 时≈4.30≥4 → 首次 t=5）；随后 14 个 0 下尾也越过，
        # reject_time/side 永远是首次（上尾 t=5）。
        p = TwoSidedMeanEvidence([0.5], [0.0, 1.0], 0.5)
        p.extend([1] * 5)
        self.assertEqual(p.snapshot().candidates[0].reject_time, 5)
        p.extend([0] * 14)
        c = p.snapshot().candidates[0]
        self.assertEqual(c.reject_time, 5)
        self.assertEqual(c.reject_side, "upper")
        self.assertLessEqual(c.lower_p_value, 0.25)


class DegenerateBettingTests(unittest.TestCase):
    def test_lambda_zero_only_never_rejects(self):
        # 退化下注：网格只有 λ=0，所有财富恒为 1，E≡1 < 2/alpha，永不拒绝。
        p = TwoSidedMeanEvidence([0.2, 0.5, 0.8], [0.0], 0.05)
        s = p.extend([1] * 50 + [0] * 50)
        for c in s.candidates:
            self.assertEqual(c.upper_evidence, 1.0)
            self.assertEqual(c.lower_evidence, 1.0)
            self.assertEqual(c.upper_p_value, 1.0)
            self.assertEqual(c.lower_p_value, 1.0)
            self.assertEqual(c.p_value_two_sided, 1.0)
            self.assertFalse(c.rejected)
        self.assertEqual(s.surviving, (0.2, 0.5, 0.8))
        self.assertEqual(s.rejected, ())

    def test_single_lambda_grid_is_degenerate_mixture(self):
        # 单元素网格：混合就是一条轨道；mu=0.5, λ=2, 全 1。
        # 无 λ=0 稀释，W+=2^t：t=2 时 W+=4 恰好达到阈值即拒绝（t=1 时 2<4）。
        p = TwoSidedMeanEvidence([0.5], [2.0], 0.5)
        p.update(1)
        self.assertIsNone(p.snapshot().candidates[0].reject_time)
        s = p.extend([1, 1])
        c = s.candidates[0]
        self.assertAlmostEqual(c.upper_evidence, 8.0)
        self.assertEqual(c.reject_time, 2)

    def test_duplicate_mus_counted(self):
        p = TwoSidedMeanEvidence([0.5, 0.5], [0.0, 2.0], 0.5)
        s = p.extend([1, 1, 1])
        self.assertEqual(len(s.candidates), 2)
        self.assertEqual(s.rejected, (0.5, 0.5))
        self.assertEqual(s.surviving, ())


class CandidateGridBehaviorTests(unittest.TestCase):
    def test_lambda_cap_uses_extreme_candidates(self):
        # cap = min_i min(1/mu_i, 1/(1-mu_i))；
        # mus={.1,.5,.9} → 端点候选给出 1/0.9 = 1.111...
        cap = 1.0 / 0.9
        p = TwoSidedMeanEvidence([0.1, 0.5, 0.9], [0.0, cap], 0.1)
        self.assertEqual(len(p.lambdas), 2)
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence([0.1, 0.5, 0.9], [cap + 1e-12], 0.1)
        # 只有内部候选时 cap 可以更大：mu=0.5 单候选允许 λ=2
        TwoSidedMeanEvidence([0.5], [2.0], 0.1)

    def test_high_stream_rejects_only_low_candidates_via_upper(self):
        # 持续高观测（20 个 0.9）：低候选被上尾拒绝，最高候选 0.8 此样本量下
        # 证据尚不足而存活；结论仅限网格点。手算轨道临界（grid {0,1.25}，
        # 混合 E+=(1+(1+1.25(0.9-μ))^t)/2 ≥ 40 要求单轨 ≥79）：
        # μ=.2 → t≈7；μ=.4 → t≈9；μ=.6 → t≈14；μ=.8 → t≈38。
        mus = [0.2, 0.4, 0.6, 0.8]
        p = TwoSidedMeanEvidence(mus, [0.0, 1.25], 0.05)
        s = p.extend([0.9] * 20)
        for c in s.candidates:
            if c.mu in s.rejected:
                self.assertEqual(c.reject_side, "upper")
        self.assertEqual(s.rejected, (0.2, 0.4, 0.6))
        self.assertEqual(s.surviving, (0.8,))
        # 网格之外（如 0.3）在结果中根本不存在——这是网格限制
        self.assertTrue(all(mu in mus for mu in s.surviving + s.rejected))

    def test_low_stream_rejects_only_high_candidates_via_lower(self):
        # 镜像：20 个 0.1，下尾单轨因子 1+1.25(μ-0.1)，临界与上尾对称：
        # μ=.8 → t≈7；μ=.6 → t≈9；μ=.4 → t≈14；μ=.2 → t≈38。
        # 故 0.4、0.6、0.8 被下尾拒绝，仅 0.2 存活。
        mus = [0.2, 0.4, 0.6, 0.8]
        p = TwoSidedMeanEvidence(mus, [0.0, 1.25], 0.05)
        s = p.extend([0.1] * 20)
        for c in s.candidates:
            if c.mu in s.rejected:
                self.assertEqual(c.reject_side, "lower")
        self.assertEqual(s.rejected, (0.4, 0.6, 0.8))
        self.assertEqual(s.surviving, (0.2,))

    def test_surviving_monotone_shrink_and_ordered(self):
        # 全 1 流，grid {0,1}，阈值 40：μ=.3 于 t≈9、μ=.5 于 t≈11 被上尾拒绝，
        # μ=.7 在 t=13 时仍存活（1.3^13≈30，混合≈16 < 40）。
        mus = [0.3, 0.5, 0.7]
        p = TwoSidedMeanEvidence(mus, [0.0, 1.0], 0.05)
        prev = set(mus)
        for x in [1] * 13:
            s = p.update(x)
            cur = set(s.surviving)
            self.assertTrue(cur <= prev)  # 存活集合只缩不增
            # 存活/拒绝元组保持构造顺序
            self.assertEqual(list(s.surviving), sorted(s.surviving, key=mus.index))
            self.assertEqual(list(s.rejected), sorted(s.rejected, key=mus.index))
            prev = cur
        self.assertEqual(p.snapshot().surviving, (0.7,))
        self.assertEqual(p.snapshot().rejected, (0.3, 0.5))


class IndependentFractionOracleTests(unittest.TestCase):
    def test_matches_fraction_oracle_random_stream(self):
        mus = [0.25, 0.5, 0.75]
        grid = [0.0, 1.0, 4.0 / 3.0]  # 4/3 = cap（端点候选决定）
        alpha = 0.2
        xs = lcg_values(50)
        ref = RefTwoSided(mus, grid, alpha)
        proc = TwoSidedMeanEvidence(mus, grid, alpha)
        for x in xs:
            ref_out = ref.update(x)
            snap = proc.update(x)
            self.assertEqual(snap.t, ref.t)
            for j, c in enumerate(snap.candidates):
                e_up, e_lo, rej = ref_out[j]
                for got, exp in ((c.upper_log_evidence, math.log(float(e_up))),
                                 (c.lower_log_evidence, math.log(float(e_lo)))):
                    self.assertLess(
                        abs(got - exp), 1e-12 * max(1.0, abs(exp)),
                        msg=f"t={ref.t} mu={mus[j]} got={got} ref={exp}",
                    )
                self.assertAlmostEqual(
                    c.max_upper_log_evidence,
                    math.log(float(ref.max_up[j])),
                    delta=1e-12 * max(1.0, abs(math.log(float(ref.max_up[j])))),
                )
                self.assertAlmostEqual(
                    c.max_lower_log_evidence,
                    math.log(float(ref.max_lo[j])),
                    delta=1e-12 * max(1.0, abs(math.log(float(ref.max_lo[j])))),
                )
                p_up = min(1.0, 1.0 / float(ref.max_up[j]))
                p_lo = min(1.0, 1.0 / float(ref.max_lo[j]))
                self.assertAlmostEqual(c.upper_p_value, p_up, delta=1e-12)
                self.assertAlmostEqual(c.lower_p_value, p_lo, delta=1e-12)
                self.assertAlmostEqual(
                    c.p_value_two_sided, min(1.0, 2 * min(p_up, p_lo)),
                    delta=1e-12,
                )
                self.assertEqual(c.rejected, rej)
            self.assertEqual(
                tuple(round(v, 12) for v in snap.surviving),
                tuple(round(v, 12) for v in ref.surviving()),
            )

    def test_bernoulli_enumeration_two_sided_hit_mass_tight(self):
        # H0：p=mu=1/2，单轨道 λ=2，总 α=1/2（每侧 1/4，阈值 4）。
        # 上尾命中要求前两步全 1（质量 1/4），下尾要求全 0（1/4），
        # 两事件互斥，联合命中质量恰为 1/2 = α（Bonferroni 界紧）。
        mass = two_sided_hit_mass(Fraction(1, 2), [Fraction(2)],
                                  Fraction(1, 2), n=5)
        self.assertEqual(mass, Fraction(1, 2))

    def test_bernoulli_enumeration_hit_mass_le_alpha(self):
        for mu in (Fraction(1, 2), Fraction(2, 5)):
            cap = min(Fraction(1) / mu, Fraction(1) / (1 - mu))
            for lam in (Fraction(0), cap, cap / 2):
                for alpha in (Fraction(1, 4), Fraction(1, 10)):
                    with self.subTest(mu=mu, lam=lam, alpha=alpha):
                        mass = two_sided_hit_mass(mu, [lam], alpha, n=5)
                        self.assertLessEqual(mass, alpha)


class ValidationTests(unittest.TestCase):
    def test_bad_mus(self):
        for bad in ([], [0.0], [1.0], [0.5, -0.1], [0.5, 1.0000001]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    TwoSidedMeanEvidence(bad, [1.0], 0.1)
        for bad in ([True], [0.5, float("nan")], "05", 0.5, None):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    TwoSidedMeanEvidence(bad, [1.0], 0.1)

    def test_bad_lambdas(self):
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence([0.5], [], 0.1)
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence([0.5], [-1e-16], 0.1)
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence([0.5], [float("inf")], 0.1)
        with self.assertRaises(TypeError):
            TwoSidedMeanEvidence([0.5], 2.0, 0.1)
        with self.assertRaises(TypeError):
            TwoSidedMeanEvidence([0.5], "01", 0.1)

    def test_bad_alpha(self):
        for bad in (0, 1, -0.1, 2, float("nan"), True):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    TwoSidedMeanEvidence([0.5], [1.0], bad)

    def test_bad_observations_and_atomicity(self):
        # mus={.3,.6} 的公共 λ 上界 = min(1/.7)≈1.4286，取 1.4 合法。
        p = TwoSidedMeanEvidence([0.3, 0.6], [0.0, 1.4], 0.1)
        p.update(0.7)
        before = p.to_dict()
        for bad in (1.5, -0.1, float("nan"), float("inf"), True, "0.5", None):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    p.update(bad)
                self.assertEqual(p.t, 1)
        # 批量：第 2 个元素非法 → 整批拒绝，状态字节级不变
        with self.assertRaises(ValueError):
            p.extend([0.4, 1.2, 0.5])
        self.assertEqual(p.to_dict(), before)
        with self.assertRaises(TypeError):
            p.extend([0.4, False, 0.5])
        self.assertEqual(p.to_dict(), before)
        # 合法批次正常推进
        s = p.extend([0.4, 0.5])
        self.assertEqual(s.t, 3)

    def test_candidate_log_wealth_index_check(self):
        p = TwoSidedMeanEvidence([0.5], [0.0, 2.0], 0.5)
        p.update(0)
        self.assertEqual(len(p.candidate_log_wealth(0)), 2)
        for bad in (1, -1, True):
            with self.assertRaises(IndexError):
                p.candidate_log_wealth(bad)

    def test_grids_copied_defensively(self):
        mus, grid = [0.4, 0.6], [0.0, 1.5]
        p = TwoSidedMeanEvidence(mus, grid, 0.1)
        mus.append(0.5)
        grid.append(9.0)
        self.assertEqual(len(p.mus), 2)
        self.assertEqual(len(p.lambdas), 2)


class SerializationTests(unittest.TestCase):
    def _streamed(self):
        p = TwoSidedMeanEvidence([0.2, 0.5, 0.8], [0.0, 1.25], 0.07)
        p.extend([0.95, 0.1, 0.7, 0.0, 1.0, 0.33, 0.8])
        return p

    def test_round_trip_and_continue(self):
        p = self._streamed()
        import json
        text = p.to_json()
        json.loads(text)
        r = TwoSidedMeanEvidence.from_json(text)
        self.assertEqual(r.snapshot(), p.snapshot())
        self.assertEqual(r.mus, p.mus)
        self.assertEqual(r.lambdas, p.lambdas)
        self.assertEqual(r.alpha, p.alpha)
        tail = [0.2, 0.9, 0.5, 0.1]
        p.extend(tail)
        r.extend(tail)
        self.assertEqual(r.snapshot(), p.snapshot())
        for i in range(len(p.mus)):
            self.assertEqual(r.candidate_log_wealth(i), p.candidate_log_wealth(i))

    def test_round_trip_at_t0(self):
        p = TwoSidedMeanEvidence([0.3, 0.7], [0.0, 1.4], 0.01)
        r = TwoSidedMeanEvidence.from_json(p.to_json())
        self.assertEqual(r.snapshot(), p.snapshot())

    def test_corrupted_states_rejected(self):
        d = self._streamed().to_dict()

        bad_kind = copy.deepcopy(d)
        bad_kind["kind"] = "other"
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence.from_dict(bad_kind)

        bad_version = copy.deepcopy(d)
        bad_version["version"] = 99
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence.from_dict(bad_version)

        bad_mus = copy.deepcopy(d)
        bad_mus["mus"].append(0.99)
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence.from_dict(bad_mus)

        bad_mu_value = copy.deepcopy(d)
        bad_mu_value["mus"][0] = 0.0
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence.from_dict(bad_mu_value)

        bad_len = copy.deepcopy(d)
        bad_len["upper"].append(bad_len["upper"][0])
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence.from_dict(bad_len)

        # 篡改子过程 alpha（破坏 alpha/2 平分）
        bad_half = copy.deepcopy(d)
        bad_half["upper"][0]["alpha"] = 0.07
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence.from_dict(bad_half)

        # 篡改某个子过程的 t（上下尾失步）
        bad_t = copy.deepcopy(d)
        bad_t["lower"][1]["t"] = 0
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence.from_dict(bad_t)

        # 篡改候选镜像 m：lower 子过程 m 必须为 1-mu
        bad_mirror = copy.deepcopy(d)
        bad_mirror["lower"][0]["m"] = 0.2
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence.from_dict(bad_mirror)

        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence.from_json("{broken")
        with self.assertRaises(TypeError):
            TwoSidedMeanEvidence.from_json(123)

    def test_legacy_one_sided_state_not_accepted_here(self):
        # 单侧 version=1 状态没有 kind 标识，不能被双侧恢复入口接受。
        one = me.MeanEvidenceProcess(0.5, [0.0, 2.0], 0.25).to_dict()
        with self.assertRaises(ValueError):
            TwoSidedMeanEvidence.from_dict(one)


if __name__ == "__main__":
    unittest.main()
