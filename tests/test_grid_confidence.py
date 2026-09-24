"""有限均值网格双侧置信集测试：独立 Fraction 参考 + 契约边界。

期望值（拒绝时刻、log 证据、拒绝概率质量）由本文件内的独立有理数
实现给出：RefTwoSided 直接按定义计算上尾因子 1+λ(x-m) 与下尾因子
1+λ(m-x)，不调用被测核心 mean_evidence 产生期望值；mean_evidence
仅在对照测试中被导入。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import unittest
from fractions import Fraction

import mean_evidence as me
from mean_evidence import ConfidenceSnapshot, MeanGridConfidenceSet


# ---------------------------------------------------------------------------
# 独立参考实现：全程 Fraction 精确有理运算（不来自被测核心）
# ---------------------------------------------------------------------------

def _fr(x) -> Fraction:
    return Fraction(float(x))


class RefTwoSided:
    """按定义直接计算的精确双侧参考。

    上尾财富 U_j = ∏(1 + λ(x_t - m_j))，下尾财富 L_j = ∏(1 + λ(m_j - x_t))，
    各取 λ 网格均匀混合；任一侧混合财富 ≥ 2/alpha 即拒绝该候选。
    """

    def __init__(self, candidates, lambdas, alpha):
        self.cands = [_fr(c) for c in candidates]
        self.lams = [_fr(v) for v in lambdas]
        self.threshold = Fraction(2) / _fr(alpha)
        self.upper = [[Fraction(1) for _ in self.lams] for _ in self.cands]
        self.lower = [[Fraction(1) for _ in self.lams] for _ in self.cands]
        self.t = 0
        self.rej_time = [None] * len(self.cands)

    def update(self, x):
        fx = _fr(x)
        self.t += 1
        k = len(self.lams)
        for j, m in enumerate(self.cands):
            for i, lam in enumerate(self.lams):
                self.upper[j][i] *= 1 + lam * (fx - m)
                self.lower[j][i] *= 1 + lam * (m - fx)
            if self.rej_time[j] is None:
                eu = sum(self.upper[j], Fraction(0)) / k
                el = sum(self.lower[j], Fraction(0)) / k
                if eu >= self.threshold or el >= self.threshold:
                    self.rej_time[j] = self.t

    def log_upper(self, j) -> float:
        e = sum(self.upper[j], Fraction(0)) / len(self.lams)
        return -math.inf if e == 0 else math.log(float(e))

    def log_lower(self, j) -> float:
        e = sum(self.lower[j], Fraction(0)) / len(self.lams)
        return -math.inf if e == 0 else math.log(float(e))


def lcg_values(n: int, seed: int, lo: float, hi: float) -> list[float]:
    """仓库内确定性 LCG，取 [lo, hi)。"""
    out = []
    state = seed
    for _ in range(n):
        state = (1103515245 * state + 12345) % (2 ** 31)
        out.append(lo + (hi - lo) * (state / (2 ** 31)))
    return out


# ---------------------------------------------------------------------------
# 构造与快照基础
# ---------------------------------------------------------------------------

class BasicsTests(unittest.TestCase):
    def test_initial_snapshot(self):
        cs = MeanGridConfidenceSet([0.3, 0.5], [0.0, 1.0], 0.1)
        s = cs.snapshot()
        self.assertEqual(s.t, 0)
        self.assertEqual(s.candidates, (0.3, 0.5))
        self.assertEqual(s.rejected, ())
        self.assertEqual(s.confidence_set, (0.3, 0.5))
        self.assertEqual(s.upper_log_evidence, (0.0, 0.0))
        self.assertEqual(s.lower_log_evidence, (0.0, 0.0))
        self.assertEqual(s.rejection_times, (None, None))
        self.assertEqual(cs.confidence_set(), (0.3, 0.5))

    def test_alpha_split_equally(self):
        # 双侧 alpha 固定平分：每侧 alpha/2，每侧阈值 2/alpha。
        cs = MeanGridConfidenceSet([0.5], [1.0], 0.1)
        s = cs.snapshot()
        self.assertEqual(s.alpha, 0.1)
        self.assertEqual(s.alpha_side, 0.05)
        self.assertEqual(cs.alpha_side, 0.05)
        self.assertAlmostEqual(s.threshold, 20.0)

    def test_snapshot_is_frozen(self):
        cs = MeanGridConfidenceSet([0.5], [1.0], 0.1)
        s = cs.update(1)
        self.assertIsInstance(s, ConfidenceSnapshot)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.t = 9  # type: ignore[misc]

    def test_exact_threshold_hit_counts_as_rejection(self):
        # m=1/2, λ=2, alpha=1/2 → 每侧阈值 4；两个 1 使上尾财富恰为 4。
        # 全为 2 的幂，浮点精确，不存在舍入歧义。
        cs = MeanGridConfidenceSet([0.5], [2.0], 0.5)
        s1 = cs.update(1)
        self.assertEqual(s1.confidence_set, (0.5,))
        s2 = cs.update(1)
        self.assertEqual(s2.confidence_set, ())
        self.assertEqual(s2.rejected, (0.5,))
        self.assertEqual(s2.rejection_times, (2,))

    def test_rejection_is_sticky(self):
        # 拒绝不可逆：之后喂入支持该候选的数据也不恢复。
        cs = MeanGridConfidenceSet([0.5], [2.0], 0.5)
        cs.extend([1, 1])  # t=2 拒绝
        s = cs.extend([0, 0, 0])
        self.assertEqual(s.rejected, (0.5,))
        self.assertEqual(s.rejection_times, (2,))
        self.assertEqual(s.confidence_set, ())


# ---------------------------------------------------------------------------
# 固定下注退化（lambda 全为 0）
# ---------------------------------------------------------------------------

class DegenerateBetTests(unittest.TestCase):
    def test_zero_lambda_grid_never_rejects(self):
        # λ 全 0：所有因子恒为 1，财富恒 1，任何极端序列都不拒绝。
        cs = MeanGridConfidenceSet([0.1, 0.5, 0.9], [0.0], 0.01)
        s = cs.extend([1] * 200 + [0] * 200)
        self.assertEqual(s.confidence_set, (0.1, 0.5, 0.9))
        self.assertEqual(s.rejected, ())
        self.assertEqual(s.upper_log_evidence, (0.0, 0.0, 0.0))
        self.assertEqual(s.lower_log_evidence, (0.0, 0.0, 0.0))
        self.assertEqual(s.rejection_times, (None, None, None))

    def test_zero_lambda_track_keeps_evidence_positive(self):
        # 混合网格含 λ=0 轨道：其余轨道归零后混合证据仍 > 0。
        cs = MeanGridConfidenceSet([0.5], [0.0, 2.0], 0.05)
        s = cs.update(0)  # λ=2 上尾轨道归零
        self.assertTrue(math.isfinite(s.upper_log_evidence[0]))
        # 上尾混合 E = (1 + 0)/2 = 0.5
        self.assertAlmostEqual(s.upper_log_evidence[0], math.log(0.5))


# ---------------------------------------------------------------------------
# 财富归零
# ---------------------------------------------------------------------------

class ZeroWealthTests(unittest.TestCase):
    def test_upper_wealth_zero_is_permanent(self):
        # m=1/2, λ=2=1/m：x=0 时上尾因子 1+2(0-1/2)=0，上尾财富永久归零；
        # 但下尾财富每步乘 2，候选仍可由下尾拒绝（t=5：2^5=32 ≥ 20）。
        cs = MeanGridConfidenceSet([0.5], [2.0], 0.1)
        for _ in range(4):
            s = cs.update(0)
        self.assertEqual(s.upper_log_evidence, (-math.inf,))
        self.assertEqual(s.confidence_set, (0.5,))
        s = cs.update(0)  # t=5
        self.assertEqual(s.upper_log_evidence, (-math.inf,))  # 归零不可复活
        self.assertEqual(s.confidence_set, ())
        self.assertEqual(s.rejection_times, (5,))

    def test_lower_wealth_zero_is_permanent(self):
        # 对称情形：x=1 使下尾因子 1+2(1/2-1)=0，下尾永久归零；
        # 上尾每步乘 2，t=5 拒绝。
        cs = MeanGridConfidenceSet([0.5], [2.0], 0.1)
        s = cs.extend([1] * 5)
        self.assertEqual(s.lower_log_evidence, (-math.inf,))
        self.assertEqual(s.confidence_set, ())
        self.assertEqual(s.rejection_times, (5,))

    def test_zero_wealth_matches_fraction_reference(self):
        # 独立精确参考：λ=2 轨道恰为 0，混合证据 = 另一轨道的一半。
        cs = MeanGridConfidenceSet([0.5], [0.0, 2.0], 0.01)
        ref = RefTwoSided([0.5], [0.0, 2.0], 0.01)
        for x in [0, 0, 1, 0, 1, 1]:
            s = cs.update(x)
            ref.update(x)
            for got, want in (
                (s.upper_log_evidence[0], ref.log_upper(0)),
                (s.lower_log_evidence[0], ref.log_lower(0)),
            ):
                if want == -math.inf:
                    self.assertEqual(got, -math.inf)  # 归零是结构事实，须精确
                else:
                    # 有限值允许 1 ulp 级舍入差（logsumexp vs 直接 log）
                    self.assertAlmostEqual(got, want, delta=1e-14)


# ---------------------------------------------------------------------------
# 全 0 / 全 1 序列
# ---------------------------------------------------------------------------

class AllZeroAllOneTests(unittest.TestCase):
    def test_all_zero_rejects_every_candidate_via_lower_tail(self):
        # 全 0 序列：真实均值 0 小于所有候选，下尾证据增长，最终全部拒绝，
        # 置信集为空（网格内没有真值——网格限制的合法结果）。
        cands = [0.3, 0.5, 0.7]
        cs = MeanGridConfidenceSet(cands, [1.0], 0.1)
        ref = RefTwoSided(cands, [1.0], 0.1)
        for _ in range(15):
            s = cs.update(0)
            ref.update(0)
        self.assertEqual(s.confidence_set, ())
        self.assertEqual(s.rejected, tuple(cands))
        # 拒绝时刻与独立 Fraction 参考逐一相等
        self.assertEqual(list(s.rejection_times), ref.rej_time)
        # 手算锚点：m=1/2, λ=1 时下尾每步乘 3/2，(3/2)^7≈17.09<20，
        # (3/2)^8≈25.63≥20 → t=8
        self.assertEqual(s.rejection_times[1], 8)
        # 上尾从未越界：上尾财富每步乘 (1-λm) < 1
        for j in range(3):
            self.assertLess(s.upper_log_evidence[j], 0.0)

    def test_all_one_rejects_every_candidate_via_upper_tail(self):
        cands = [0.3, 0.5, 0.7]
        cs = MeanGridConfidenceSet(cands, [1.0], 0.1)
        ref = RefTwoSided(cands, [1.0], 0.1)
        for _ in range(15):
            s = cs.update(1)
            ref.update(1)
        self.assertEqual(s.confidence_set, ())
        self.assertEqual(list(s.rejection_times), ref.rej_time)
        # 手算锚点：m=1/2, λ=1 时上尾每步乘 3/2 → t=8
        self.assertEqual(s.rejection_times[1], 8)
        for j in range(3):
            self.assertLess(s.lower_log_evidence[j], 0.0)

    def test_all_zero_with_max_lambda_hand_computed(self):
        # m=1/2, λ=2（=min(1/m, 1/(1-m)) 的允许最大值），alpha=1/10：
        # 全 0 时下尾每步乘 2，2^4=16<20，2^5=32≥20 → 恰在 t=5 拒绝；
        # 上尾第一步即归零。全部为 2 的幂，浮点精确。
        cs = MeanGridConfidenceSet([0.5], [2.0], 0.1)
        s = cs.extend([0] * 4)
        self.assertEqual(s.confidence_set, (0.5,))
        s = cs.update(0)
        self.assertEqual(s.confidence_set, ())
        self.assertEqual(s.rejection_times, (5,))


# ---------------------------------------------------------------------------
# 正常行为：置信集收缩到真值附近
# ---------------------------------------------------------------------------

class ConfidenceSetBehaviorTests(unittest.TestCase):
    def test_constant_stream_keeps_true_candidate(self):
        # 恒定 x=1/2：候选 1/2 的所有因子恰为 1（财富恒 1，永不拒绝）；
        # 候选 1/5 被上尾拒绝、4/5 被下尾拒绝，时刻由独立参考给出。
        cands = [0.2, 0.5, 0.8]
        cs = MeanGridConfidenceSet(cands, [1.0], 0.1)
        ref = RefTwoSided(cands, [1.0], 0.1)
        for _ in range(20):
            s = cs.update(0.5)
            ref.update(0.5)
        self.assertEqual(s.confidence_set, (0.5,))
        self.assertEqual(s.rejected, (0.2, 0.8))
        self.assertEqual(list(s.rejection_times), ref.rej_time)
        self.assertIsNone(s.rejection_times[1])
        # 手算锚点：λ=1 时 1/5 的上尾每步乘 13/10，
        # (13/10)^11≈17.9<20，(13/10)^12≈23.3≥20 → t=12
        self.assertEqual(Fraction(13, 10) ** 11 < 20 <= Fraction(13, 10) ** 12, True)
        self.assertEqual(s.rejection_times[0], 12)
        self.assertEqual(s.rejection_times[2], 12)
        # 真候选两侧 log 证据恒为 0（因子全为 1）
        self.assertEqual(s.upper_log_evidence[1], 0.0)
        self.assertEqual(s.lower_log_evidence[1], 0.0)

    def test_matches_fraction_reference_on_lcg_stream(self):
        # 确定性 LCG 流：逐点对照独立 Fraction 参考的 log 证据与拒绝时刻。
        cands = [0.3, 0.5, 0.7]
        grid = [0.0, 0.5, 1.0]
        alpha = 0.1
        cs = MeanGridConfidenceSet(cands, grid, alpha)
        ref = RefTwoSided(cands, grid, alpha)
        xs = lcg_values(60, seed=20260924, lo=0.4, hi=0.9)  # 均值约 0.65
        for x in xs:
            s = cs.update(x)
            ref.update(x)
            for j in range(3):
                for got, want in (
                    (s.upper_log_evidence[j], ref.log_upper(j)),
                    (s.lower_log_evidence[j], ref.log_lower(j)),
                ):
                    scale = max(1.0, abs(want))
                    self.assertLess(
                        abs(got - want), 1e-12 * scale,
                        msg=f"t={ref.t} j={j} core={got} ref={want}",
                    )
        self.assertEqual(list(s.rejection_times), ref.rej_time)
        # 该流均值约 0.65：候选 0.3 应被拒绝，0.7 应保留
        self.assertIn(0.3, s.rejected)
        self.assertIn(0.7, s.confidence_set)

    def test_extend_matches_single_updates(self):
        a = MeanGridConfidenceSet([0.3, 0.6], [0.0, 1.0], 0.2)
        a.extend([0.9, 0.1, 0.7, 0.2])
        b = MeanGridConfidenceSet([0.3, 0.6], [0.0, 1.0], 0.2)
        for x in (0.9, 0.1, 0.7, 0.2):
            b.update(x)
        self.assertEqual(a.snapshot(), b.snapshot())

    def test_extend_accepts_generator(self):
        cs = MeanGridConfidenceSet([0.5], [1.0], 0.2)
        cs.extend(float(x) for x in [0, 1, 1])
        self.assertEqual(cs.t, 3)


# ---------------------------------------------------------------------------
# 小规模枚举：真候选被拒绝的概率 ≤ alpha（并集界）
# ---------------------------------------------------------------------------

class UnionBoundEnumerationTests(unittest.TestCase):
    def test_rejection_mass_of_true_candidate_le_alpha(self):
        # 真均值 μ=1/2 在网格中；x_t iid Bernoulli(1/2)，枚举全部 2^8 条
        # 路径，按精确路径概率累计“候选 1/2 曾被拒绝”的质量，应 ≤ alpha。
        # 拒绝判定来自被测实现，路径概率为独立 Fraction 计算。
        alpha = 0.2
        cs_template = MeanGridConfidenceSet([0.3, 0.5, 0.7], [1.0], alpha)
        n = 8
        mass = Fraction(0)
        for bits in range(2 ** n):
            cs = MeanGridConfidenceSet([0.3, 0.5, 0.7], [1.0], alpha)
            rejected_true = False
            for t in range(n):
                x = (bits >> (n - 1 - t)) & 1
                s = cs.update(x)
                if 0.5 in s.rejected:
                    rejected_true = True
            if rejected_true:
                mass += Fraction(1, 2 ** n)
        self.assertLessEqual(mass, _fr(alpha))
        # 界非平凡：该配置下确实有路径会拒绝真候选（否则测试退化）
        self.assertGreater(mass, Fraction(0))
        del cs_template


# ---------------------------------------------------------------------------
# 输入校验与错误语义
# ---------------------------------------------------------------------------

class ValidationTests(unittest.TestCase):
    def test_candidates_validation(self):
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet([], [1.0], 0.1)  # 空网格
        with self.assertRaises(TypeError):
            MeanGridConfidenceSet("0.5", [1.0], 0.1)  # 字符串
        with self.assertRaises(TypeError):
            MeanGridConfidenceSet(0.5, [1.0], 0.1)  # 非序列
        for bad in (0, 1, 0.0, 1.0, -0.2, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    MeanGridConfidenceSet([bad], [1.0], 0.1)
        with self.assertRaises(TypeError):
            MeanGridConfidenceSet([True], [1.0], 0.1)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    MeanGridConfidenceSet([bad], [1.0], 0.1)
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet([0.5, 0.5], [1.0], 0.1)  # 重复候选

    def test_lambda_two_sided_bound(self):
        # candidates=[0.2]：上尾允许 λ≤5，但下尾要求 λ≤1/0.8=1.25。
        MeanGridConfidenceSet([0.2], [1.25], 0.1)  # 恰在边界：允许
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet([0.2], [1.25 + 1e-9], 0.1)
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet([0.2], [2.0], 0.1)
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet([0.5], [-1e-16], 0.1)
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet([0.5], [], 0.1)
        for bad in (float("nan"), float("inf"), True):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    MeanGridConfidenceSet([0.5], [bad], 0.1)

    def test_alpha_validation(self):
        for bad in (0, 1, 0.0, 1.0, -0.5, 2.0):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    MeanGridConfidenceSet([0.5], [1.0], bad)
        with self.assertRaises(TypeError):
            MeanGridConfidenceSet([0.5], [1.0], True)
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet([0.5], [1.0], float("nan"))

    def test_observation_validation(self):
        cs = MeanGridConfidenceSet([0.5], [1.0], 0.1)
        with self.assertRaises(TypeError):
            cs.update(True)
        with self.assertRaises(ValueError):
            cs.update(1.5)
        with self.assertRaises(ValueError):
            cs.update(float("nan"))
        self.assertEqual(cs.t, 0)  # 均未推进

    def test_extend_failure_leaves_no_partial_change(self):
        cs = MeanGridConfidenceSet([0.3, 0.7], [0.0, 1.0], 0.1)
        cs.extend([0.4, 0.6])
        before = cs.to_dict()
        with self.assertRaises(ValueError):
            cs.extend([0.5, 1.5, 0.5])  # 越界
        self.assertEqual(cs.to_dict(), before)
        with self.assertRaises(ValueError):
            cs.extend([0.5, float("nan")])
        self.assertEqual(cs.to_dict(), before)
        with self.assertRaises(TypeError):
            cs.extend([0.5, True])
        self.assertEqual(cs.to_dict(), before)
        with self.assertRaises(TypeError):
            cs.extend("01")
        self.assertEqual(cs.to_dict(), before)


# ---------------------------------------------------------------------------
# 序列化与恢复
# ---------------------------------------------------------------------------

class GridSerializationTests(unittest.TestCase):
    def _streamed(self):
        cs = MeanGridConfidenceSet([0.3, 0.5, 0.7], [0.0, 0.5, 1.0], 0.1)
        cs.extend(lcg_values(25, seed=7, lo=0.2, hi=0.9))
        return cs

    def test_json_round_trip_and_continue(self):
        cs = self._streamed()
        text = cs.to_json()
        json.loads(text)  # 必须是合法 JSON
        r = MeanGridConfidenceSet.from_json(
            text, candidates=[0.3, 0.5, 0.7], lambdas=[0.0, 0.5, 1.0], alpha=0.1
        )
        self.assertEqual(r.snapshot(), cs.snapshot())
        tail = [0.8, 0.05, 0.6]
        cs.extend(tail)
        r.extend(tail)
        self.assertEqual(r.snapshot(), cs.snapshot())

    def test_restore_at_zero(self):
        cs = MeanGridConfidenceSet([0.25, 0.75], [1.0], 0.05)
        r = MeanGridConfidenceSet.from_json(cs.to_json())
        self.assertEqual(r.snapshot(), cs.snapshot())
        self.assertEqual(r.t, 0)

    def test_restore_rejects_config_change(self):
        d = self._streamed().to_dict()
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(d, candidates=[0.3, 0.5, 0.71])
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(d, candidates=[0.3, 0.5])
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(d, lambdas=[0.0, 0.5])
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(d, alpha=0.05)
        # 断言一致时恢复成功
        MeanGridConfidenceSet.from_dict(
            d, candidates=[0.3, 0.5, 0.7], lambdas=[0.0, 0.5, 1.0], alpha=0.1
        )

    def test_restore_rejects_corrupted_states(self):
        d = self._streamed().to_dict()

        bad_version = copy.deepcopy(d)
        bad_version["version"] = 99
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(bad_version)

        bad_side_len = copy.deepcopy(d)
        bad_side_len["upper"] = bad_side_len["upper"][:2]
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(bad_side_len)

        # 篡改子过程的 m：与 candidates 不一致
        bad_m = copy.deepcopy(d)
        bad_m["upper"][0]["m"] = 0.31
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(bad_m)

        # 篡改下尾子过程的 m：必须恰为 1 - candidate
        bad_lm = copy.deepcopy(d)
        bad_lm["lower"][1]["m"] = 0.4
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(bad_lm)

        # 子过程 alpha 不是顶层 alpha/2
        bad_a = copy.deepcopy(d)
        bad_a["upper"][2]["alpha"] = 0.049
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(bad_a)

        # 两侧 t 不一致
        bad_t = copy.deepcopy(d)
        bad_t["lower"][0]["t"] += 1
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(bad_t)

        # 子状态内部损坏（log_wealth 长度不符）
        bad_sub = copy.deepcopy(d)
        bad_sub["upper"][0]["log_wealth"].append(0.0)
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(bad_sub)

        bad_t_bool = copy.deepcopy(d)
        bad_t_bool["t"] = True
        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_dict(bad_t_bool)

        with self.assertRaises(ValueError):
            MeanGridConfidenceSet.from_json("{not json")
        with self.assertRaises(TypeError):
            MeanGridConfidenceSet.from_dict(["not", "a", "dict"])

    def test_dead_track_round_trip(self):
        cs = MeanGridConfidenceSet([0.5], [2.0], 0.1)
        cs.update(0)  # 上尾唯一轨道归零
        r = MeanGridConfidenceSet.from_json(cs.to_json())
        self.assertEqual(r.snapshot(), cs.snapshot())
        self.assertEqual(r.snapshot().upper_log_evidence, (-math.inf,))


if __name__ == "__main__":
    unittest.main()
