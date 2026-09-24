"""固定输入演示：可选停止下的均值证据过程 + 有限网格双侧置信集。

运行：python demo.py（约 1 秒内，纯标准库、离线、确定性数据）。

演示内容：
1. 正常结果 A：观测流均值高于 m，证据 E_t 增长并首次越过 1/alpha ——
   展示可选停止（停止时刻、anytime p 值、越过前后轨迹）。
2. 正常结果 B：观测流均值低于 m，证据不越界，p 值维持较大值。
3. 实际触发的异常边界：lambda > 1/m 被拒绝（ValueError），
   以及批量追加中混入越界观测时整批不留部分变更。
4. 状态序列化/恢复后继续推进与未中断完全一致。
5. 有限均值网格双侧置信集（alpha 固定平分）：正常收缩到真值附近、
   全 0 序列导致的空置信集与财富归零边界、实际触发的非法输入异常。
"""

from __future__ import annotations

import time

from mean_evidence import MeanEvidenceProcess, MeanGridConfidenceSet

M = 0.5
GRID = [0.0, 0.5, 1.0, 2.0]  # 2.0 = 1/m 为允许的最大 lambda
ALPHA = 0.05
THRESHOLD = 1.0 / ALPHA


def lcg(n: int, seed: int = 20260923, lo: float = 0.0, hi: float = 1.0):
    """仓库内确定性线性同余序列，取 [lo, hi)。"""
    state = seed
    out = []
    for _ in range(n):
        state = (1103515245 * state + 12345) % (2 ** 31)
        out.append(lo + (hi - lo) * (state / (2 ** 31)))
    return out


def banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("-" * 72)


def main() -> None:
    t0 = time.perf_counter()

    # --- 场景 A：均值高于 m 的确定性流，展示可选停止 ----------------------
    banner("场景 A（正常结果·触发停止）：H0 条件均值 <= 0.5，alpha=0.05")
    print(f"固定 lambda 网格 {GRID}（含端点 lambda*=1/m=2），阈值 1/alpha={THRESHOLD:g}")
    stream_hi = lcg(400, seed=20260923, lo=0.55, hi=1.0)  # 均值约 0.775 > m
    proc = MeanEvidenceProcess(M, GRID, ALPHA)
    print(f"样本均值（前400点）= {sum(stream_hi) / len(stream_hi):.4f} > m={M}")
    print("逐点轨迹（停止前及停止当步）：")
    stop_snap = None
    for x in stream_hi:
        snap = proc.update(x)
        if snap.t <= 4 or snap.stopped:
            print(f"  t={snap.t:3d}  x_t={x:.4f}  E_t={snap.evidence:10.4g}  "
                  f"max E={snap.max_evidence:10.4g}  p_t={snap.p_value:.4g}")
        if snap.stopped and stop_snap is None:
            stop_snap = snap
            break
    assert stop_snap is not None
    print(f"=> 首次 E_t >= 1/alpha 的时刻（可选停止时间）= {stop_snap.stop_time}")
    print(f"=> 该时刻 anytime p = min(1, 1/max E) = {stop_snap.p_value:.6g} <= alpha={ALPHA}")
    print("   按 Ville 不等式，H0（条件均值<=m）下如此早停止的概率 <= alpha，")
    print("   停止时刻可以依赖当前数据任意选择，结论不需要事先固定样本量。")

    # --- 场景 B：均值低于 m 的流，不停止 -----------------------------------
    banner("场景 B（正常结果·不拒绝）：观测整体低于 m")
    stream_lo = lcg(400, seed=424242, lo=0.0, hi=0.45)  # 均值约 0.225 < m
    proc_b = MeanEvidenceProcess(M, GRID, ALPHA)
    snap_b = proc_b.extend(stream_lo)
    print(f"样本均值（400点）= {sum(stream_lo) / len(stream_lo):.4f} < m={M}")
    print(f"  t={snap_b.t}  E_t={snap_b.evidence:.4g}  "
          f"max E={snap_b.max_evidence:.4g}  p_t={snap_b.p_value:.4g}")
    print(f"=> stopped={snap_b.stopped}，未越过阈值 {THRESHOLD:g}，不拒绝 H0。")

    # --- 场景 C：实际触发的异常边界 ----------------------------------------
    banner("场景 C（实际触发的异常边界）")

    print("C1 构造非法 lambda（3.0 > 1/m = 2.0）：")
    try:
        MeanEvidenceProcess(M, [0.0, 3.0], ALPHA)
    except ValueError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")

    print("C2 批量追加混入越界观测 x=1.5（原子性，不留部分变更）：")
    proc_c = MeanEvidenceProcess(M, GRID, ALPHA)
    proc_c.extend([0.8, 0.7])
    state_before = proc_c.to_dict()
    try:
        proc_c.extend([0.9, 1.5, 0.6])  # 第 2 个元素非法
    except ValueError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")
    print(f"  失败后 t 仍为 {proc_c.t}（=2），状态与失败前完全一致："
          f"{proc_c.to_dict() == state_before}")

    print("C3 从序列化状态恢复后试图更改 m：")
    restored = MeanEvidenceProcess.from_json(proc.to_json())
    try:
        MeanEvidenceProcess.from_dict(restored.to_dict(), m=0.51)
    except ValueError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")

    # --- 场景 D：序列化恢复后继续，与未中断一致 ----------------------------
    banner("场景 D（状态序列化/恢复，恢复后不得变更 m/grid）")
    full = MeanEvidenceProcess(M, GRID, ALPHA)
    full.extend(stream_hi[:50])
    split = MeanEvidenceProcess(M, GRID, ALPHA)
    split.extend(stream_hi[:20])
    resumed = MeanEvidenceProcess.from_json(
        split.to_json(), m=M, lambdas=GRID, alpha=ALPHA
    )
    resumed.extend(stream_hi[20:50])
    same_wealth = resumed.log_wealth() == full.log_wealth()
    same_snapshot = resumed.snapshot() == full.snapshot()
    print(f"  分段恢复并继续 30 步后，财富轨道一致：{same_wealth}；快照一致：{same_snapshot}")

    # --- 场景 E：有限均值网格上的双侧置信集 -------------------------------
    banner("场景 E（有限均值网格双侧置信集，alpha=0.1 固定平分为每侧 0.05）")
    candidates = [0.2, 0.35, 0.5, 0.65, 0.8]
    lam_max = min(min(1.0 / c, 1.0 / (1.0 - c)) for c in candidates)
    grid2 = [0.0, 0.5, 1.0, lam_max]  # 1.25 = min_j min(1/m_j, 1/(1-m_j))
    print(f"候选网格 {candidates}，共享 lambda 网格 {grid2}")
    print("每个候选维护上尾（H0: μ<=m_j）与下尾（H0: μ>=m_j）两条财富，")
    print("任一侧混合财富 >= 2/alpha=20 即拒绝该候选；下注为固定常数 λ，")
    print("不依赖任何观测，自动满足“下注仅依赖过去”。")

    stream_mid = lcg(300, seed=777, lo=0.45, hi=0.85)  # 均值约 0.65
    cs = MeanGridConfidenceSet(candidates, grid2, 0.1)
    print(f"样本均值（300点）= {sum(stream_mid) / len(stream_mid):.4f}")
    prev = None
    for x in stream_mid:
        snap = cs.update(x)
        cur = snap.confidence_set
        if cur != prev:
            print(f"  t={snap.t:3d}  置信集收缩 -> "
                  f"{[round(c, 2) for c in cur]}"
                  + ("" if cur else "（空：网格内已无未拒绝候选）"))
            prev = cur
        if not cur:
            break
    snap = cs.snapshot()
    print(f"=> 最终置信集 {[round(c, 2) for c in snap.confidence_set]}，"
          f"被拒绝 {[round(c, 2) for c in snap.rejected]}")
    print(f"   各候选拒绝时刻 {list(snap.rejection_times)}")

    print("E2 边界：全 0 序列（真实均值 0 不在网格内）")
    cs_zero = MeanGridConfidenceSet([0.5], [2.0], 0.1)
    for _ in range(5):
        snap_z = cs_zero.update(0)
    print(f"  x=0 使上尾因子 1+2(0-1/2)=0：上尾财富归零且永久不可复活，"
          f"upper_log_evidence={snap_z.upper_log_evidence[0]}")
    print(f"  下尾每步乘 2，t={snap_z.t} 时越过 20，置信集为空 "
          f"{snap_z.confidence_set}——空集表示真值不在给定网格内（网格限制）")

    print("E3 实际触发的异常边界：")
    print("E3a 重复候选均值：")
    try:
        MeanGridConfidenceSet([0.5, 0.5], [1.0], 0.1)
    except ValueError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")
    print("E3b lambda 超过双侧共享上界（候选 0.2 的下尾只允许 λ<=1.25）：")
    try:
        MeanGridConfidenceSet([0.2], [1.5], 0.1)
    except ValueError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")
    print("E3c 批量追加混入越界观测（原子性，不留部分变更）：")
    cs_c = MeanGridConfidenceSet(candidates, grid2, 0.1)
    cs_c.extend([0.6, 0.7])
    state_before = cs_c.to_dict()
    try:
        cs_c.extend([0.9, 1.5, 0.6])
    except ValueError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")
    print(f"  失败后 t 仍为 {cs_c.t}（=2），状态与失败前完全一致："
          f"{cs_c.to_dict() == state_before}")

    print("E4 置信集状态序列化/恢复后继续，与未中断一致：")
    full_g = MeanGridConfidenceSet(candidates, grid2, 0.1)
    full_g.extend(stream_mid[:60])
    part_g = MeanGridConfidenceSet(candidates, grid2, 0.1)
    part_g.extend(stream_mid[:25])
    resumed_g = MeanGridConfidenceSet.from_json(
        part_g.to_json(), candidates=candidates, lambdas=grid2, alpha=0.1
    )
    resumed_g.extend(stream_mid[25:60])
    print(f"  分段恢复并继续 35 步后，快照一致："
          f"{resumed_g.snapshot() == full_g.snapshot()}")

    print("\n" + "=" * 72)
    print(f"演示完成，实际耗时 {time.perf_counter() - t0:.3f} 秒（无 sleep、无网络）。")
    print("保证仅依赖 H0：E[x_t | F_{t-1}] <= m（允许自适应、可非同分布观测）；")
    print("模块不据此宣称独立同分布，也不对该条件均值界之外的假设作任何推广。")


if __name__ == "__main__":
    main()
