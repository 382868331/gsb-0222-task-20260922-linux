"""固定输入演示：可选停止下的均值证据过程。

运行：python demo.py（约 1 秒内，纯标准库、离线、确定性数据）。

演示内容：
1. 正常结果 A：观测流均值高于 m，证据 E_t 增长并首次越过 1/alpha ——
   展示可选停止（停止时刻、anytime p 值、越过前后轨迹）。
2. 正常结果 B：观测流均值低于 m，证据不越界，p 值维持较大值。
3. 实际触发的异常边界：lambda > 1/m 被拒绝（ValueError），
   以及批量追加中混入越界观测时整批不留部分变更。
4. 状态序列化/恢复后继续推进与未中断完全一致。
5. 双侧候选均值网格（TwoSidedMeanEvidence）：上下尾财富、固定平分 alpha/2、
   尚未拒绝的候选集合随流收缩；含退化下注、全 0/全 1、网格 λ 上界报错。
"""

from __future__ import annotations

import time

from mean_evidence import MeanEvidenceProcess, TwoSidedMeanEvidence

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

    # --- 场景 E：双侧候选网格 ---------------------------------------------
    banner("场景 E（双侧候选均值网格 TwoSidedMeanEvidence）")
    print("候选 mus=[0.2,0.4,0.6,0.8]，共用网格 lambdas=[0,1.25]，alpha=0.1")
    print("（公共 λ 上界 = min_i min(1/mu_i,1/(1-mu_i)) = 1/0.8 = 1.25；"
          "每侧 alpha/2=0.05，阈值 2/alpha=20）")
    two = TwoSidedMeanEvidence([0.2, 0.4, 0.6, 0.8], [0.0, 1.25], 0.1)
    print(f"初始（t=0）：尚未拒绝候选 = {two.snapshot().surviving}")
    # 前 8 个 0.95（≈强势上尾证据）；随后 12 个 0.05（下尾证据累积但不回溯）
    for x in [0.95] * 8:
        s = two.update(x)
    print("8 个 0.95 后，各候选状态：")
    for c in s.candidates:
        tag = f"拒绝(上尾 t={c.reject_time})" if c.rejected else "存活"
        print(f"  mu={c.mu:.1f}  E+={c.upper_evidence:9.4g}  "
              f"E-={c.lower_evidence:9.4g}  p_双侧={c.p_value_two_sided:.4g}  {tag}")
    print(f"=> 尚未拒绝的候选集合：{s.surviving}（网格点之外的均值无法评估）")
    for x in [0.05] * 12:
        s = two.update(0.05)
    print("再接 12 个 0.05 后：")
    for c in s.candidates:
        side_zh = "上" if c.reject_side == "upper" else "下"
        tag = (f"首次拒绝 t={c.reject_time} {side_zh}尾"
               if c.rejected else "存活")
        print(f"  mu={c.mu:.1f}  E+={c.upper_evidence:9.4g}  "
              f"E-={c.lower_evidence:9.4g}  p_双侧={c.p_value_two_sided:.4g}  {tag}")
    print(f"=> 最终尚未拒绝：{s.surviving}；首次拒绝时刻不被后续下尾证据改写。")

    # --- 场景 F：双侧的退化下注 / 全0全1 / 实际错误 ------------------------
    banner("场景 F（双侧边界：退化网格、财富归零、实际错误）")

    print("F1 退化下注（仅有 λ=0）：无论看多少 0/1，证据恒为 1，候选永不被拒：")
    deg = TwoSidedMeanEvidence([0.2, 0.5, 0.8], [0.0], 0.1)
    sd = deg.extend([1] * 20 + [0] * 20)
    print(f"  40 个 0/1 后 surviving={sd.surviving}，"
          f"任取候选 E+={sd.candidates[0].upper_evidence:g}，"
          f"p_双侧={sd.candidates[0].p_value_two_sided:g}")

    print("F2 全 1 流：下尾非零轨道首步因子 1-λ(1-μ)=0 归零且不复活；")
    print("   上尾增长并拒绝低候选（mu=.5, lambdas={0,2}, 每侧阈值 4，t=3）：")
    ones = TwoSidedMeanEvidence([0.5], [0.0, 2.0], 0.5)
    for x in [1, 1, 1]:
        so = ones.update(x)
    co = so.candidates[0]
    up_lw, lo_lw = ones.candidate_log_wealth(0)
    side_zh = "上" if co.reject_side == "upper" else "下"
    print(f"  E+={co.upper_evidence:g}（拒绝，{side_zh}尾 t={co.reject_time}），"
          f"E-={co.lower_evidence:g}；上尾 log 财富={up_lw}，下尾={lo_lw}")

    print("F3 全 0 流的镜像：下尾在 t=3 拒绝：")
    zeros = TwoSidedMeanEvidence([0.5], [0.0, 2.0], 0.5)
    cz = zeros.extend([0, 0, 0]).candidates[0]
    print(f"  E-={cz.lower_evidence:g}，下尾拒绝 t={cz.reject_time}")

    print("F4 λ 超过公共上界（候选含 0.1 时 cap=min(1/0.1,1/0.9)≈1.111，给 2.0）：")
    try:
        TwoSidedMeanEvidence([0.1, 0.5, 0.9], [2.0], 0.1)
    except ValueError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")

    print("F5 候选均值碰到 0/1 边界（必须严格位于 (0,1)）：")
    try:
        TwoSidedMeanEvidence([0.5, 1.0], [1.0], 0.1)
    except ValueError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")

    print("F6 批量混入非法观测（x=1.5）：整批拒绝，全部候选状态不变：")
    atom = TwoSidedMeanEvidence([0.3, 0.7], [0.0, 1.4], 0.1)
    atom.extend([0.8, 0.2])
    before = atom.to_dict()
    try:
        atom.extend([0.9, 1.5, 0.1])
    except ValueError as exc:
        print(f"  捕获 {type(exc).__name__}: {exc}")
    print(f"  失败后 t={atom.t}（=2），状态与失败前完全一致：{atom.to_dict() == before}")

    print("\n" + "=" * 72)
    print(f"演示完成，实际耗时 {time.perf_counter() - t0:.3f} 秒（无 sleep、无网络）。")
    print("保证仅依赖 H0：E[x_t | F_{t-1}] <= m（允许自适应、可非同分布观测）；")
    print("模块不据此宣称独立同分布，也不对该条件均值界之外的假设作任何推广。")


if __name__ == "__main__":
    main()
