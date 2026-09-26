"""固定起点权重混合证据过程的确定性离线演示。

运行：python demo_iteration.py
不访问网络，不使用账号、密钥、sleep 或预录输出。
"""

from __future__ import annotations

import time

from mean_evidence import RestartEvidenceMixture


M = 0.5
GRID = [0.0, 2.0]
ALPHA = 0.05
MAX_STARTS = 8
WEIGHTS = [1.0 / MAX_STARTS] * MAX_STARTS


def show(label: str, snap) -> None:
    contributions = ", ".join(
        f"s{index + 1}={value:.4g}"
        for index, value in enumerate(snap.start_contributions)
    )
    print(
        f"{label:24s} t={snap.t:2d} E={snap.evidence:10.4g} "
        f"future={snap.future_weight:.4g} stop={snap.stop_time} [{contributions}]"
    )


def main() -> None:
    started = time.perf_counter()
    print("固定起点混合：每个可能起点重启原有下注财富过程，权重事先固定且总和为 1")
    print(f"m={M}, grid={GRID}, alpha={ALPHA}, threshold={1 / ALPHA:g}")
    print(f"max_starts={MAX_STARTS}, weights={[round(w, 6) for w in WEIGHTS]}")

    print("\n[1] 无变化：观测持续低于 m；未开启起点的质量始终保留为 1。")
    no_change = RestartEvidenceMixture(M, GRID, ALPHA, WEIGHTS, MAX_STARTS)
    for t, _ in enumerate((0, 0, 0, 0, 0), start=1):
        snap = no_change.update(0)
        if t in (1, 2, 5):
            show("x=0", snap)

    print("\n[2] 晚变化：前 3 点为 0，之后为 1；每个后缀独立从头累计。")
    late = RestartEvidenceMixture(M, GRID, ALPHA, WEIGHTS, MAX_STARTS)
    stop_snap = None
    for t in range(1, 16):
        x = 0 if t <= 3 else 1
        snap = late.update(x)
        if t in (1, 3, 4, 8) or (snap.stopped and stop_snap is None):
            show(f"x={x}", snap)
        if snap.stopped and stop_snap is None:
            stop_snap = snap

    assert stop_snap is not None
    largest = max(
        range(len(stop_snap.start_contributions)),
        key=lambda i: stop_snap.start_contributions[i],
    ) + 1
    print(
        f"首次越阈值时刻={stop_snap.stop_time}，当时最大贡献起点为 s{largest}。"
    )
    print("该起点仅用于解释当前各分量贡献，不是认证后的变点估计。")

    print("\n[3] 零权重：s2 持续更新，但权重为 0，不能改变混合值。")
    zero_weight = RestartEvidenceMixture(M, GRID, ALPHA, [1.0, 0.0], 2)
    snap = zero_weight.extend([1, 1])
    show("weights=[1,0]", snap)
    second_evidence = zero_weight.component_snapshot(2).evidence
    print(f"s2 自身证据={second_evidence:.4g}，混合贡献={snap.start_contributions[1]:g}")

    print("\n[4] 报警后继续：首次报警时刻冻结，后续证据仍按同一权重规则更新。")
    after_alarm = RestartEvidenceMixture(M, GRID, 2.0 / 3.0, [1.0, 0.0], 2)
    first = after_alarm.update(1)
    continued = after_alarm.extend([0, 0])
    show("报警步 x=1", first)
    show("继续 x=0,0", continued)

    elapsed = time.perf_counter() - started
    print(f"\n演示完成，实际耗时 {elapsed:.3f} 秒；结论仅针对给定 H0。")
    assert elapsed < 90.0


if __name__ == "__main__":
    main()
