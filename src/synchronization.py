import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import heapq
from mpl_toolkits.mplot3d import Axes3D  # 3D 图支持

# ======================================
# 全局内部时钟
# ======================================
CLOCK_LENGTH = 24
DUTY_CYCLE = 3


# ======================================
# 模型函数
# ======================================
def simulate_fireflies(N=150, L=CLOCK_LENGTH, r=0.1, T=5000, seed=1):
    FLASH_LEN = L // DUTY_CYCLE
    FLASH_START = L - FLASH_LEN

    rng = np.random.default_rng(seed)
    phase_history = np.zeros((T, N), dtype=int)

    # 1. 随机位置
    pos = rng.random((N, 2))
    dists = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    neigh = (dists < r) & (dists > 0)
    avg_neighbors = neigh.sum(axis=1).mean()

    # 2. 初始相位 & 闪烁状态（完全复刻 C）
    phases = rng.integers(0, L, size=N)
    flashing = (phases >= FLASH_START)

    flash_counts = np.zeros(T, dtype=int)

    for t in range(T):
        phase_history[t] = phases % L

        # 3. 记录闪烁数（与 C 相同）
        flash_counts[t] = flashing.sum()

        # 4. 相位推进（每个时间步开头）
        phases = (phases + 1)

        # 5. 根据新相位确定闪烁（必须分 3 个条件写）
        phase_mod = phases % L
        flashing = (phase_mod >= FLASH_START)

        # 当前步骤闪烁状态已经更新完毕（与 C 一致）

        # 6. 相位==26 的萤火虫检测邻居 (C: CYCLE_LENGTH/2 + 1 = 26)
        TRIGGER_PHASE = FLASH_START + 1
        idxs = np.where((phases % L) == TRIGGER_PHASE)[0]

        delta = np.zeros(N, dtype=int)

        for i in idxs:
            neigh_i = neigh[i]
            k = neigh_i.sum()
            if k == 0:
                continue

            # 检测邻居闪烁数量
            if flashing[neigh_i].sum() > (k / 2):
                delta[i] = 1  # 提前相位

        # 7. 相位提前
        phases = (phases + delta) % L

    return avg_neighbors, flash_counts, pos, phase_history


# ======================================
# Task A
# ======================================
def taskA(output_dir="../output"):
    N = 150
    L = CLOCK_LENGTH
    T = 5000
    rs = [0.05, 0.1, 0.5, 1.4]
    seed = 15
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # 固定 seed=1 → 同一位置用于不同 r
    for r in rs:
        avg_n, flash_counts, pos, phase_history = simulate_fireflies(N=N, L=L, r=r, T=T, seed=seed)
        # plot_firefly_positions_from_sim(pos, r, seed, f"{output_dir}/phase_xyz_r{r}.png")
        # plot_phase_xyz(
        #     phase_history,
        #     L,
        #     r=r,
        #     seed=seed,
        #     save_path=f"{output_dir}/phase_xyz_r{r}.png"
        # )

        results[r] = (avg_n, flash_counts)
        print(f"r = {r}: avg neighbors ≈ {avg_n:.2f}")

    # 绘图
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    T_range = np.arange(T)

    for ax, r in zip(axes, rs):
        avg_n, flash_counts = results[r]
        ax.plot(T_range, flash_counts)
        ax.set_ylim(0, 150)
        ax.set_yticks([0, 50, 100, 150])
        ax.set_ylabel("Flashing")
        ax.set_title(f"r = {r}, avg neighbors ≈ {avg_n:.1f}, seed={seed}")

    axes[-1].set_xlabel("Time step")
    plt.tight_layout()

    save_path = os.path.join(output_dir, "taskA.png")
    plt.savefig(save_path, dpi=300)
    print(f"Task A 图像已保存到: {save_path}")

    plt.show()


# ======================================
# Task B（多进程）
# ======================================
def _worker_task(args):
    r, T, seed = args
    # seed = 1
    _, flash_counts, _, phase_history = simulate_fireflies(r=r, L=CLOCK_LENGTH, T=T, seed=seed)
    last_cycle = flash_counts[-2 * CLOCK_LENGTH:]
    amp = last_cycle.max() - last_cycle.min()
    return r, amp, phase_history, seed


def taskB(output_dir="../output"):
    os.makedirs(output_dir, exist_ok=True)

    r_values = np.arange(0.025, 1.4001, 0.025)
    num_runs = 500
    T = 3000

    # 生成任务
    tasks = []
    for r in r_values:
        for _ in range(num_runs):
            seed = np.random.randint(1, 10 ** 9)
            tasks.append((float(r), T, seed))
    print(f"Task B 总任务数: {len(tasks)}")

    # 多进程执行
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(_worker_task, tasks)

    # 聚合结果
    amp_dict = {float(r): [] for r in r_values}
    for r, amp, _, _ in results:
        amp_dict[r].append(amp)

    # 求平均幅度
    r_sorted = []
    amp_sorted = []
    for r in r_values:
        r_sorted.append(r)
        amp_sorted.append(np.mean(amp_dict[r]))

    # 绘图（散点图）
    plt.figure(figsize=(10, 6))

    # 每一次运行都画一个散点
    for r in r_values:
        amps = amp_dict[r]
        rs = np.full(len(amps), r)  # x 轴为同一个 r
        plt.scatter(rs, amps, s=8, alpha=0.3)  # 低透明度方便观察密度

    plt.xlabel("r")
    plt.ylabel("amplitude (last cycle)")
    plt.title(f"Task B: Scatter of all amplitudes vs r (Clock_length = {CLOCK_LENGTH}, Duty_cycle = 1/{DUTY_CYCLE})")
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"taskB_scatter_{CLOCK_LENGTH}_{DUTY_CYCLE}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Task B 散点图已保存到: {save_path}")

    plt.show()

    # ---------------------------
    #  绘制热力图（每个 r 的 amplitude 分布）
    # ---------------------------

    # 把所有 amplitude 收集起来确定全局范围
    all_amps = np.concatenate([amp_dict[r] for r in r_values])
    amp_min, amp_max = np.min(all_amps), np.max(all_amps)

    # 设置 amplitude 分箱数，比如 60 个 bins
    num_bins = 60
    bins = np.linspace(amp_min, amp_max, num_bins + 1)

    # 准备热力图矩阵： shape = (num_bins, len(r_values))
    heat_matrix = np.zeros((num_bins, len(r_values)))

    # 对每个 r 分布进行统计
    for j, r in enumerate(r_values):
        hist, _ = np.histogram(amp_dict[r], bins=bins)
        heat_matrix[:, j] = hist

    # 对 count 做 log 变换，避免大值压扁颜色
    heat_matrix_log = np.log1p(heat_matrix)

    # 画热力图
    plt.figure(figsize=(12, 6))
    plt.imshow(
        heat_matrix_log,
        aspect='auto',
        origin='lower',
        extent=(r_values[0], r_values[-1], amp_min, amp_max),
        cmap='hot'  # 红黑最明显的配色
    )
    plt.colorbar(label="log(count + 1)")
    plt.xlabel("r")
    plt.ylabel("amplitude")
    plt.title(
        f"Task B: Amplitude distribution heatmap (log scaled) (Clock_length = {CLOCK_LENGTH}, Duty_cycle = 1/{DUTY_CYCLE})")
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"taskB_heatmap_log_{CLOCK_LENGTH}_{DUTY_CYCLE}.png")
    plt.savefig(save_path, dpi=300)
    print(f"Task B 热力图（log 变换）已保存到: {save_path}")

    plt.show()


def clock_distribution(output_dir="../data"):
    os.makedirs(output_dir, exist_ok=True)

    # r_values = np.arange(0.025, 1.4001, 0.025)
    r_values = [1.4]
    num_runs = 500
    T = 5000

    # 生成任务
    tasks = []
    for r in r_values:
        for _ in range(num_runs):
            seed = np.random.randint(1, 10 ** 9)
            tasks.append((float(r), T, seed))

    print(f"Task B 总任务数: {len(tasks)}")

    # 多进程执行
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(_worker_task, tasks)

        # 聚合结果
        # 两个分类目录
        high_dir = os.path.join(output_dir, "high_performance_phase")
        low_dir = os.path.join(output_dir, "low_performance_phase")

        os.makedirs(high_dir, exist_ok=True)
        os.makedirs(low_dir, exist_ok=True)

        # 按 amp 分类保存
        for r, amp, phase_history, seed in results:

            if amp >= 140:
                save_dir = high_dir
            else:
                save_dir = low_dir

            filepath = os.path.join(
                save_dir,
                f"phase_history_{seed}.csv"
            )

            np.savetxt(
                filepath,
                phase_history,
                delimiter=",",
                fmt="%d"
            )

        print("保存完成")


# ======================================
# 绘制位置
# ======================================
def plot_firefly_positions_from_sim(pos, r, seed=None, save_path=None):
    """
    绘制萤火虫位置，并根据给定的通信半径 r 画出邻居连线。
    pos：simulate_fireflies 返回的真实位置
    r：邻居判定半径
    seed：可选，仅用于标题展示
    """

    N = pos.shape[0]

    # 计算距离矩阵
    dists = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    neigh = (dists < r) & (dists > 0)

    plt.figure(figsize=(7, 7))

    # === 1. 连线：画在底层、线条要细 ===
    for i in range(N):
        for j in np.where(neigh[i])[0]:
            # 只画 i<j，防止双线叠加
            if i < j:
                plt.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    color="gray",
                    linewidth=0.4,  # 更细的线
                    alpha=0.6  # 半透明更干净
                )

    # === 2. 绘制萤火虫点 ===
    plt.scatter(pos[:, 0], pos[:, 1], s=35, c="gold", edgecolor="black", zorder=3)

    title = f"Firefly Positions with neighbor links (r={r}) (seed={seed})"
    if seed is not None:
        title += f", seed={seed}"
    plt.title(title)

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect("equal")
    plt.grid(True, alpha=0.25)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"萤火虫位置图已保存到: {save_path}")

    plt.show()


def plot_phase_xyz(phase_history, L, r, seed, save_path=None):
    """
    绘制 TaskA 中萤火虫相位随时间变化的 3D XYZ 图。

    参数：
        phase_history: shape (T, N) 的相位记录（simulate_fireflies 返回）
        L: 相位周期（例如 50）
        r: 邻居半径（显示在标题中）
        seed: 随机种子（显示在标题中）
    """
    phase_history = phase_history[:100]

    T, N = phase_history.shape

    # 统计每个时间的相位数量（0~L-1）

    phase_count = np.zeros((T, L), dtype=int)
    for t in range(T):
        phase_count[t] = np.bincount(phase_history[t], minlength=L)

    # 绘图
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(111, projection="3d")

    Ts = np.arange(T)
    Ph = np.arange(L)
    TT, PP = np.meshgrid(Ts, Ph, indexing='ij')

    surf = ax.plot_surface(
        TT, PP, phase_count,
        cmap="viridis",
        edgecolor="none",
        rstride=4,
        cstride=4,
        alpha=0.9
    )

    ax.set_box_aspect((3, 1, 0.6))

    ax.set_xlabel("Time")
    ax.set_ylabel("Phase (0~%d)" % (L - 1))
    ax.set_zlabel("Count")

    ax.set_title(f"Phase Distribution Over Time (r={r}, seed={seed})")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Phase XYZ 图已保存到: {save_path}")

    plt.tight_layout()

    plt.show()

    return phase_count


def plot_clock_heatmap_from_csv(
        csv_path,
        output_dir="../output",
        L=CLOCK_LENGTH,
):
    os.makedirs(output_dir, exist_ok=True)

    # ========= 1. 读取 CSV =========
    phase_history_full = np.loadtxt(csv_path, delimiter=",", dtype=int)
    T_total, N = phase_history_full.shape

    title = os.path.basename(csv_path).replace(".csv", "")

    # =========================================================
    # Part A: 初始态热力图（t = 0，对应第 1 秒）
    # =========================================================
    init_phases = phase_history_full[0]  # shape = (N,)
    init_counts = np.bincount(init_phases, minlength=L)

    plt.figure(figsize=(6, 6))
    plt.imshow(
        init_counts[:, None],  # (L, 1)
        aspect="auto",
        origin="lower",
        extent=[1, 2, 0, L]  # x 轴只表示第 1 秒
    )

    plt.xlabel("Time")
    plt.ylabel("Clock phase")
    plt.title(title + " (initial state)")

    plt.xticks([1], ["1"])  # x 轴只显示 1

    cbar = plt.colorbar()
    cbar.set_label("Count")

    plt.tight_layout()

    init_png = os.path.join(
        output_dir,
        title + "_initial_state_distribution.png"
    )
    # plt.savefig(init_png, dpi=300)
    plt.close()

    print(f"初始态分布图已保存: {init_png}")

    # =========================================================
    # Part B: 稳态热力图（最后 100 个时间步）
    # =========================================================
    tail = 30
    if T_total > tail:
        phase_history = phase_history_full[-tail:, :]
    else:
        phase_history = phase_history_full

    T, N = phase_history.shape

    heatmap = np.zeros((T, L), dtype=int)
    for t in range(T):
        heatmap[t] = np.bincount(
            phase_history[t],
            minlength=L
        )

    t_start = T_total - T
    t_end = T_total

    plt.figure(figsize=(24, 8))
    plt.imshow(
        heatmap.T,
        aspect="auto",
        origin="lower",
        extent=[t_start, t_end, 0, L]
    )

    plt.xlabel("Time (simulation step)")
    plt.ylabel("Clock phase")
    plt.title(title + f" (steady state: t = {t_start} … {t_end})")

    cbar = plt.colorbar()
    cbar.set_label("Count")

    plt.tight_layout()

    steady_png = os.path.join(
        output_dir,
        title + "_heatmap_last100.png"
    )
    plt.savefig(steady_png, dpi=300)
    plt.close()

    print(f"稳态热力图已保存: {steady_png}")


# ======================================
# 主程序入口
# ======================================
if __name__ == "__main__":
    # taskA()

    # taskB()

    # clock_distribution()

    input_dir = "../data/low_performance_phase"
    output_dir = "../output/low_performance_phase"

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            csv_path = os.path.join(input_dir, filename)

            print(f"Processing: {filename}")

            plot_clock_heatmap_from_csv(
                csv_path=csv_path,
                output_dir=output_dir,
                L=CLOCK_LENGTH,
            )
