from operator import truediv
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import os
import networkx as nx
import matplotlib.animation as animation
import pandas as pd

# ======================================
# 全局参数
# ======================================
CLOCK_LENGTH = 10
DUTY_CYCLE = 2
Num_firefly = 100


# ======================================
# 核心模型：固定 K 度随机正则图
# ======================================
def simulate_fireflies_graph_fast(adj_matrix, K, T, seed):
    N = adj_matrix.shape[0]
    L = CLOCK_LENGTH
    FLASH_LEN = L // DUTY_CYCLE
    FLASH_START = L - FLASH_LEN

    rng = np.random.default_rng(seed)
    phases = rng.integers(0, L, size=N)
    flashing = (phases >= FLASH_START)

    flash_counts = np.zeros(T, dtype=int)
    # --- 1. 新增：初始化相位历史矩阵 ---
    phase_history = np.zeros((T, N), dtype=int)

    for t in range(T):
        # --- 2. 新增：记录当前相位 (取模确保在 0-L 之间) ---
        phase_history[t] = phases % L

        flash_counts[t] = flashing.sum()
        phases = (phases + 1)
        phase_mod = phases % L
        flashing = (phase_mod >= FLASH_START)

        TRIGGER_PHASE = FLASH_START + 1
        idxs = np.where((phases % L) == TRIGGER_PHASE)[0]
        delta = np.zeros(N, dtype=int)

        for i in idxs:
            if flashing[adj_matrix[i]].sum() > (K / 2):
                delta[i] = 1

        phases = (phases + delta) % L

    # --- 3. 修改：将 None 替换为 phase_history ---
    return flash_counts, phase_history


# ======================================
# Task B：多进程相变分析
# ======================================
def _worker_graph_task(args):
    # 现在 args 包含了预生成的 adj_matrix
    k, T, seed, adj_matrix = args

    # 修改 simulate_fireflies_graph 使其能直接接受 adj_matrix
    flash_counts, _ = simulate_fireflies_graph_fast(adj_matrix, k, T, seed)

    last_cycle = flash_counts[-2 * CLOCK_LENGTH:]
    amp = last_cycle.max() - last_cycle.min()
    return k, amp

def _worker_graph_task_random_topo(args):
    k, T, seed = args
    rng = np.random.default_rng(seed)

    while True:
        graph_seed = int(rng.integers(0, 2 ** 32 - 1))
        # G = nx.random_regular_graph(d=k, n=Num_firefly, seed=graph_seed)
        py_rng = random.Random(graph_seed)
        ig.set_random_number_generator(py_rng)

        G = ig.Graph.K_Regular(n=Num_firefly, k=k)
        if G.is_connected():
            break

    adj_matrix = np.array(G.get_adjacency().data, dtype=bool)
    sim_seed = int(rng.integers(0, 2 ** 32 - 1))
    flash_counts, _ = simulate_fireflies_graph_fast(adj_matrix, k, T, sim_seed)
    last_cycle = flash_counts[-2 * CLOCK_LENGTH:]
    return k, int(last_cycle.max() - last_cycle.min())

def taskB_graph(output_dir="../output", random_topo=True):
    """
    random_topo=True  : 每次仿真随机生成一张新拓扑（拓扑+初始相位都随机）
    random_topo=False : 每个 K 预生成一张固定连通拓扑，只随机初始相位
    """
    os.makedirs(output_dir, exist_ok=True)

    # k_values = np.arange(2, Num_firefly-1, 10)
    k_values = np.append(np.arange(2, Num_firefly - 1, 10), Num_firefly - 1)
    num_runs = 1_00
    T = 1000

    tasks = []

    if random_topo:
        print(f"随机拓扑 + 随机初始相位，N={Num_firefly}")
        print("-" * 45)
        for k in k_values:
            for _ in range(num_runs):
                seed = np.random.randint(1, 10 ** 9)
                tasks.append((int(k), T, seed))
    else:
        print(f"开始预生成连通网络拓扑 (N={Num_firefly}, K=1 到 {Num_firefly - 1})...")
        print("-" * 45)
        for k in k_values:
            try:
                if k < 2:
                    print(f"跳过 K = {k:2d} | 理由: K=1 无法形成全连通正则图")
                    continue
                print(f"Generating a regular graph with K = {k:2d} ...", end="\r")
                connected = False
                attempts = 0
                while not connected:
                    attempts += 1
                    # G = nx.random_regular_graph(d=k, n=Num_firefly)
                    G = ig.Graph.K_Regular(n=Num_firefly, k=k)
                    if G.is_connected():
                        connected = True
                    if attempts > 1000:
                        raise Exception("超过1000次尝试仍无法生成连通图")
                adj_matrix = np.array(G.get_adjacency().data, dtype=bool)
                for _ in range(num_runs):
                    seed = np.random.randint(1, 10 ** 9)
                    tasks.append((int(k), T, seed, adj_matrix))
                print(f"Successfully generated K = {k:2d} | Attempts: {attempts:2d} | Number of tasks: {num_runs}")
            except Exception as e:
                print(f"\n警告: K={k:2d} 拓扑生成失败: {e}")

    worker = _worker_graph_task_random_topo if random_topo else _worker_graph_task

    print("-" * 45)
    print(f"拓扑准备就绪。开始模拟，总任务数: {len(tasks)}")
    print("-" * 45)

    amp_dict = {int(k): [] for k in k_values}
    k_completion = {int(k): 0 for k in k_values}

    with mp.Pool(mp.cpu_count()) as pool:
        results_objs = [pool.apply_async(worker, (t,)) for t in tasks]
        for r in results_objs:
            k, amp = r.get()
            amp_dict[k].append(amp)
            k_completion[k] += 1
            if k_completion[k] == num_runs:
                print(f"完成 K = {k:2d} | 平均振幅: {np.mean(amp_dict[k]):.2f}")

    # --- 绘图 1：散点图 (复刻原风格) ---
    plt.figure(figsize=(10, 6))
    for k in k_values:
        amps = amp_dict[k]
        ks = np.full(len(amps), k)
        plt.scatter(ks, amps, s=8, alpha=0.3, color='C0')

    plt.xlabel("K (Degree)")
    plt.ylabel("amplitude (last cycle)")
    plt.title(f"{random_topo}_Task B: Scatter of amplitudes vs K (L={CLOCK_LENGTH}, Duty={DUTY_CYCLE})")
    plt.grid(True, linestyle=':', alpha=0.6)

    save_scatter = os.path.join(output_dir, f"{random_topo}_taskB_graph_scatter{Num_firefly}_numRuns{num_runs}.png")
    plt.savefig(save_scatter, dpi=300,bbox_inches='tight')
    plt.close()

    # --- 绘图 2：热力图 (复刻原风格) ---
    all_amps = [a for k in k_values for a in amp_dict[k]]
    amp_min, amp_max = 0, Num_firefly
    num_bins = Num_firefly
    bins = np.linspace(amp_min, amp_max, num_bins + 1)

    heat_matrix = np.zeros((num_bins, len(k_values)))
    for j, k in enumerate(k_values):
        hist, _ = np.histogram(amp_dict[k], bins=bins)
        heat_matrix[:, j] = hist

    plt.figure(figsize=(12, 6))
    plt.imshow(
        np.log1p(heat_matrix),
        aspect='auto',
        origin='lower',
        extent=(k_values[0], k_values[-1], amp_min, amp_max),
        cmap='hot'
    )
    plt.colorbar(label="log(count + 1)")
    plt.xlabel("K (Degree)")
    plt.ylabel("amplitude")
    plt.title(f"{random_topo}_Task B: Amplitude distribution heatmap (K-Regular Graph)")

    save_heatmap = os.path.join(output_dir, f"{random_topo}_taskB_graph_heatmap{Num_firefly}_numRuns{num_runs}.png")
    plt.savefig(save_heatmap, dpi=300,bbox_inches='tight')
    plt.close()

    # --- 保存热力图矩阵为 CSV（与图片同名） ---
    save_heatmap_csv = os.path.splitext(save_heatmap)[0] + ".csv"

    # 确保是数值矩阵
    heat_matrix = np.asarray(heat_matrix, dtype=float)

    # y 方向反转
    heat_matrix = np.flipud(heat_matrix)

    # 行标签（反向 amplitude）
    amp_rows = np.arange(num_bins - 1, -1, -1)

    df_heat = pd.DataFrame(
        heat_matrix,
        index=amp_rows,
        columns=k_values
    )

    df_heat.to_csv(
        save_heatmap_csv,
        sep=";",
        index_label="amplitude_bin"
    )

    print(f"热力图矩阵 CSV 已保存至: {save_heatmap_csv}")


# ======================================
# 多线程 Worker 函数
# ======================================
def _topology_worker(args):
    adj_matrix, K, T, seed = args
    flash_counts, phase_history = simulate_fireflies_graph_fast(adj_matrix, K, T, seed)

    last_cycle = flash_counts[-2 * CLOCK_LENGTH:]
    final_amp = int(last_cycle.max() - last_cycle.min())

    def get_sync_ratio(phases):
        same_phase = (phases[:, None] == phases[None, :])
        return np.mean((same_phase & adj_matrix).sum(axis=1) / K)

    initial_ratio = get_sync_ratio(phase_history[0])
    final_ratio = get_sync_ratio(phase_history[-1])

    return final_amp, initial_ratio, final_ratio, phase_history


# ======================================
# 拓扑分析主函数 (多线程版)
# ======================================
def topology_analysis_multiprocess(K=4, num_runs=1000, T=3000):
    print(f"--- 开启多线程拓扑分析 (K={K}, 实验次数={num_runs}) ---")

    # 1. 预生成连通图
    connected = False
    while not connected:
        G = nx.random_regular_graph(d=K, n=Num_firefly)
        if nx.is_connected(G):
            connected = True
    adj_matrix = nx.to_numpy_array(G, dtype=bool)

    # 2. 准备并行任务
    tasks = []
    for _ in range(num_runs):
        seed = np.random.randint(0, 10 ** 9)
        tasks.append((adj_matrix, K, T, seed))

    # 3. 执行多线程任务
    results = []
    with mp.Pool(mp.cpu_count()) as pool:
        async_results = [pool.apply_async(_topology_worker, (t,)) for t in tasks]
        for r in async_results:
            results.append(r.get())  # 拿到的是 (amp, init, final, history)

    # 取第一个样本进行聚类展示
    sample_history = results[0][3]

    # 4. 数据处理时注意索引偏移 (原本是 0,1,2 变成 0,1,2,3)
    results.sort(key=lambda x: x[0], reverse=True)
    sorted_amps = [x[0] for x in results]
    sorted_init_ratios = [x[1] for x in results]
    sorted_final_ratios = [x[2] for x in results]

    x_indices = np.arange(len(results))

    # 5. 绘图 (保持原有风格)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左轴：振幅
    color_amp = 'royalblue'
    ax1.bar(x_indices, sorted_amps, color=color_amp, alpha=0.4, label='Final Amplitude')
    ax1.set_xlabel('Experiment Index (Sorted by Amplitude High -> Low)', fontsize=12)
    ax1.set_ylabel('Steady State Amplitude', color=color_amp, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_amp)
    ax1.set_ylim(0, Num_firefly + 1)

    # 右轴：邻居一致性比例
    ax2 = ax1.twinx()

    # 增加：初始时的比例 (使用灰色或浅色，保持风格统一)
    ax2.plot(x_indices, sorted_init_ratios, color='gray', linestyle=':', alpha=1.0, linewidth=2.5,
             label='Initial Neighbor Sync Ratio (t=0)')

    # 原有：结束时的比例
    color_ratio = 'crimson'
    ax2.plot(x_indices, sorted_final_ratios, color=color_ratio, marker='o', markersize=2,
             linestyle='-', linewidth=1, label='Final Neighbor Sync Ratio (t=End)')

    ax2.set_ylabel('Avg Neighbor Sync Ratio', color=color_ratio, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_ratio)
    ax2.set_ylim(0, 1.05)

    # 整理图例
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title(f"Topology Analysis: N={Num_firefly}, K={K}", fontsize=14)
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    fig.tight_layout()

    # 保存图片，文件名包含 K 和 N
    save_path = f"../output/topology_analysis_K{K}_N{Num_firefly}.png"
    plt.savefig(save_path, dpi=300,bbox_inches='tight')
    print(f"图表已保存至: {save_path}")

    plt.close()

    plot_best_worst_by_clock_color(adj_matrix, results, K, Num_firefly, T)
    save_sync_comparison_animation_mp4(adj_matrix, results, K, Num_firefly)

    plot_k_regular_spring_layout_snapshot(adj_matrix, results, K, Num_firefly, T)
    save_spring_sync_comparison_animation_mp4(adj_matrix, results, K, Num_firefly)


def plot_best_worst_by_clock_color(adj_matrix, results, K, N, T):
    """
    绘制同步最好（第一组）和同步最差（最后一组）的对比图。
    节点颜色由其当前Clock相位决定，Clock颜色固定。
    """
    L = CLOCK_LENGTH
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    G = nx.from_numpy_array(adj_matrix)

    # 定义 L 个 Clock 相位的固定颜色映射
    # 使用 Set1 确保颜色均匀且区分度高
    clock_phase_cmap = plt.get_cmap('Set1', L)

    def get_fixed_layout(phases):
        pos = {}
        outer_r = 10  # 8个相位圆心的轨道半径
        inner_r = 1.0  # 同一 Clock 内部节点的散布半径

        for p in range(L):
            center_angle = 2 * np.pi * p / L
            cx, cy = outer_r * np.cos(center_angle), outer_r * np.sin(center_angle)
            nodes_in_phase = np.where(phases == p)[0]

            for i, node in enumerate(nodes_in_phase):
                if len(nodes_in_phase) > 1:
                    sub_angle = 2 * np.pi * i / len(nodes_in_phase)
                    pos[node] = np.array([cx + inner_r * np.cos(sub_angle),
                                          cy + inner_r * np.sin(sub_angle)])
                else:
                    pos[node] = np.array([cx, cy])
        return pos

    # results 已经按振幅从高到低排序：results[0] 是最好，results[-1] 是最差
    best_data = {"hist": results[0][3], "amp": results[0][0], "row": 0, "label": "Best Case"}
    worst_data = {"hist": results[-1][3], "amp": results[-1][0], "row": 1, "label": "Worst Case"}

    limit = 14  # 统一坐标轴范围
    for data in [best_data, worst_data]:
        row = data["row"]
        for col, t_idx, t_label in [(0, 0, "t=0"), (1, -1, f"t={T}")]:
            ax = axes[row, col]
            current_phases = data["hist"][t_idx]  # 获取当前时刻的相位
            pos = get_fixed_layout(current_phases)

            # 绘制 Clock 位置文本，颜色与该Clock相位固定颜色一致
            for p in range(L):
                angle = 2 * np.pi * p / L
                ax.text(12.5 * np.cos(angle), 12.5 * np.sin(angle), f"Clock {p}",
                        color=clock_phase_cmap(p / (L - 1)), ha='center', va='center', fontweight='bold', fontsize=10)

            # 绘制连边
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=1.0, edge_color='gray', width=1.0)

            # 绘制节点：node_color 直接使用当前相位值映射到固定的 clock_phase_cmap
            nodes_drawing = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=150,
                                                   node_color=current_phases,  # 关键：节点颜色由其当前相位决定
                                                   cmap=clock_phase_cmap,
                                                   vmin=0, vmax=L - 1,  # 确保颜色映射范围正确
                                                   edgecolors='black', linewidths=0.8)

            ax.set_title(f"{data['label']} ({t_label})\nAmplitude: {data['amp']}", fontsize=14)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect('equal')
            ax.axis('off')

    plt.suptitle(f"Synchronization Comparison by Clock Phase (N={N}, K={K})",
                 fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=5.0)

    save_path = f"../output/best_worst_clock_K{K}_N{N}.png"
    plt.savefig(save_path, dpi=300,bbox_inches='tight')
    print(f"对比追踪图已保存至: {save_path}")
    plt.close()


def save_sync_comparison_animation_mp4(adj_matrix, results, K, N):
    L = CLOCK_LENGTH
    # results[0] 是最好组
    best_history = results[0][3]
    worst_history = results[-1][3]

    # 计算 Best Case 的闪烁序列用于判断截止时间
    FLASH_START = L - (L // DUTY_CYCLE)
    best_flashes = np.sum(best_history >= FLASH_START, axis=1)
    worst_flashes = np.sum(worst_history >= FLASH_START, axis=1)
    # 计算 Best Case 最后的幅值
    final_best_amp = best_flashes[-L:].max() - best_flashes[-L:].min()

    # 计算 Worst Case 最后的幅值
    final_worst_amp = worst_flashes[-L:].max() - worst_flashes[-L:].min()

    # 1. 核心逻辑：只采集 Best Case 的稳定点
    def get_best_stop_time(flashes):
        # 预计算幅值序列
        amps = []
        for t in range(L, len(flashes)):
            seg = flashes[t - L: t]
            amps.append(seg.max() - seg.min())

        # 寻找幅值不再增长的转折点
        # 我们寻找第一个达到“历史最大幅值”并保持稳定的点
        max_amp_found = max(amps)
        for t, a in enumerate(amps):
            if a >= max_amp_found:
                # 找到最大幅值后，再往后延续 40 步作为展示
                return min(t + L + 40, len(flashes))
        return len(flashes)

    end_frame = get_best_stop_time(best_flashes)
    print(f"检测到 Best Case 在 t={end_frame - 40} 达到稳定。动画总帧数: {end_frame}")

    # 2. 设置画布与颜色
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    # 修复警告：使用 matplotlib.colormaps
    cmap = plt.get_cmap('Set1', L)
    G = nx.from_numpy_array(adj_matrix)
    limit = 14

    def update(t):
        for row_axes in axes:
            for ax in row_axes: ax.clear()

        def draw_cell(ax, phases_hist, title, current_t, is_dynamic=False):
            # 如果是动态列，取当前时刻 t；如果是静态列，固定取 t=0
            t_idx = current_t if is_dynamic else 0
            p_data = phases_hist[t_idx]

            # 固定布局
            outer_r, inner_r = 10, 1.2
            pos = {}
            for p in range(L):
                angle = 2 * np.pi * p / L
                cx, cy = outer_r * np.cos(angle), outer_r * np.sin(angle)
                nodes = np.where(p_data == p)[0]
                for i, node in enumerate(nodes):
                    sub_a = 2 * np.pi * i / len(nodes) if len(nodes) > 1 else 0
                    pos[node] = np.array([cx + inner_r * np.cos(sub_a), cy + inner_r * np.sin(sub_a)])

                # 绘制固定位置的 Clock 文本
                ax.text(12.8 * np.cos(angle), 12.8 * np.sin(angle), f"C{p}",
                        color=cmap(p / (L - 1)), ha='center', va='center', fontweight='bold')

            nx.draw_networkx_edges(G, pos, ax=ax, alpha=1.0, edge_color='gray')
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=150,
                                   node_color=p_data, cmap=cmap, vmin=0, vmax=L - 1, edgecolors='black')

            ax.set_title(title, fontsize=13)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect('equal')
            ax.axis('off')

        # 绘制四格：Worst Case 也强制使用 Best Case 的时间进度 t
        draw_cell(axes[0, 0], best_history, f"Best Case Initial (t=0), Final Amp={final_best_amp}", t, False)
        draw_cell(axes[0, 1], best_history, f"Best Case Real-time (t={t})", t, True)
        draw_cell(axes[1, 0], worst_history, f"Worst Case Initial (t=0), Final Amp={final_worst_amp}", t, False)
        draw_cell(axes[1, 1], worst_history, f"Worst Case Real-time (t={t})", t, True)

        plt.suptitle(f"Phase Sync Evolution Comparison (N={N}, K={K})",
                     fontsize=16)

    # 3. 保存动画
    # 取 step=2 步长，让视频生成快一倍，且不影响观看
    ani = animation.FuncAnimation(fig, update, frames=range(0, end_frame, 1), interval=80)

    save_path = f"../output/sync_comparison_K{K}_N{N}.mp4"
    try:
        # writer='ffmpeg' 需要你电脑安装了 ffmpeg
        ani.save(save_path, writer='ffmpeg', fps=3)
        plt.close(fig)
        print(f"动画已保存至: {save_path}")
    except Exception as e:
        print(f"保存失败，请检查是否安装了 ffmpeg: {e}")
        plt.close(fig)


def plot_k_regular_spring_layout_snapshot(adj_matrix, results, K, N, T,
                                          save_path=None, seed=42):
    """
    与 plot_best_worst_by_clock_color 一样的逻辑，但布局改成 spring layout（2D）：
    - best = results[0], worst = results[-1]（results 已按 amp 排序）
    - 2x2: (best t=0, best t=T) / (worst t=0, worst t=T)
    - 节点颜色 = clock 相位（Set1 离散 L 色，保证每个 clock 不同）
    - spring layout 固定一次（整个图一致）
    """
    L = CLOCK_LENGTH
    G = nx.from_numpy_array(adj_matrix)

    # 离散 colormap（Matplotlib 3.7+ 友好）
    clock_phase_cmap = plt.get_cmap('Set1', L)

    # 固定 spring layout（只算一次，保证四张图位置一致）
    pos = nx.spring_layout(G, seed=seed)

    # best / worst
    best_hist = results[0][3]
    worst_hist = results[-1][3]

    best_amp = results[0][0]
    worst_amp = results[-1][0]

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    def draw_one(ax, phases, title):
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.8, edge_color='gray', width=0.8)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=140,
            node_color=phases,  # ✅ 必须是长度 N 的 1D 数组
            cmap=clock_phase_cmap,
            vmin=0, vmax=L - 1,
            edgecolors='black',
            linewidths=0.6
        )
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    # t=0 和 t=T（你原函数用的是 t=0 和 t=-1；这里按你的接口给 T，同时也兼容）
    draw_one(axes[0, 0], best_hist[0], f"Best Case (t=0)\nAmplitude: {best_amp}")
    draw_one(axes[0, 1], best_hist[-1], f"Best Case (t={T})\nAmplitude: {best_amp}")

    draw_one(axes[1, 0], worst_hist[0], f"Worst Case (t=0)\nAmplitude: {worst_amp}")
    draw_one(axes[1, 1], worst_hist[-1], f"Worst Case (t={T})\nAmplitude: {worst_amp}")

    plt.suptitle(f"Spring Layout Snapshot by Clock Phase (N={N}, K={K})", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path is None:
        save_path = f"../output/best_worst_spring_K{K}_N{N}.png"

    plt.savefig(save_path, dpi=300,bbox_inches='tight')
    print(f"Spring 四宫格对比图已保存至: {save_path}")
    plt.close()

    return pos


def save_spring_sync_comparison_animation_mp4(adj_matrix, results, K, N, seed=42):
    """
    生成与 save_sync_comparison_animation_mp4 一样形式的四宫格视频，但布局改为 spring layout：
    - 左列：初始 (t=0) 静态
    - 右列：实时 (t=t) 动态
    - 上行：Best Case
    - 下行：Worst Case
    动画时长判定逻辑：与 save_sync_comparison_animation_mp4 完全一致（基于 Best Case 幅值稳定点）
    """
    L = CLOCK_LENGTH
    G = nx.from_numpy_array(adj_matrix)

    # results[0] 是最好组，results[-1] 是最差组（你前面已按 amp 排序）
    best_history = results[0][3]
    worst_history = results[-1][3]

    # --- 复用你原来的“稳定点截断”逻辑 ---
    FLASH_START = L - (L // DUTY_CYCLE)
    best_flashes = np.sum(best_history >= FLASH_START, axis=1)
    worst_flashes = np.sum(worst_history >= FLASH_START, axis=1)

    final_best_amp = best_flashes[-L:].max() - best_flashes[-L:].min()
    final_worst_amp = worst_flashes[-L:].max() - worst_flashes[-L:].min()

    def get_best_stop_time(flashes):
        amps = []
        for t in range(L, len(flashes)):
            seg = flashes[t - L: t]
            amps.append(seg.max() - seg.min())

        max_amp_found = max(amps)
        for t, a in enumerate(amps):
            if a >= max_amp_found:
                return min(t + L + 40, len(flashes))
        return len(flashes)

    end_frame = get_best_stop_time(best_flashes)
    print(f"检测到 Best Case 在 t={end_frame - 40} 达到稳定。动画总帧数: {end_frame}")

    # --- 弹簧布局：固定一次，用于整个视频（关键：节点位置不随时间变化） ---
    pos = nx.spring_layout(G, seed=seed)

    # 颜色：离散 Set1，确保每个 clock 不同
    cmap = plt.get_cmap('Set1', L)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    def draw_cell(ax, phases_hist, title, current_t, is_dynamic=False):
        ax.clear()
        t_idx = current_t if is_dynamic else 0
        phases = phases_hist[t_idx]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.7, edge_color='gray', width=0.8)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=140,
            node_color=phases,
            cmap=cmap,
            vmin=0, vmax=L - 1,
            edgecolors='black',
            linewidths=0.6
        )

        ax.set_title(title, fontsize=13)
        ax.axis('off')

    def update(t):
        # Best
        draw_cell(axes[0, 0], best_history,
                  f"Best Case Initial (t=0), Final Amp={final_best_amp}",
                  t, is_dynamic=False)
        draw_cell(axes[0, 1], best_history,
                  f"Best Case Real-time (t={t})",
                  t, is_dynamic=True)

        # Worst
        draw_cell(axes[1, 0], worst_history,
                  f"Worst Case Initial (t=0), Final Amp={final_worst_amp}",
                  t, is_dynamic=False)
        draw_cell(axes[1, 1], worst_history,
                  f"Worst Case Real-time (t={t})",
                  t, is_dynamic=True)

        plt.suptitle(f"Spring-Layout Phase Sync Evolution (N={N}, K={K})", fontsize=16)

    ani = animation.FuncAnimation(fig, update, frames=range(0, end_frame, 1), interval=80)

    save_path = f"../output/spring_sync_comparison_K{K}_N{N}.mp4"
    try:
        ani.save(save_path, writer='ffmpeg', fps=3)  # 需要系统安装 ffmpeg
        plt.close(fig)
        print(f"Spring layout 动画已保存至: {save_path}")
    except Exception as e:
        print(f"保存失败，请检查是否安装了 ffmpeg: {e}")
        plt.close(fig)


import json


def _amp_init_worker(args):
    """
    args = (adj_matrix, K, T, seed)
    return: (seed, final_amp, init_phases_list) or None
    """
    adj_matrix, K, T, seed = args
    flash_counts, phase_history = simulate_fireflies_graph_fast(adj_matrix, K, T, seed)
    last_cycle = flash_counts[-2 * CLOCK_LENGTH:]
    final_amp = int(last_cycle.max() - last_cycle.min())
    # 只需要初始相位（减少进程间传输）
    init_phases = phase_history[0].astype(int).tolist()
    return seed, final_amp, init_phases


def collect_initial_clocks_and_save_mp_stream(
        K: int,
        target_amp: int,
        num_runs: int = 10000,
        T: int = 3000,
        output_dir: str = "../output/repro_cases",
        topo_seed: int | None = 12345,
        ensure_connected: bool = True,
        n_workers: int | None = None,
        chunksize: int = 200,
        flush_every: int = 1,  # 命中多少条 flush 一次（避免每条刷盘）
        progress_every: int = 500,  # 处理多少条打印一次进度
):
    import csv
    import time

    os.makedirs(output_dir, exist_ok=True)
    N = Num_firefly

    # 1) 生成连通 K-regular 图（只生成一次）
    rng_topo = np.random.default_rng(topo_seed)
    attempts = 0
    while True:
        attempts += 1
        graph_seed = int(rng_topo.integers(0, 2 ** 32 - 1))
        G = nx.random_regular_graph(d=K, n=N, seed=graph_seed)
        if (not ensure_connected) or nx.is_connected(G):
            break
        if attempts > 3000:
            raise RuntimeError(f"生成连通 K={K} 正则图失败（尝试超过 {attempts} 次）")

    adj_matrix = nx.to_numpy_array(G, dtype=bool)

    # 2) 先把拓扑相关文件写掉（立刻可复现）
    meta = {
        "K": int(K),
        "N": int(N),
        "CLOCK_LENGTH": int(CLOCK_LENGTH),
        "DUTY_CYCLE": int(DUTY_CYCLE),
        "T": int(T),
        "target_amp": int(target_amp),
        "num_runs": int(num_runs),
        "topo_seed": None if topo_seed is None else int(topo_seed),
        "graph_seed_used": int(graph_seed),
        "attempts_to_get_connected": int(attempts),
        "ensure_connected": bool(ensure_connected),
        "n_workers": int(n_workers if n_workers is not None else mp.cpu_count()),
        "chunksize": int(chunksize),
        "flush_every": int(flush_every),
    }

    topo_npz = os.path.join(output_dir, f"topology_K{K}_N{N}.npz")
    np.savez_compressed(
        topo_npz,
        adj_matrix=adj_matrix.astype(np.uint8),
        meta_json=json.dumps(meta, ensure_ascii=False),
    )

    edgelist_csv = os.path.join(output_dir, f"topology_K{K}_N{N}_edgelist.csv")
    edges = np.array(list(G.edges()), dtype=int)
    pd.DataFrame(edges, columns=["u", "v"]).to_csv(edgelist_csv, index=False)

    # 3) 流式写 matches（先写表头，后续追加）
    matches_csv = os.path.join(output_dir, f"matches_K{K}_N{N}_amp{target_amp}.csv")

    # 覆盖写入表头（每次运行重新生成该文件）
    with open(matches_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "amp", "init_phases_json"])

    # 4) 多进程采样：边跑边写
    if n_workers is None:
        n_workers = mp.cpu_count()

    rng = np.random.default_rng()

    def task_iter():
        for _ in range(num_runs):
            seed = int(rng.integers(0, 10 ** 9))
            yield (adj_matrix, K, T, seed)

    processed = 0
    hit = 0
    last_t = time.time()
    buffer_rows = []  # 小缓冲，减少磁盘写次数

    with mp.Pool(processes=n_workers) as pool:
        try:
            for seed, final_amp, init_phases in pool.imap_unordered(
                    _amp_init_worker, task_iter(), chunksize=chunksize
            ):
                processed += 1

                if final_amp == target_amp:
                    hit += 1
                    buffer_rows.append([
                        int(seed),
                        int(final_amp),
                        json.dumps(init_phases)  # init_phases 已是 list[int]
                    ])

                    # 达到 flush 条数就落盘
                    if len(buffer_rows) >= flush_every:
                        with open(matches_csv, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerows(buffer_rows)
                        buffer_rows.clear()

                # 进度打印（按处理次数）
                if progress_every and (processed % progress_every == 0):
                    now = time.time()
                    speed = progress_every / max(now - last_t, 1e-9)
                    last_t = now
                    print(f"[进度] processed={processed}/{num_runs} | hit={hit} | speed≈{speed:.1f} it/s")

        finally:
            # 结束前把缓冲刷盘
            if buffer_rows:
                with open(matches_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(buffer_rows)
                buffer_rows.clear()

            # 如果 break 提前结束了，尽快停掉 pool
            pool.terminate()
            pool.join()

    save_paths = {"topology_npz": topo_npz, "edgelist_csv": edgelist_csv, "matches_csv": matches_csv}

    print(f"[保存完成] 拓扑: {topo_npz}")
    print(f"[保存完成] 边表: {edgelist_csv}")
    print(f"[保存完成] 样本: {matches_csv}")
    print(f"[结果] processed={processed} | 匹配到 {hit} 条 (target_amp={target_amp})")

    # 如果你还需要返回 matches（内存里），这里就不再返回完整列表了（否则又回到占内存）
    # 你可以返回 hit 数量即可；需要画图时从 matches_csv 读取
    return adj_matrix, hit, save_paths


def load_saved_topology_and_matches(topo_npz_path: str, matches_csv_path: str):
    """
    加载已保存的拓扑和匹配样本，便于复现。
    返回：adj_matrix(bool), meta(dict), matches(list[dict])
    """
    data = np.load(topo_npz_path, allow_pickle=True)
    adj_matrix = data["adj_matrix"].astype(bool)
    meta = json.loads(str(data["meta_json"]))

    df = pd.read_csv(matches_csv_path)
    matches = []
    for _, r in df.iterrows():
        init_phases = np.array(json.loads(r["init_phases_json"]), dtype=int)
        matches.append({
            "seed": int(r["seed"]),
            "amp": int(r["amp"]),
            "init_phases": init_phases,
        })

    return adj_matrix, meta, matches


def reproduce_one_case(adj_matrix, K, T, seed):
    """
    用保存的 adj_matrix + seed 复现一条样本（返回 flash_counts, phase_history）
    """
    flash_counts, phase_history = simulate_fireflies_graph_fast(adj_matrix, K, T, seed)
    return flash_counts, phase_history


import math


def load_saved_topology_and_matches(topo_npz_path: str, matches_csv_path: str):
    """
    加载已保存的拓扑和匹配样本（用于复现/绘图）。
    返回：adj_matrix(bool), meta(dict), matches(list[dict])
    """
    data = np.load(topo_npz_path, allow_pickle=True)
    adj_matrix = data["adj_matrix"].astype(bool)
    meta = json.loads(str(data["meta_json"]))

    df = pd.read_csv(matches_csv_path)
    matches = []
    for _, r in df.iterrows():
        init_phases = np.array(json.loads(r["init_phases_json"]), dtype=int)
        matches.append({
            "seed": int(r["seed"]),
            "amp": int(r["amp"]),
            "init_phases": init_phases,
        })

    return adj_matrix, meta, matches


def plot_initial_states_spring_grid_from_repro_cases(
        repro_dir: str = "../output/repro_cases",
        K: int | None = None,
        N: int | None = None,
        target_amp: int | None = None,
        topo_npz_path: str | None = None,
        matches_csv_path: str | None = None,
        max_samples: int | None = 36,
        seed_layout: int = 42,
        max_cols: int = 6,
        node_size: int = 140,
        edge_alpha: float = 0.6,
        edge_width: float = 0.8,
        figsize_per_cell: float = 3.2,
        save_path: str | None = None,
        show: bool = False,
        run_firefly: bool = False,
        T_sim: int = 300,
        tail: int = 30,
        heatmap_subdir: str = "steady_heatmaps",
        max_heatmaps: int | None = None,
        snapshot_back_T: int = 30,
        neighbor_csv_subdir: str = "neighbor_clock_csv"

):
    """
    从 repro_cases 目录读取 topology + matches，然后把所有匹配样本的 init_phases
    以 spring-layout 网格形式画在一个 figure 里。

    你可以两种方式指定输入：
    1) 直接给 topo_npz_path & matches_csv_path（最稳）
    2) 给 repro_dir + (K,N,target_amp) 让它自动按命名规则去找：
         topology_K{K}_N{N}.npz
         matches_K{K}_N{N}_amp{target_amp}.csv
    """
    import glob
    # ----------- 1) 解析文件路径 -----------
    if topo_npz_path is None:
        if K is None:
            raise ValueError("未提供 topo_npz_path 时，必须提供 K")
        if N is None:
            # 允许不传 N：尝试从目录中找一个 topology_K{K}_N*.npz
            cand = sorted(glob.glob(os.path.join(repro_dir, f"topology_K{K}_N*.npz")))
            if not cand:
                raise FileNotFoundError(f"找不到 topology_K{K}_N*.npz 于 {repro_dir}")
            topo_npz_path = cand[0]
        else:
            topo_npz_path = os.path.join(repro_dir, f"topology_K{K}_N{N}.npz")

    if matches_csv_path is None:
        if target_amp is None or K is None:
            raise ValueError("未提供 matches_csv_path 时，必须提供 K 和 target_amp")
        if N is None:
            # 从 topo_npz 文件名里推断 N（如果没传）
            base = os.path.basename(topo_npz_path)
            # 期望类似 topology_K13_N36.npz
            try:
                N = int(base.split("_N")[1].split(".")[0])
            except Exception:
                raise ValueError("无法从 topo_npz_path 推断 N，请显式传入 N")
        matches_csv_path = os.path.join(repro_dir, f"matches_K{K}_N{N}_amp{target_amp}.csv")

    if not os.path.exists(topo_npz_path):
        raise FileNotFoundError(f"topology 文件不存在: {topo_npz_path}")
    if not os.path.exists(matches_csv_path):
        raise FileNotFoundError(f"matches 文件不存在: {matches_csv_path}")

    # ----------- 2) 加载数据 -----------
    adj_matrix, meta, matches = load_saved_topology_and_matches(topo_npz_path, matches_csv_path)

    # 兼容：如果没传 K/N，用 meta 补齐
    K = int(meta.get("K", K if K is not None else 0))
    N = int(meta.get("N", adj_matrix.shape[0]))
    L = int(meta.get("CLOCK_LENGTH", CLOCK_LENGTH))

    if len(matches) == 0:
        print("matches.csv 里没有任何样本可画。")
        return None

    # 限制样本数，避免 figure 太大
    if max_samples is not None:
        matches = matches[:max_samples]

    # ----------- 3) 固定 spring layout 并绘图 -----------
    G = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(G, seed=seed_layout)
    cmap = plt.get_cmap("Set1", L)

    M = len(matches)
    ncols = min(max_cols, math.ceil(math.sqrt(M)))
    nrows = math.ceil(M / ncols)

    fig_w = ncols * figsize_per_cell
    fig_h = nrows * figsize_per_cell
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.axis("off")

        if idx >= M:
            ax.set_visible(False)
            continue

        m = matches[idx]
        phases = m["init_phases"]
        amp = m.get("amp", None)
        sd = m.get("seed", None)

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=edge_alpha, edge_color="gray", width=edge_width)
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_size=node_size,
            node_color=phases,
            cmap=cmap, vmin=0, vmax=L - 1,
            edgecolors="black", linewidths=0.6
        )

        ax.set_title(f"#{idx}  seed={sd}\ninit  amp={amp}", fontsize=10)

    fig.suptitle(
        f"Initial States (K={K}, N={N}, L={L}, samples={M})\n"
        f"topo: {os.path.basename(topo_npz_path)} | matches: {os.path.basename(matches_csv_path)}",
        fontsize=14
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])

    if save_path is None:
        # 默认保存到同目录，命名更明确
        save_path = os.path.join(
            repro_dir,
            f"init_states_spring_grid_K{K}_N{N}_amp{int(matches[0]['amp'])}_M{M}.png"
        )

    plt.savefig(save_path, dpi=300,bbox_inches='tight')
    print(f"初始状态网格图已保存至: {save_path}")

    # =========================================================
    # Part C: 对每个初始状态运行 firefly，并绘制稳态 clock 热力图
    # =========================================================
    heatmap_paths = []
    if run_firefly:
        hm_dir = os.path.join(repro_dir, heatmap_subdir)
        os.makedirs(hm_dir, exist_ok=True)

        # 最多生成多少张热力图
        M2 = len(matches) if max_heatmaps is None else min(len(matches), max_heatmaps)

        print(f"[Firefly] 开始生成稳态热力图：{M2} 个样本 | T_sim={T_sim} | tail={tail}")

        for idx in range(M2):
            m = matches[idx]
            phases0 = m["init_phases"]
            sd = m.get("seed", None)
            amp = m.get("amp", None)

            _, ph_hist = simulate_fireflies_graph_with_init_phases(
                adj_matrix=adj_matrix,
                K=K,
                T=T_sim,
                init_phases=phases0
            )

            title = f"K{K}_N{N}_amp{amp}_idx{idx}_seed{sd}"
            png = plot_clock_steady_heatmap_from_phase_history(
                ph_hist,
                title=title,
                output_dir=hm_dir,
                L=L,
                tail=tail
            )
            heatmap_paths.append(png)

            # ---------- 新增：导出 t = T_sim - T 时刻的邻居 clock 分布 ----------
            t_pick = max(0, T_sim - snapshot_back_T)
            phases_t = ph_hist[t_pick]  # shape=(N,)

            csv_dir = os.path.join(repro_dir, neighbor_csv_subdir)
            os.makedirs(csv_dir, exist_ok=True)

            csv_path = os.path.join(
                csv_dir,
                f"neighbor_clock_K{K}_N{N}_amp{amp}_idx{idx}_seed{sd}_t{t_pick}_and_t{t_pick + 1}.csv"
            )

            export_neighbor_clock_distribution_t_and_t1_from_history(
                adj_matrix=adj_matrix,
                phase_history=ph_hist,
                t_index=t_pick,
                L=L,
                K=K,
                save_csv_path=csv_path,
                sample_title=f"K{K}_N{N}_amp{amp}_idx{idx}_seed{sd}",
                delimiter=";",
                blank_lines_between=2,
            )

        print(f"[Firefly] 完成：共保存 {len(heatmap_paths)} 张稳态热力图 -> {hm_dir}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if run_firefly:
        return save_path, heatmap_paths
    return save_path


def simulate_fireflies_graph_with_init_phases(adj_matrix, K, T, init_phases):
    """
    使用给定 K-regular 拓扑 adj_matrix 和指定的初始相位 init_phases
    来运行 firefly 逻辑。返回 (flash_counts, phase_history)。

    注意：不依赖 seed；init_phases 会自动 mod L。
    """
    N = adj_matrix.shape[0]
    L = CLOCK_LENGTH
    FLASH_LEN = L // DUTY_CYCLE
    FLASH_START = L - FLASH_LEN

    phases = np.asarray(init_phases, dtype=int).copy()
    if phases.shape[0] != N:
        raise ValueError(f"init_phases 长度不等于 N: {phases.shape[0]} vs {N}")
    phases %= L

    flashing = (phases >= FLASH_START)

    flash_counts = np.zeros(T, dtype=int)
    phase_history = np.zeros((T, N), dtype=int)

    for t in range(T):
        # 记录相位
        phase_history[t] = phases

        # 记录闪烁数
        flash_counts[t] = flashing.sum()

        # 相位自然+1
        phases = phases + 1
        phase_mod = phases % L
        flashing = (phase_mod >= FLASH_START)

        # 触发相位点判断
        TRIGGER_PHASE = FLASH_START + 1
        idxs = np.where((phases % L) == TRIGGER_PHASE)[0]
        delta = np.zeros(N, dtype=int)

        for i in idxs:
            if flashing[adj_matrix[i]].sum() > (K / 2):
                delta[i] = 1

        phases = (phases + delta) % L

    return flash_counts, phase_history


def plot_clock_steady_heatmap_from_phase_history(
        phase_history,
        title,
        output_dir,
        L=CLOCK_LENGTH,
        tail=30,
):
    os.makedirs(output_dir, exist_ok=True)

    phase_history_full = np.asarray(phase_history, dtype=int)
    T_total, N = phase_history_full.shape

    # 取最后 tail 步
    if T_total > tail:
        phase_history = phase_history_full[-tail:, :]
    else:
        phase_history = phase_history_full

    T, N = phase_history.shape

    heatmap = np.zeros((T, L), dtype=int)
    for t in range(T):
        heatmap[t] = np.bincount(phase_history[t], minlength=L)

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

    steady_png = os.path.join(output_dir, title + f"_heatmap_tail{tail}.png")
    plt.savefig(steady_png, dpi=300,bbox_inches='tight')
    plt.close()

    print(f"稳态热力图已保存: {steady_png}")
    return steady_png


def export_neighbor_clock_distribution_t_and_t1_from_history(
        adj_matrix,
        phase_history,
        t_index,
        L,
        K=None,
        save_csv_path="../output/neighbor_clock_distribution_t_and_t1.csv",
        sample_title=None,
        delimiter=";",
        sort_by_clock=True,
        sort_by_node=True,
        blank_lines_between=2,
):
    """
    从真实 phase_history 导出 t 和 t+1 两个时刻的邻居 clock 分布（同一 CSV，中间空行）。
    注意：t+1 取的是 phase_history[t_index+1]（真实模拟结果），不是 phases+1。
    """
    import csv
    import os
    import numpy as np

    phase_history = np.asarray(phase_history, dtype=int)
    T_total, N = phase_history.shape

    if not (0 <= t_index < T_total - 1):
        raise ValueError(f"t_index 必须满足 0 <= t_index <= {T_total-2}，当前={t_index}")

    phases_t = phase_history[t_index]
    phases_t1 = phase_history[t_index + 1]

    out_dir = os.path.dirname(save_csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header = ["sample", "t_index", "node", "node_clock", "degree"] + [f"C{d}" for d in range(L)]

    def build_rows(phases, this_t_index):
        rows = []
        for i in range(N):
            x = int(phases[i])
            neigh = np.where(adj_matrix[i])[0]
            neigh_ph = phases[neigh]

            counts = np.bincount(neigh_ph, minlength=L)

            deg = len(neigh) if K is None else int(K)

            row = [
                "" if sample_title is None else sample_title,
                int(this_t_index),
                int(i),
                int(x),
                int(deg),
            ] + counts.astype(int).tolist()

            rows.append(row)

        if sort_by_clock:
            if sort_by_node:
                rows.sort(key=lambda r: (r[3], r[2]))  # node_clock, node
            else:
                rows.sort(key=lambda r: r[3])

        return rows

    rows_t  = build_rows(phases_t,  t_index)
    rows_t1 = build_rows(phases_t1, t_index + 1)

    with open(save_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)

        writer.writerow(header)
        writer.writerows(rows_t)

        for _ in range(blank_lines_between):
            writer.writerow([])

        writer.writerow(header)
        writer.writerows(rows_t1)

    print(f"已保存真实 t 与 t+1 到同一 CSV: {save_csv_path}")
    return save_csv_path

# ======================================
# 新增：完美匹配添加工具函数
# ======================================
def try_add_perfect_matching(adj_matrix, N, rng, max_attempts=400):
    """
    在补图中寻找一个随机完美匹配，使每个节点恰好增加 1 条边。
    成功返回新邻接矩阵（bool），失败返回 None。
    """
    for _ in range(max_attempts):
        perm = rng.permutation(N).tolist()
        matched = [False] * N
        edges = []
        ok = True

        for u in perm:
            if matched[u]:
                continue
            # 候选：未匹配、非自身、且原图中不存在该边（即在补图中）
            cands = [v for v in range(N)
                     if not matched[v] and v != u and not adj_matrix[u, v]]
            if not cands:
                ok = False
                break
            v = int(rng.choice(cands))
            edges.append((u, v))
            matched[u] = True
            matched[v] = True

        if ok and len(edges) == N // 2 and all(matched):
            new_adj = adj_matrix.copy()
            for u, v in edges:
                new_adj[u, v] = True
                new_adj[v, u] = True
            return new_adj

    return None


# ======================================
# 新增：多进程 worker（固定 init_phases 跑仿真）
# ======================================
def _progressive_candidate_worker(args):
    """
    args = (adj_matrix, K, T, init_phases)
    返回 amp（最后一个完整周期的 max-min）
    """
    adj_matrix, K, T, init_phases = args
    fc, _ = simulate_fireflies_graph_with_init_phases(adj_matrix, K, T, init_phases)
    lc = fc[-2 * CLOCK_LENGTH:]
    return int(lc.max() - lc.min())


# ======================================
# 新增：渐进边增加分析主函数
# ======================================
def progressive_edge_addition_analysis(
        data_dir: str,
        K_start: int = 15,
        target_amp: int = 15,
        N: int = None,
        sample_idx: int = 0,
        num_candidates: int = 100,
        T: int = 300,
        output_dir: str = "../output",
        rng_seed: int = 42,
        n_workers: int = None,
):
    """
    渐进边增加分析
    ──────────────────────────────────────────────
    算法流程：
      1. 从 matches 中取第 sample_idx 条样本，固定 init_phases
      2. 每轮向当前图加一个随机完美匹配（每节点度 +1）→ 生成 num_candidates 个候选
      3. 对每个候选以固定 init_phases 跑仿真，计算 amp
      4. 选 amp 最低且 < N 的拓扑作为下一轮基图
      5. 终止条件：所有候选 amp = N（全同步）或 K = 35
    ──────────────────────────────────────────────
    """
    if N is None:
        N = Num_firefly
    if n_workers is None:
        n_workers = mp.cpu_count()

    # 1. 加载起始拓扑和样本
    topo_npz = os.path.join(data_dir, f"topology_K{K_start}_N{N}.npz")
    matches_csv = os.path.join(data_dir, f"matches_K{K_start}_N{N}_amp{target_amp}.csv")

    print("=" * 60)
    print(f"渐进边增加分析  (N={N}, K_start={K_start}, target_amp={target_amp})")
    print("=" * 60)

    adj_matrix, meta, matches = load_saved_topology_and_matches(topo_npz, matches_csv)

    if len(matches) == 0:
        raise ValueError("matches 为空，请先运行 collect_initial_clocks_and_save_mp_stream")

    chosen = matches[sample_idx]
    init_phases = chosen['init_phases']
    print(f"固定初始状态：样本 #{sample_idx}  seed={chosen['seed']}  amp={chosen['amp']}")

    # 2. 验证起始 amp
    rng = np.random.default_rng(rng_seed)
    K = K_start
    current_adj = adj_matrix.astype(bool).copy()

    fc, _ = simulate_fireflies_graph_with_init_phases(current_adj, K, T, init_phases)
    lc = fc[-2 * CLOCK_LENGTH:]
    init_amp = int(lc.max() - lc.min())
    print(f"K={K:2d} | 确认起始 amp = {init_amp}")
    print("-" * 60)

    history = [{
        'K': K,
        'chosen_amp': init_amp,
        'all_amps': [init_amp],
        'ratio_lt_N': float(init_amp < N),
        'mean_amp': float(init_amp),
        'median_amp': float(init_amp),
        'min_amp': init_amp,
        'max_amp': init_amp,
        'num_valid': 1,
    }]

    # 3. 主循环
    while K < Num_firefly-1:
        K_new = K + 1
        print(f"K={K:2d} → K={K_new:2d} | 生成 {num_candidates} 个候选拓扑 ...", flush=True)

        # 生成候选邻接矩阵
        seen = set()
        candidates = []
        fail_count = 0

        for _ in range(num_candidates):
            cand = try_add_perfect_matching(current_adj, N, rng)
            if cand is None:
                fail_count += 1
                continue
            key = cand.tobytes()
            if key not in seen:
                seen.add(key)
                candidates.append(cand)

        # 如果一个都没找到：最多重试 max_retry 次
        max_retry = 10000
        retry = 0
        while not candidates and retry < max_retry:
            cand = try_add_perfect_matching(current_adj, N, rng)
            if cand is not None:
                candidates.append(cand)
            retry += 1

        K_new = K + 1
        # 如果候选为空
        if not candidates:
            if K == N - 3:
                print(f"{N - 3}->{N - 2}无法构造，直接跳到 K={N - 1}")

                # 补齐到完全图
                comp = ~current_adj
                np.fill_diagonal(comp, False)
                current_adj = current_adj | comp

                K_new = N - 1
                candidates = [current_adj]
            else:
                print("  构造失败，终止")
                break
        else:
            print(f"  实际候选数: {len(candidates)} / {num_candidates}（重复/失败: {num_candidates - len(candidates)}）")

        # 并行仿真（固定 init_phases）
        tasks = [(c, K_new, T, init_phases) for c in candidates]
        with mp.Pool(n_workers) as pool:
            all_amps = pool.map(_progressive_candidate_worker, tasks)

        all_amps = [int(a) for a in all_amps]
        ratio = sum(a < N for a in all_amps) / len(all_amps)
        mean_amp = float(np.mean(all_amps))
        median_amp = float(np.median(all_amps))
        min_amp = int(min(all_amps))
        max_amp = int(max(all_amps))

        # 选 amp 最低且 < N 的拓扑
        best_amp = N + 1
        best_adj = None
        for cand, amp in zip(candidates, all_amps):
            if amp < N and amp < best_amp:
                best_amp = amp
                best_adj = cand.copy()

        chosen_amp = best_amp if best_adj is not None else N

        print(f"  chosen_amp={chosen_amp:2d}  mean={mean_amp:.2f}  "
              f"median={median_amp:.1f}  min={min_amp}  max={max_amp}  "
              f"ratio<{N}={ratio:.1%}  (valid={len(all_amps)})")

        history.append({
            'K': K_new,
            'chosen_amp': chosen_amp,
            'all_amps': all_amps,
            'ratio_lt_N': ratio,
            'mean_amp': mean_amp,
            'median_amp': median_amp,
            'min_amp': min_amp,
            'max_amp': max_amp,
            'num_valid': len(all_amps),
        })

        if best_adj is None:
            print(f"  所有候选均已达到 amp={N}（全同步），停止。")
            break

        current_adj = best_adj
        K = K_new

    # 4. 绘图
    os.makedirs(output_dir, exist_ok=True)
    _plot_progressive_results(history, N, K_start, target_amp, sample_idx, output_dir)

    # 5. 保存 CSV
    rows = []
    for r in history:
        rows.append({
            'K': r['K'],
            'chosen_amp': r['chosen_amp'],
            'mean_amp': round(r['mean_amp'], 4),
            'median_amp': round(r['median_amp'], 4),
            'min_amp': r['min_amp'],
            'max_amp': r['max_amp'],
            'ratio_lt_N': round(r['ratio_lt_N'], 6),
            'num_valid_candidates': r['num_valid'],
        })
    csv_path = os.path.join(
        output_dir,
        f"progressive_edge_K{K_start}_N{N}_amp{target_amp}_sample{sample_idx}.csv"
    )
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"结果 CSV 已保存: {csv_path}")

    return history


# ======================================
# 新增：折线图绘制
# ======================================
def _plot_progressive_results(history, N, K_start, target_amp, sample_idx, output_dir):
    """
    绘制三部分内容：

    1️⃣ 上图：Amplitude vs K
         - chosen
         - mean / median
         - min-max 范围带

    2️⃣ 下图：比例 ratio (amp < N)

    3️⃣ 额外单独输出：
         - 每个 K 的 amp 分布热力图（与 TaskB 风格一致）
         - 同时导出 heatmap CSV
    """
    import matplotlib.ticker as mticker

    os.makedirs(output_dir, exist_ok=True)

    # =========================
    # 折线图部分
    # =========================

    k_vals  = [r['K']           for r in history]
    chosen  = [r['chosen_amp']  for r in history]
    means   = [r['mean_amp']    for r in history]
    medians = [r['median_amp']  for r in history]
    mins    = [r['min_amp']     for r in history]
    maxs    = [r['max_amp']     for r in history]
    ratios  = [r['ratio_lt_N']  for r in history]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(k_vals) * 0.9), 11),
                             sharex=True)
    fig.suptitle(
        f"Progressive Edge Addition Analysis\n"
        f"N={N}, K_start={K_start}, init amp={target_amp}",
        fontsize=14, fontweight='bold'
    )

    # ── 上图：Amplitude ──
    ax1 = axes[0]
    ax1.fill_between(k_vals, mins, maxs,
                     alpha=0.15, color='steelblue', label='Candidate Min-Max Range')
    ax1.plot(k_vals, means,   'b--', lw=1.6, alpha=0.7, label='Mean Amp')
    ax1.plot(k_vals, maxs, 'c:', lw=1.6, alpha=0.8, label='Max Amp')
    ax1.plot(k_vals, chosen,  'o-',  color='royalblue', lw=2.5, ms=8,
             label='Minimal Amp')

    for x, y in zip(k_vals, chosen):
        ax1.annotate(str(y), (x, y),
                     textcoords="offset points", xytext=(0, 9),
                     ha='center', fontsize=9, color='royalblue', fontweight='bold')

    # ax1.axhline(N, color='red', ls='--', lw=1.8, alpha=0.85, label=f'amp = {N}')
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('Amplitude vs Connectivity Degree K', fontsize=12)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, ls=':', alpha=0.5)
    ax1.set_ylim(0, N + 5)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(4))

    # ── 下图：比例 ──
    ax2 = axes[1]
    ax2.fill_between(k_vals, 0, ratios, alpha=0.22, color='crimson')
    ax2.plot(k_vals, ratios, 's-', color='crimson', lw=2.5, ms=8,
             label=f'Proportion (amp < {N})')

    for x, y in zip(k_vals, ratios):
        ax2.annotate(f'{y:.0%}', (x, y),
                     textcoords="offset points", xytext=(0, 9),
                     ha='center', fontsize=9, color='crimson', fontweight='bold')

    ax2.axhline(1.0, color='gray', ls=':', lw=1.2, alpha=0.7)
    ax2.axhline(0.5, color='gray', ls=':', lw=1.2, alpha=0.5)
    ax2.set_xlabel('K  (Connectivity Degree)', fontsize=12)
    ax2.set_ylabel(f'Proportion  (amp < {N})', fontsize=12)
    ax2.set_ylim(-0.05, 1.15)
    ax2.xaxis.set_major_locator(mticker.FixedLocator(k_vals))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.legend(fontsize=9, loc='lower left')
    ax2.grid(True, ls=':', alpha=0.5)

    ax2.set_xticks(k_vals)
    ax2.set_xticklabels([str(k) for k in k_vals], fontsize=10)
    ax2.xaxis.set_major_locator(mticker.FixedLocator(list(k_vals)))

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(
        output_dir,
        f"progressive_edge_K{K_start}_N{N}_amp{target_amp}_sample{sample_idx}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"折线图已保存: {save_path}")

    # =========================
    # 热力图部分（与 TaskB 风格一致）
    # =========================

    amp_min, amp_max = 0, N
    num_bins = N + 1
    bins = np.linspace(amp_min, amp_max, num_bins + 1)

    heat_matrix = np.zeros((num_bins, len(k_vals)))

    for j, r in enumerate(history):
        amps = r.get("all_amps", [r["chosen_amp"]])
        hist, _ = np.histogram(amps, bins=bins)
        heat_matrix[:, j] = hist

    plt.figure(figsize=(12, 6))
    plt.figure(figsize=(12, 6))

    plt.imshow(
        np.log1p(heat_matrix),
        aspect='auto',
        origin='lower',
        cmap='hot'
    )

    plt.colorbar(label="log(count + 1)")

    # 关键：把横轴改为离散 index
    plt.xticks(
        ticks=np.arange(len(k_vals)),
        labels=[str(k) for k in k_vals]
    )

    plt.yticks(
        ticks=np.arange(0, N + 1, 4)
    )

    plt.xlabel("K (Degree)")
    plt.ylabel("amplitude")

    plt.title(
        f"Progressive Edge Addition: Amplitude Distribution Heatmap\n"
        f"N={N}, K_start={K_start}, init amp={target_amp}, sample#{sample_idx}"
    )

    heatmap_png = os.path.join(
        output_dir,
        f"progressive_edge_heatmap_K{K_start}_N{N}_amp{target_amp}_sample{sample_idx}.png"
    )
    plt.savefig(heatmap_png, dpi=300,bbox_inches='tight')
    plt.close()
    print(f"热力图已保存: {heatmap_png}")

    # 保存 CSV（与 TaskB 同格式）
    heat_matrix_flipped = np.flipud(heat_matrix)
    amp_rows = np.arange(num_bins - 1, -1, -1)

    df_heat = pd.DataFrame(
        heat_matrix_flipped,
        index=amp_rows,
        columns=k_vals
    )

    heatmap_csv = os.path.splitext(heatmap_png)[0] + ".csv"
    df_heat.to_csv(
        heatmap_csv,
        sep=";",
        index_label="amplitude_bin"
    )

    print(f"热力图矩阵 CSV 已保存: {heatmap_csv}")

# ======================================
# 主程序入口
# ======================================
if __name__ == "__main__":
    # 执行 Task B
    # taskB_graph()

    # topology_analysis_multiprocess(K=13, num_runs=100)

    target_amp = 15
    K = 15
    data_dir = f"../data/repro_cases/N{Num_firefly}_K{K}_amp{target_amp}"

    # adj, hit, paths = collect_initial_clocks_and_save_mp_stream(
    #     K=K,
    #     target_amp=target_amp,
    #     num_runs=100_0000,
    #     T=300,
    #     output_dir=data_dir,
    #     topo_seed=20260127,
    #     flush_every=1,
    #     progress_every=500,
    # )

    # plot_initial_states_spring_grid_from_repro_cases(
    #     repro_dir=data_dir,
    #     K=K, N=Num_firefly, target_amp=target_amp,
    #     max_samples=50, max_cols=10, seed_layout=42,
    #     run_firefly=True,
    #     T_sim=300,
    #     tail=30,
    #     max_heatmaps=20
    # )

    # ── 渐进边增加分析 ──
    # progressive_edge_addition_analysis(
    #     data_dir=data_dir,
    #     K_start=K,
    #     target_amp=target_amp,
    #     N=Num_firefly,
    #     sample_idx=0,  # 从 matches 中取第 0 条样本
    #     num_candidates=1000,  # 每个 K 生成 100 个候选拓扑
    #     T=300,
    #     output_dir="../data/progressive_edge_addition",
    #     rng_seed=42,
    # )

    taskB_graph("../data/distribution",random_topo=False)
    # taskB_graph("../data/distribution", random_topo=False)
