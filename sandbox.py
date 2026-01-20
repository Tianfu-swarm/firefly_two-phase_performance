import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import constants



CLOCK_LENGTH = 24
DUTY_CYCLE = 2

# ======================================
# 模型函数
# ======================================
def simulate_fireflies( g0, g1, g2, N=150, L=CLOCK_LENGTH, r=0.1, T=1000, seed=1, plot_flag=False):
    FLASH_LEN = L // DUTY_CYCLE
    FLASH_START = L - FLASH_LEN

    rng = np.random.default_rng(seed)
    phase_history = np.zeros((T, N), dtype=int)
    groups_history = np.zeros((T, 1), dtype=int)

    # 1. 随机位置
    pos = rng.random((N, 2))
    dists = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
    neigh = (dists < r) & (dists > 0)
    avg_neighbors = neigh.sum(axis=1).mean()

    # 2. 初始相位 & 闪烁状态（完全复刻 C）
    phases = rng.integers(0, L, size=N)
    counts = np.bincount(phases, minlength=CLOCK_LENGTH)
    print(f"Inital phases: {counts}")
    
    
    # NOTE this init causes a break of the system
    # phases = np.concatenate([
    #     np.full(N // 5, 0, dtype=int),
    #     np.full(N // 5, 2, dtype=int),
    #     np.full(N // 5, 4, dtype=int),
    #     np.full(N // 5, 6, dtype=int),
    #     np.full(N // 5, 8, dtype=int),
    # ])
    
    # if plot_flag:
        # make figure with 2 subplots
        # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 10))
        # ax[0].hist(phases, bins=np.linspace(0, L, N))
        # ax[0].set_title('Initial phase distribution')
        # ax[0].set_xlabel('Phase')
        # ax[0].set_ylabel('Number of fireflies')
        # plt.show()
    
    
    flashing = (phases >= FLASH_START)

    flash_counts = np.zeros(T, dtype=int)

    for t in range(T):
        phase_history[t] = phases % L   # TODO this operation is probably not necessary?!
        values, counts = np.unique(phase_history[t], return_counts=True)
        groups_history[t] = len(counts)

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
    
    # if plot_flag:
    #     ax[1].hist(phases, bins=np.linspace(0, L, N))
    #     ax[1].set_title('Final phase distribution')
    #     ax[1].set_xlabel('Phase')
    #     ax[1].set_ylabel('Number of fireflies')
    #     # plt.show()
    
    if plot_flag:
        bar_counter = 0
        bar_length = CLOCK_LENGTH / 2
        y_ticks_c = []
        y_ticks_label = []
        
        plt.figure()
        colorlist = list(plt.cm.hsv(np.linspace(0, 1, CLOCK_LENGTH, endpoint=False)))
        # print(phases)
        # exit(111)
        for i in range(CLOCK_LENGTH):
            if i in phases:
                
                y_ticks_c.append(bar_counter)
                y_ticks_label.append(f"G{bar_counter}")
                start = CLOCK_LENGTH - i
                end = start + bar_length
                # we need to draw a vertical line at i + 1 and i +2 to indicate the checking (should be red dashed)
                plt.axvline(x=(start + 1) % CLOCK_LENGTH, color=colorlist[i], linestyle='--')
                plt.axvline(x=(start + 2) % CLOCK_LENGTH, color=colorlist[i], linestyle='--')
                
                if start >= 0 and end <= CLOCK_LENGTH:
                    # no wrap-around
                    plt.barh(
                        y=bar_counter,
                        width=bar_length,
                        left=start,
                        color=colorlist[i]
                    )
                    # annotate the bar with the number of i occuring in phases
                    count_i = np.sum(phases == i)
                    plt.text(start, bar_counter, str(count_i), color='black', ha='center', va='center', fontsize=20)
                elif start >= 0 and end > CLOCK_LENGTH:
                    # wrap-around: split into two bars
                    first_len = CLOCK_LENGTH - start
                    second_len = end - CLOCK_LENGTH
                    
                    # part 1: from i to CLOCK_LENGTH
                    plt.barh(
                        y=bar_counter,
                        width=first_len,
                        left=start,
                        color=colorlist[i]
                    )
                    
                    # part 2: from 0 onward
                    plt.barh(
                        y=bar_counter,
                        width=second_len,
                        left=0,
                        color=colorlist[i]
                    )
                    # annotate the bar with the number of i occuring in phases
                    count_i = np.sum(phases == i)
                    plt.text(start, bar_counter, str(count_i), color='black', ha='center', va='center', fontsize=20)
                else:
                    # wrap-around: split into two bars
                    first_len = CLOCK_LENGTH + start
                    second_len = end
                    
                    # part 1: from i to CLOCK_LENGTH
                    plt.barh(
                        y=bar_counter,
                        width=first_len,
                        left=CLOCK_LENGTH+start,
                        color=colorlist[i]
                    )
                    
                    # part 2: from 0 onward
                    plt.barh(
                        y=bar_counter,
                        width=second_len,
                        left=0,
                        color=colorlist[i]
                    )
                    # annotate the bar with the number of i occuring in phases
                    count_i = np.sum(phases == i)
                    plt.text(CLOCK_LENGTH+start, bar_counter, str(count_i), color='black', ha='center', va='center', fontsize=20)
                
                bar_counter += 1
        
        plt.xlabel("Clock counter")
        plt.xlim(0, CLOCK_LENGTH)
        plt.yticks(y_ticks_c,y_ticks_label)
        plt.title("Final phases of fireflies")
        # plt.show()
    
    return avg_neighbors, flash_counts, pos, phase_history, groups_history



if __name__ == "__main__":
    # for interactive plots
    if platform.system() == "Darwin":
        matplotlib.use('QtAgg')
        plt.rcParams.update({'font.size': 20})
    elif platform.system() == "Linux":
        def is_headless():
            return os.environ.get("DISPLAY", "") == ""
        
        
        if not is_headless():
            matplotlib.use('QtAgg')
        pd.set_option('display.max_rows', None)
    elif platform.system() == "Windows":
        print("Windows is not a proper operating system in general!")
        exit(12)
    # seed = 0
    # while True:
    #     # seed = np.random.randint(0, 100000)
    #     avg_neighbors, flash_counts, pos, phase_history = simulate_fireflies(0, 0, 0, r=2, seed=seed, plot_flag=False)
    #     if flash_counts.max() < 150:
    #         print(f"{seed},")
    #         # break
    #     seed += 1
    # exit(12)
    # for g0 in range(CLOCK_LENGTH):
    #     for g1 in range(CLOCK_LENGTH):
    #         for g2 in range(CLOCK_LENGTH):
    #             avg_neighbors, flash_counts, pos, phase_history = simulate_fireflies(g0, g1, g2, r=2, seed=57185)
    #             # check if max flash_counts < 150
    #             if flash_counts.max() < 150:
    #                 print(f"g0: {g0}, g1: {g1}, g2: {g2}, max flash_counts: {flash_counts.max()}")
    
    for seed in constants.SEEDS_BREAKING_24:
        avg_neighbors, flash_counts, pos, phase_history, groups_history = simulate_fireflies(0,0,0,r=2,T=1000, seed=seed, plot_flag=True)
    
        plt.figure()
        plt.plot(groups_history)
        plt.xlabel("time")
        plt.ylabel("number of  sub groups")
        plt.show()
        
    
    print(f"phase_history {phase_history.shape}")
    for t in range(990, 1000):
        counts = np.bincount(phase_history[t], minlength=CLOCK_LENGTH)
        print(f"Time {t}: {counts}")
    
    # plt.plot(flash_counts)
    # plt.title('Currently flashing fireflies over time')
    # plt.xlabel('Timestep')
    # plt.ylabel('Flashing fireflies')
    # plt.ylim(0, 155)
    # plt.show()