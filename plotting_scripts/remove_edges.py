from cProfile import label

import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


C = 10
N = 100
T = 10000
t_switch = 1000
flash_proportion = 0.5
noise_level = 0.0
# print(f"keys 1 {flash_count.keys()}")
# print(flash_count[90].keys())
param = 3

for param in [0,1,2,3]:

    Ns = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    Cs = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]  # , 54, 58, 62, 66, 70
    use = "r"  # "k" or "r"
    # load data
    save_flash_counts = {}
    for N in Ns:
        save_flash_counts[N] = {}
        for C in Cs:
            save_flash_counts[N][C] = np.nan
            try:
                with open(
                    f'/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={C}_T={T}_flash_proportion={flash_proportion}_update_noise={noise_level}_k_regular_graph_transition_flash_counts.pkl',
                    'rb') as f:
                    flash_count = pickle.load(f)
                # for reduced_graph in flash_count.keys():
                #     for run in flash_count[reduced_graph].keys():
                #         plt.plot(flash_count[reduced_graph][run][:5000], label=f"Flash count")
                #         plt.vlines(t_switch, 0, N, colors='r', linestyles='dashed', label='t_switch')
                #         plt.title(f"{N - 1} -> k={reduced_graph} | N={N}, C={C} | T={T} | seed={run}")
                #         plt.xlabel("time")
                #         plt.ylabel("number of flashes")
                #         plt.legend()
                #         plt.show()
                for reduced_graph in [list(flash_count.keys())[param]]:
                    save_flash_counts[N][C] = 0.0
                    for run in flash_count[reduced_graph].keys():
                        if np.max(flash_count[reduced_graph][run]) < (N * 0.9):
                            if save_flash_counts[N][C] is np.nan:
                                save_flash_counts[N][C] = 1
                            else:
                                save_flash_counts[N][C] += 1
                            # save_flash_counts[N][C] += 1 if save_flash_counts[N][C] is not np.nan else 1
                        # save_flash_counts[N][C] += 1 if np.max(flash_count[reduced_graph][run]) >= (N * 0.9) else 0
            except:
                print(f"File not found....")
                continue
    
    
    heatmap = np.zeros((len(Cs), len(Ns)))
    
    for i, C in enumerate(Cs):
        for j, N in enumerate(Ns):
            heatmap[i, j] = save_flash_counts.get(N, {}).get(C, np.nan)
            # if heatmap[i, j] == 0.0:
            #     heatmap[i, j] = np.nan
    
    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        heatmap / 10,  # normalize by total runs (40)
        origin="lower",
        aspect="auto",
        cmap="plasma",
        vmin=0,
        vmax=1,
        # norm=LogNorm(vmin=1e-4, vmax=1)  # avoid log(0)
    )
    
    ax.set_xticks(np.arange(len(Ns)))
    ax.set_xticklabels(Ns)
    
    ax.set_yticks(np.arange(len(Cs)))
    ax.set_yticklabels(Cs)
    
    ax.set_xlabel(f"N | {param+1} x 10 edges removed")
    ax.set_ylabel("C")
    
    fig.colorbar(im, ax=ax, label="Asynchronous runs / Total runs")
    
    plt.tight_layout()
    plt.show()