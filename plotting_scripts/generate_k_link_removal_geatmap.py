from cProfile import label

import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import pickle
from matplotlib.colors import LinearSegmentedColormap
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

param = 7
base_dir = "/Volumes/Data/other/2026_firefly_synchronization"
# base_dir = "/home/till/PycharmProjects/firefly_two-phase_performance/results/"
T = 10000
Ns = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
Cs = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]  #

experiment_tag = "_local"
heatmap_path = (f"{base_dir}/compressed_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={param}.npz")

heatmap_path_async = (f"{base_dir}/compressed_async_runs_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={param}.npz")

heatmap_path_async_lower_phase = (f"{base_dir}/compressed_async_lower_phase_runs_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={param}.npz")

save_flash_counts = {}
save_flash_counts_async = {}
save_flash_counts_async_lower_phase = {}
for N in Ns:
    save_flash_counts[N] = {}
    save_flash_counts_async[N] = {}
    save_flash_counts_async_lower_phase[N] = {}
    for C in Cs:
        save_flash_counts[N][C] = 0.0
        save_flash_counts_async[N][C] = 0.0
        save_flash_counts_async_lower_phase[N][C] = 0.0
        
        try:
            data = pd.read_pickle(f'{base_dir}/'
                                  f'k_regular_graph{experiment_tag}/'
                                  f'flash_proportion=0.5_qr_threshold=0.5_update_noise={0.0}/'
                                  f'N={N}_C={C}_T={T}_k_regular_graph_flash_counts_8_it.pkl')
            
            # check async runs
            if param > 1.0:
                indicator = param
            else:
                indicator = int(N - (N * param))
            for run in data[indicator].keys():
                if not (np.max(data[indicator][run]) == N):
                    save_flash_counts_async[N][C] += 1
            
            # lower phase
            for run in data[indicator].keys():
                if np.max(data[indicator][run]) <= N * 0.85:
                    save_flash_counts_async_lower_phase[N][C] += 1
            
            # gios approach
            for run in data[indicator].keys():
                save_flash_counts[N][C] += np.max(data[indicator][run]) / N
            print(f"Done loading {N}/{C}")
        
        except FileNotFoundError:
            save_flash_counts[N][C] = np.nan
            save_flash_counts_async[N][C] = np.nan
            save_flash_counts_async_lower_phase[N][C] = np.nan
            print(f"File {N}/{C} not found.")
        except:
            save_flash_counts[N][C] = np.nan
            save_flash_counts_async[N][C] = np.nan
            save_flash_counts_async_lower_phase[N][C] = np.nan
            print(f"path not found: \n"
            f'{base_dir}/'
            f'k_regular_graph{experiment_tag}/'
            f'flash_proportion=0.5_qr_threshold=0.5_update_noise={0.0}/'
            f'N={N}_C={C}_T={T}_k_regular_graph_flash_counts_8_it.pkl')

heatmap = np.zeros((len(Cs), len(Ns)))

heatmap_async = np.zeros((len(Cs), len(Ns)))

heatmap_async_lower_phase = np.zeros((len(Cs), len(Ns)))

for i, C in enumerate(Cs):
    for j, N in enumerate(Ns):
        heatmap[i, j] = save_flash_counts.get(N, {}).get(C, np.nan)
        
        heatmap_async[i, j] = save_flash_counts_async.get(N, {}).get(C, np.nan)
        
        heatmap_async_lower_phase[i, j] = save_flash_counts_async_lower_phase.get(N, {}).get(C, np.nan)

# store compressed result matrix
np.savez_compressed(heatmap_path, arr=heatmap)

np.savez_compressed(heatmap_path_async, arr=heatmap_async)

np.savez_compressed(heatmap_path_async_lower_phase, arr=heatmap_async_lower_phase)

try:
    matplotlib.use('QtAgg')
    norm1 = LogNorm(vmin=1e-3,
                    vmax=1)
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap_async_lower_phase / 1000, cmap="plasma", norm=norm1)
    ax.invert_yaxis()
except:
    sns.heatmap(heatmap_async_lower_phase, annot=True, cmap="coolwarm")

plt.show()
