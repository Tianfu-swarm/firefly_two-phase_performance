from cProfile import label

import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

T = 10000
Ns = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
Cs = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]  #

experiment_tag = "_local"
heatmap_path_00 = (f"/home/till/PycharmProjects/firefly_two-phase_performance/results/"
                f"compressed_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={0.0}.npz")
heatmap_path_005 = (f"/home/till/PycharmProjects/firefly_two-phase_performance/results/"
                f"compressed_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={0.05}.npz")
heatmap_path_01 = (f"/home/till/PycharmProjects/firefly_two-phase_performance/results/"
                f"compressed_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={0.1}.npz")
heatmap_path_02 = (f"/home/till/PycharmProjects/firefly_two-phase_performance/results/"
                f"compressed_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={0.2}.npz")

heatmap_path_00_async = (f"/home/till/PycharmProjects/firefly_two-phase_performance/results/"
                f"compressed_async_runs_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={0.0}.npz")
heatmap_path_005_async = (f"/home/till/PycharmProjects/firefly_two-phase_performance/results/"
                f"compressed_async_runs_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={0.05}.npz")
heatmap_path_01_async = (f"/home/till/PycharmProjects/firefly_two-phase_performance/results/"
                f"compressed_async_runs_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={0.1}.npz")
heatmap_path_02_async = (f"/home/till/PycharmProjects/firefly_two-phase_performance/results/"
                f"compressed_async_runs_results_k_graph_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={0.2}.npz")

if os.path.isfile(heatmap_path_00):
    print(f"{heatmap_path_00} already exists. loading...")
    heatmap_00 = np.load(heatmap_path_00)["arr"]
    heatmap_005 = np.load(heatmap_path_005)["arr"]
    heatmap_01 = np.load(heatmap_path_01)["arr"]
    heatmap_02 = np.load(heatmap_path_02)["arr"]
    heatmap_00_async = np.load(heatmap_path_00_async)["arr"]
    heatmap_005_async = np.load(heatmap_path_005_async)["arr"]
    heatmap_01_async = np.load(heatmap_path_01_async)["arr"]
    heatmap_02_async = np.load(heatmap_path_02_async)["arr"]

else:
    save_flash_counts_00 = {}
    save_flash_counts_005 = {}
    save_flash_counts_01 = {}
    save_flash_counts_02 = {}
    save_flash_counts_00_async = {}
    save_flash_counts_005_async = {}
    save_flash_counts_01_async = {}
    save_flash_counts_02_async = {}
    for N in Ns:
        save_flash_counts_00[N] = {}
        save_flash_counts_005[N] = {}
        save_flash_counts_01[N] = {}
        save_flash_counts_02[N] = {}
        save_flash_counts_00_async[N] = {}
        save_flash_counts_005_async[N] = {}
        save_flash_counts_01_async[N] = {}
        save_flash_counts_02_async[N] = {}
        for C in Cs:
            save_flash_counts_00[N][C] = 0.0
            save_flash_counts_005[N][C] = 0.0
            save_flash_counts_01[N][C] = 0.0
            save_flash_counts_02[N][C] = 0.0
            save_flash_counts_00_async[N][C] = 0.0
            save_flash_counts_005_async[N][C] = 0.0
            save_flash_counts_01_async[N][C] = 0.0
            save_flash_counts_02_async[N][C] = 0.0
            
            try:
                data = pd.read_pickle('~/PycharmProjects/firefly_two-phase_performance/results/'
                                   f'k_regular_graph{experiment_tag}/'
                                   f'flash_proportion=0.5_qr_threshold=0.5_update_noise={0.0}/'
                                   f'N={N}_C={C}_T=10000_k_regular_graph_flash_counts.pkl')
                
                # check async runs
                for run in data[int(N)].keys():
                    if not (np.max(data[int(N)][run]) == N):
                        save_flash_counts_00_async[N][C] += 1
                    if not (np.max(data[int(N - (N * 0.05))][run]) == N):
                        save_flash_counts_005_async[N][C] += 1
                    if not (np.max(data[int(N - (N * 0.1))][run]) == N):
                        save_flash_counts_01_async[N][C] += 1
                    if not (np.max(data[int(N - (N * 0.2))][run]) == N):
                        save_flash_counts_02_async[N][C] += 1
                        
                # gios approach
                for run in data[int(N)].keys():
                    # if np.max(data[int(N)][run])
                    save_flash_counts_00[N][C] += np.max(data[int(N)][run]) / N
                    # if C == 10 and N == 100:
                    #     counts, bins = np.histogram(data[int(N)][run])
                    #     print("Counts:", counts)
                    #     print("Bins:", bins)
                    #     exit(12)
                    # print(f"{run}: {np.max(data[int(N)][run])} N={N}, C={C}, T={T}")
                    save_flash_counts_005[N][C] += np.max(data[int(N - (N * 0.05))][run]) / N
                    save_flash_counts_01[N][C] += np.max(data[int(N - (N * 0.1))][run]) / N
                    save_flash_counts_02[N][C] += np.max(data[int(N - (N * 0.2))][run]) / N
                # print(f"Done loading {N}/{C}")
                
            except FileNotFoundError:
                save_flash_counts_00[N][C] = np.nan
                save_flash_counts_005[N][C] = np.nan
                save_flash_counts_01[N][C] = np.nan
                save_flash_counts_02[N][C] = np.nan
                save_flash_counts_00_async[N][C] = np.nan
                save_flash_counts_005_async[N][C] = np.nan
                save_flash_counts_01_async[N][C] = np.nan
                save_flash_counts_02_async[N][C] = np.nan
                print(f"File {N}/{C} not found.")
    
    heatmap_00 = np.zeros((len(Cs), len(Ns)))
    heatmap_005 = np.zeros((len(Cs), len(Ns)))
    heatmap_01 = np.zeros((len(Cs), len(Ns)))
    heatmap_02 = np.zeros((len(Cs), len(Ns)))
    
    heatmap_00_async = np.zeros((len(Cs), len(Ns)))
    heatmap_005_async = np.zeros((len(Cs), len(Ns)))
    heatmap_01_async = np.zeros((len(Cs), len(Ns)))
    heatmap_02_async = np.zeros((len(Cs), len(Ns)))

    for i, C in enumerate(Cs):
        for j, N in enumerate(Ns):
            heatmap_00[i, j] = save_flash_counts_00.get(N, {}).get(C, np.nan)
            heatmap_005[i, j] = save_flash_counts_005.get(N, {}).get(C, np.nan)
            heatmap_01[i, j] = save_flash_counts_01.get(N, {}).get(C, np.nan)
            heatmap_02[i, j] = save_flash_counts_02.get(N, {}).get(C, np.nan)
            
            heatmap_00_async[i, j] = save_flash_counts_00_async.get(N, {}).get(C, np.nan)
            heatmap_005_async[i, j] = save_flash_counts_005_async.get(N, {}).get(C, np.nan)
            heatmap_01_async[i, j] = save_flash_counts_01_async.get(N, {}).get(C, np.nan)
            heatmap_02_async[i, j] = save_flash_counts_02_async.get(N, {}).get(C, np.nan)
    
    # store compressed result matrix
    np.savez_compressed(heatmap_path_00, arr=heatmap_00)
    np.savez_compressed(heatmap_path_005, arr=heatmap_005)
    np.savez_compressed(heatmap_path_01, arr=heatmap_01)
    np.savez_compressed(heatmap_path_02, arr=heatmap_02)
    
    np.savez_compressed(heatmap_path_00_async, arr=heatmap_00_async)
    np.savez_compressed(heatmap_path_005_async, arr=heatmap_005_async)
    np.savez_compressed(heatmap_path_01_async, arr=heatmap_01_async)
    np.savez_compressed(heatmap_path_02_async, arr=heatmap_02_async)

print(heatmap_00)

# Define colors (normalized)
# start = np.array([154, 160, 167]) / 255  # grey
# end = np.array([0, 169, 224]) / 255  # blue
#
# # Create colormap
# cmap = LinearSegmentedColormap.from_list(
#     "grey_to_blue",
#     [start, end]
# )
#
# for heatmap, param in zip([heatmap_00, heatmap_005, heatmap_01, heatmap_02], [0.0, 0.05, 0.1, 0.2]):
#     fig, ax = plt.subplots(figsize=(8, 5))
#     im = ax.imshow(
#         heatmap / 100,  # normalize by total runs per setting (1000 seeds)
#         origin="lower",
#         aspect="auto",
#         cmap="plasma",
#         vmin=0,
#         vmax=1,
#         # norm=LogNorm(vmin=1e-4, vmax=3)  # avoid log(0)
#     )
#
#     ax.set_xticks(np.arange(len(Ns)))
#     ax.set_xticklabels(Ns)
#
#     ax.set_yticks(np.arange(len(Cs)))
#     ax.set_yticklabels(Cs)
#
#     ax.set_xlabel(f"N | {param * 100} % noise on the update rule")
#     ax.set_ylabel("C")
#
#     fig.colorbar(im, ax=ax, label="Asynchronous runs / Total runs")
#
#     plt.tight_layout()
# plt.show()
