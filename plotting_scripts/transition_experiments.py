# PLOT SCRIPT FOR TRANSITION EXPERIMENT

from cProfile import label

import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
import pickle

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

T = 10000
t_switch = 1000
flash_proportion = 0.5
noise_level = 0.0

Ns = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
Cs = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]  #

########################################################################################################################
# TODO: THIS IS FOR TRANSITION EXPERIMENT 2 -- in paper we present results of EXPERIMENT 1
########################################################################################################################
experiment_tag = "_2_local"  # "" = official experiment; "_2_local" = second experiment with less runtime

# for param in [0.0, 0.05, 0.1, 0.2]:  #  0.05, 0.1, 0.2, 0.3
####################################################################################################################
# load data and compute heatmap matrix
####################################################################################################################
heatmap_path_00 = (f"/Volumes"
                f"/Data"
                f"/other"
                f"/2026_firefly_synchronization"
                f"/compressed_results_transition_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_heatmap_param_0.0_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}.npz")
heatmap_path_005 = (f"/Volumes"
                f"/Data"
                f"/other"
                f"/2026_firefly_synchronization"
                f"/compressed_results_transition_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_heatmap_param_0.05_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}.npz")
heatmap_path_01 = (f"/Volumes"
                f"/Data"
                f"/other"
                f"/2026_firefly_synchronization"
                f"/compressed_results_transition_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_heatmap_param_0.1_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}.npz")
heatmap_path_02 = (f"/Volumes"
                f"/Data"
                f"/other"
                f"/2026_firefly_synchronization"
                f"/compressed_results_transition_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_heatmap_param_0.2_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}.npz")
if os.path.isfile(heatmap_path_00):
    print(f"{heatmap_path_00} already exists. loading...")
    heatmap_00 = np.load(heatmap_path_00)["arr"]
    heatmap_005 = np.load(heatmap_path_005)["arr"]
    heatmap_01 = np.load(heatmap_path_01)["arr"]
    heatmap_02 = np.load(heatmap_path_02)["arr"]
else:
    save_flash_counts_00 = {}
    save_flash_counts_005 = {}
    save_flash_counts_01 = {}
    save_flash_counts_02 = {}
    for N in Ns:
        save_flash_counts_00[N] = {}
        save_flash_counts_005[N] = {}
        save_flash_counts_01[N] = {}
        save_flash_counts_02[N] = {}
        for C in Cs:
            save_flash_counts_00[N][C] = np.nan
            save_flash_counts_005[N][C] = np.nan
            save_flash_counts_01[N][C] = np.nan
            save_flash_counts_02[N][C] = np.nan
            path = (f"/Volumes"
                    f"/Data"
                    f"/other"
                    f"/2026_firefly_synchronization"
                    f"/transition_experiment{experiment_tag}"
                    f"/N={N}_C={C}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}_k_regular_graph_transition_flash_counts.pkl")
            try:
                flash_count = pd.read_pickle(path)
            
                save_flash_counts_00[N][C] = 0.0
                save_flash_counts_005[N][C] = 0.0
                save_flash_counts_01[N][C] = 0.0
                save_flash_counts_02[N][C] = 0.0
                for run in flash_count[int(N - N * 0.0)].keys():
                    if not (np.max(flash_count[int(N - N * 0.0)][run]) == N):  # all fireflies flash atleast once at the same time -> we track asynchronous runs, so we need to check the runs where this is not the case
                        save_flash_counts_00[N][C] += 1
                    if not (np.max(flash_count[int(N - N * 0.05)][run]) == N):  # all fireflies flash atleast once at the same time -> we track asynchronous runs, so we need to check the runs where this is not the case
                        save_flash_counts_005[N][C] += 1
                    if not (np.max(flash_count[int(N - N * 0.1)][run]) == N):  # all fireflies flash atleast once at the same time -> we track asynchronous runs, so we need to check the runs where this is not the case
                        save_flash_counts_01[N][C] += 1
                    if not (np.max(flash_count[int(N - N * 0.2)][run]) == N):  # all fireflies flash atleast once at the same time -> we track asynchronous runs, so we need to check the runs where this is not the case
                        save_flash_counts_02[N][C] += 1
                print(f"Done loading {path}")
            except:
                save_flash_counts_00[N][C] = np.nan
                save_flash_counts_005[N][C] = np.nan
                save_flash_counts_01[N][C] = np.nan
                save_flash_counts_02[N][C] = np.nan
    
    heatmap_00 = np.zeros((len(Cs), len(Ns)))
    heatmap_005 = np.zeros((len(Cs), len(Ns)))
    heatmap_01 = np.zeros((len(Cs), len(Ns)))
    heatmap_02 = np.zeros((len(Cs), len(Ns)))
    
    for i, C in enumerate(Cs):
        for j, N in enumerate(Ns):
            heatmap_00[i, j] = save_flash_counts_00.get(N, {}).get(C, np.nan)
            heatmap_005[i, j] = save_flash_counts_005.get(N, {}).get(C, np.nan)
            heatmap_01[i, j] = save_flash_counts_01.get(N, {}).get(C, np.nan)
            heatmap_02[i, j] = save_flash_counts_02.get(N, {}).get(C, np.nan)
            
    # store compressed result matrix
    np.savez_compressed(heatmap_path_00, arr=heatmap_00)
    np.savez_compressed(heatmap_path_005, arr=heatmap_005)
    np.savez_compressed(heatmap_path_01, arr=heatmap_01)
    np.savez_compressed(heatmap_path_02, arr=heatmap_02)
    
####################################################################################################################
# Plot the transition experiment
####################################################################################################################

# Define colors (normalized)
start = np.array([154, 160, 167]) / 255  # grey
end = np.array([0, 169, 224]) / 255  # blue

# Create colormap
cmap = LinearSegmentedColormap.from_list(
    "grey_to_blue",
    [start, end]
)

for heatmap, param in zip([heatmap_00, heatmap_005, heatmap_01, heatmap_02], [0.0, 0.05, 0.1, 0.2]):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        heatmap / 1000,  #  normalize by total runs per setting (1000 seeds)
        origin="lower",
        aspect="auto",
        cmap="plasma",
        vmin=0,
        vmax=1,
        # norm=LogNorm(vmin=1e-4, vmax=3)  # avoid log(0)
    )
    
    ax.set_xticks(np.arange(len(Ns)))
    ax.set_xticklabels(Ns)
    
    ax.set_yticks(np.arange(len(Cs)))
    ax.set_yticklabels(Cs)
    
    ax.set_xlabel(f"N | {param * 100} % edges removed")
    ax.set_ylabel("C")
    
    fig.colorbar(im, ax=ax, label="Asynchronous runs / Total runs")
    
    plt.tight_layout()
plt.show()
