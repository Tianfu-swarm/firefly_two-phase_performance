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

T = 5000
t_switch = 1000
flash_proportion = 0.5
noise_level = 0.0

Ns = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
Cs = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]  #

########################################################################################################################
# TODO: THIS IS FOR TRANSITION EXPERIMENT 2 -- in paper we present results of EXPERIMENT 1
########################################################################################################################

for param in [0.05, 0.1, 0.2, 0.3]:
    ####################################################################################################################
    # load data and compute heatmap matrix
    ####################################################################################################################
    heatmap_path = (f"/Volumes"
                    f"/Data"
                    f"/other"
                    f"/2026_firefly_synchronization"
                    f"/compressed_results_transition_experiment_2_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_heatmap_param_{param}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}.npz")
    if os.path.isfile(heatmap_path):
        print(f"{heatmap_path} already exists. loading...")
        heatmap = np.load(heatmap_path)["arr"]
    else:
        save_flash_counts = {}
        for N in Ns:
            save_flash_counts[N] = {}
            for C in Cs:
                save_flash_counts[N][C] = np.nan
                path = (f"/Volumes"
                        f"/Data"
                        f"/other"
                        f"/2026_firefly_synchronization"
                        f"/transition_experiment_2_local"
                        f"/N={N}_C={C}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}_k_regular_graph_transition_flash_counts.pkl")
                
                flash_count = pd.read_pickle(path)
                
                save_flash_counts[N][C] = 0.0
                for run in flash_count[int(N - N * param)].keys():
                    if np.max(flash_count[int(N - N * param)][run]) < (N * 0.9):
                        if save_flash_counts[N][C] is np.nan:
                            save_flash_counts[N][C] = 1
                        else:
                            save_flash_counts[N][C] += 1
                            # save_flash_counts[N][C] += 1 if save_flash_counts[N][C] is not np.nan else 1
                        # save_flash_counts[N][C] += 1 if np.max(flash_count[reduced_graph][run]) >= (N * 0.9) else 0
                print(f"Done loading {path}")
        
        heatmap = np.zeros((len(Cs), len(Ns)))
        
        for i, C in enumerate(Cs):
            for j, N in enumerate(Ns):
                heatmap[i, j] = save_flash_counts.get(N, {}).get(C, np.nan)
                # if heatmap[i, j] == 0.0:
                #     heatmap[i, j] = np.nan
        
        # store compressed result matrix
        heatmap_path = (f"/Volumes"
                        f"/Data"
                        f"/other"
                        f"/2026_firefly_synchronization"
                        f"/compressed_results_transition_experiment_2_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_heatmap_param_{param}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}.npz")
        np.savez_compressed(heatmap_path, arr=heatmap)
    
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
    
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        heatmap / 1000,  # normalize by total runs per setting (1000 seeds)
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
    
    ax.set_xlabel(f"N | {param * 100} % edges removed")
    ax.set_ylabel("C")
    
    fig.colorbar(im, ax=ax, label="Asynchronous runs / Total runs")
    
    plt.tight_layout()
    plt.show()
