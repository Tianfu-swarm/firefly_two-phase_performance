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
experiment_tag = "_2_local"

# for param in [0.0]:  # , 0.05, 0.1, 0.2
#     ####################################################################################################################
#     # load data and compute heatmap matrix
#     ####################################################################################################################
#     heatmap_path = (f"/Volumes"
#                     f"/Data"
#                     f"/other"
#                     f"/2026_firefly_synchronization"
#                     f"/compressed_results_transition_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_heatmap_param_{param}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}.npz")
#     if os.path.isfile(heatmap_path):
#         print(f"{heatmap_path} already exists. loading...")
#         heatmap = np.load(heatmap_path)["arr"]
#         print(heatmap.shape)
#         print(heatmap)
#     else:
#         save_flash_counts = {}
#         for N in Ns:
#             save_flash_counts[N] = {}
#             for C in Cs:
#                 save_flash_counts[N][C] = np.nan
#                 path = (f"/Volumes"
#                         f"/Data"
#                         f"/other"
#                         f"/2026_firefly_synchronization"
#                         f"/transition_experiment{experiment_tag}"
#                         f"/N={N}_C={C}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}_k_regular_graph_transition_flash_counts.pkl")
#                 try:
#                     flash_count = pd.read_pickle(path)
#
#                     save_flash_counts[N][C] = 0.0
#                     for run in flash_count[int(N - N * param)].keys():
#                         # if np.max(flash_count[int(N - N * param)][run]) <= (N * 0.9):
#                         # print(f"{N}: {np.max(flash_count[int(N - N * param)][run])}")
#                         if not (np.max(flash_count[int(N - N * param)][
#                                            run]) == N):  # all fireflies flash atleast once at the same time -> we track asynchronous runs, so we need to check the runs where this is not the case
#                             save_flash_counts[N][C] += 1
#                             # save_flash_counts[N][C] += 1 if save_flash_counts[N][C] is not np.nan else 1
#                             # save_flash_counts[N][C] += 1 if np.max(flash_count[reduced_graph][run]) >= (N * 0.9) else 0
#                     print(f"Done loading {path}")
#                 except:
#                     save_flash_counts[N][C] = np.nan
#
#         heatmap = np.zeros((len(Cs), len(Ns)))
#
#         for i, C in enumerate(Cs):
#             for j, N in enumerate(Ns):
#                 heatmap[i, j] = save_flash_counts.get(N, {}).get(C, np.nan)
#                 # if heatmap[i, j] == 0.0:
#                 #     heatmap[i, j] = np.nan
#
#         # store compressed result matrix
#         heatmap_path = (f"/Volumes"
#                         f"/Data"
#                         f"/other"
#                         f"/2026_firefly_synchronization"
#                         f"/compressed_results_transition_experiment{experiment_tag}_N={np.min(Ns)}_{np.max(Ns)}_C={np.min(Cs)}_{np.max(Cs)}_heatmap_param_{param}_T={T}_flash_proportion=0.5_qr_threshold=0.5_update_noise={noise_level}.npz")
#         np.savez_compressed(heatmap_path, arr=heatmap)
#         print(heatmap.shape)
#         print(heatmap)
#

fig, axs = plt.subplots(2, 4, figsize=(12, 6))

heatmaps = []

# --- Load all heatmaps first ---
heatmaps.append(pd.read_csv('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise=0.0.csv', header=None).values)
heatmaps.append(pd.read_csv('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise=0.05.csv', header=None).values)
heatmaps.append(pd.read_csv('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise=0.1.csv', header=None).values)
heatmaps.append(pd.read_csv('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise=0.2.csv', header=None).values)

try:
    heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_results_transition_experiment_2_local_N=50_200_C=10_70_heatmap_param_0.0_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.0.npz")["arr"] / 1000)
    heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_results_transition_experiment_2_local_N=50_200_C=10_70_heatmap_param_0.05_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.0.npz")["arr"] / 1000)
    heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_results_transition_experiment_2_local_N=50_200_C=10_70_heatmap_param_0.1_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.0.npz")["arr"] / 1000)
    heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_results_transition_experiment_2_local_N=50_200_C=10_70_heatmap_param_0.2_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.0.npz")["arr"] / 1000)
except FileNotFoundError:
    pass

# # --- Global color scaling ---
# heatmaps = [np.where(h <= 0, 1e-6, h) for h in heatmaps]
# vmin = 1e-6  # min(h.min() for h in heatmaps)
# vmax = 1  # max(h.max() for h in heatmaps)
# norm = LogNorm(vmin=vmin, vmax=vmax)
#
# # --- Plot all heatmaps ---
# ims = []
# for i, ax in enumerate(axs.flat):
#     try:
#         im = ax.imshow(heatmaps[i], cmap="plasma", norm=norm)  # norm=norm, vmin=vmin, vmax=vmax
#         ims.append(im)
#
#         ax.set_xticks(np.arange(len(Ns)))
#         ax.set_xticklabels(Ns)
#         ax.set_yticks(np.arange(len(Cs)))
#         ax.set_yticklabels(Cs)
#         ax.invert_yaxis()
#     except:
#         pass
# plt.tight_layout()
#
#
# # --- Separate colorbar figure ---
# fig_cb, ax_cb = plt.subplots(figsize=(2, 6))
#
# cbar = fig_cb.colorbar(ims[0], cax=ax_cb)
# cbar.set_label("Asynchronus runs")  # optional label
#
# plt.show()

# fig, axs = plt.subplots(2, 4, figsize=(12, 6))

# --- Split heatmaps into two groups ---
heatmaps_1 = heatmaps[:4]
heatmaps_2 = heatmaps[4:8]

# --- Avoid zeros for LogNorm ---
heatmaps_1 = [np.where(h <= 0, 1e-3, h) for h in heatmaps_1]
heatmaps_2 = [np.where(h <= 0, 1e-6, h) for h in heatmaps_2]
# --- Separate normalization ---
norm1 = LogNorm(vmin=min(h.min() for h in heatmaps_1),
                vmax=max(h.max() for h in heatmaps_1))

# norm2 = LogNorm(vmin=min(h.min() for h in heatmaps_2),
#                 vmax=max(h.max() for h in heatmaps_2))
vmin = 1e-3  # min(h.min() for h in heatmaps)
vmax = 1  # max(h.max() for h in heatmaps)

ims1, ims2 = [], []

# --- Plot first row ---
for i in range(4):
    im = axs[0, i].imshow(heatmaps_1[i], cmap="plasma", norm=norm1)
    ims1.append(im)

# --- Plot second row ---
for i in range(4):
    im = axs[1, i].imshow(heatmaps_2[i], cmap="plasma", vmin=vmin, vmax=vmax)
    ims2.append(im)

# --- Axis formatting ---
for ax in axs.flat:
    # ax.set_xticks(np.arange(len(Ns)))
    # ax.set_xticklabels(Ns)
    # ax.set_yticks(np.arange(len(Cs)))
    # ax.set_yticklabels(Cs)
    ax.set_axis_off()
    ax.invert_yaxis()

plt.tight_layout()

# --- Create two colorbars ---
fig_cb, ax_cb = plt.subplots(figsize=(2, 6))

cbar1 = fig_cb.colorbar(ims1[0], cax=ax_cb)
cbar1.set_label("Asynchronus runs")  # optional label
# cbar1 = fig.colorbar(ims1[0], ax=axs[0, :], fraction=0.025, pad=0.04)
# cbar1.set_label("Asynchronous runs (set 1)")

# cbar2 = fig.colorbar(ims2[0], ax=axs[1, :], fraction=0.025, pad=0.04)
# cbar2.set_label("Asynchronous runs (set 2)")
fig_cb, ax_cb = plt.subplots(figsize=(2, 6))

cbar2 = fig_cb.colorbar(ims2[0], cax=ax_cb)
cbar2.set_label("Asynchronus runs")  # optional label

plt.show()