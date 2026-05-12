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
x_tick_labels = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
y_tick_labels = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]
ticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
experiment_tag = "_2_local"
show_case = "all"  # "odd", "even", "all"

fig, axs = plt.subplots(2, 10, figsize=(14, 7))

heatmaps = []

# --- Load all heatmaps first ---
# lower phase approach
heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.0.npz')["arr"] / 1000)
heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.1.npz')["arr"] / 1000)
heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.2.npz')["arr"] / 1000)
heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.3.npz')["arr"] / 1000)
heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.4.npz')["arr"] / 1000)
heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.5.npz')["arr"] / 1000)
heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.6.npz')["arr"] / 1000)
heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.7.npz')["arr"] / 1000)
heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.8.npz')["arr"] / 1000)
heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.9.npz')["arr"] / 1000)
# heatmaps.append(np.load('/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.9.npz')["arr"] / 1000)
# lower phase approach
heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.0.npz")["arr"] / 1000)
heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.1.npz")["arr"] / 1000)
heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.2.npz")["arr"] / 1000)
heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.3.npz")["arr"] / 1000)
heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.4.npz")["arr"] / 1000)
heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=5000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.5.npz")["arr"] / 1000)
heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.6.npz")["arr"] / 1000)
heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.7.npz")["arr"] / 1000)
heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.8.npz")["arr"] / 1000)
heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.9.npz")["arr"] / 1000)
# heatmaps.append(np.load("/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=1.0.npz")["arr"] / 1000)


# --- Split heatmaps into two groups ---
heatmaps_1 = heatmaps[:10]
heatmaps_2 = heatmaps[10:20]

if show_case == "odd":
    # --- filtering ----
    for i in range(len(heatmaps_1)):
        heatmaps_1[i] = heatmaps_1[i][:, 0::2]
        heatmaps_1[i] = heatmaps_1[i][0::2, :]
        ticks_reduced = ticks[:8]
        x_tick_labels_reduced = x_tick_labels[0::2]
        y_tick_labels_reduced = y_tick_labels[0::2]
    
    for i in range(len(heatmaps_2)):
        heatmaps_2[i] = heatmaps_2[i][:, 0::2]
        heatmaps_2[i] = heatmaps_2[i][0::2, :]
        ticks_reduced = ticks[:8]
        x_tick_labels_reduced = x_tick_labels[0::2]
        y_tick_labels_reduced = y_tick_labels[0::2]
elif show_case == "even":
    # --- filtering ----
    for i in range(len(heatmaps_1)):
        heatmaps_1[i] = heatmaps_1[i][:, 1::2]
        heatmaps_1[i] = heatmaps_1[i][1::2, :]
        ticks_reduced = ticks[:8]
        x_tick_labels_reduced = x_tick_labels[1::2]
        y_tick_labels_reduced = y_tick_labels[1::2]
    
    for i in range(len(heatmaps_2)):
        heatmaps_2[i] = heatmaps_2[i][:, 1::2]
        heatmaps_2[i] = heatmaps_2[i][1::2, :]
        ticks_reduced = ticks[:8]
        x_tick_labels_reduced = x_tick_labels[1::2]
        y_tick_labels_reduced = y_tick_labels[1::2]
else:
    ticks_reduced = ticks
    x_tick_labels_reduced = x_tick_labels
    y_tick_labels_reduced = y_tick_labels
    
# --- Avoid zeros for LogNorm ---
heatmaps_1 = [np.where(h <= 0, 1e-3, h) for h in heatmaps_1]
heatmaps_2 = [np.where(h <= 0, 1e-3, h) for h in heatmaps_2]
# --- Separate normalization ---
# norm1 = LogNorm(vmin=min(h.min() for h in heatmaps_1),
#                 vmax=max(h.max() for h in heatmaps_1))
#
# norm2 = LogNorm(vmin=min(h.min() for h in heatmaps_2),
#                 vmax=max(h.max() for h in heatmaps_2))
norm1 = LogNorm(vmin=1e-3,
                vmax=1)

# norm2 = LogNorm(vmin=1e-3,
#                 vmax=1)
vmin = 1e-3  # min(h.min() for h in heatmaps)
vmax = 1  # max(h.max() for h in heatmaps)

ims1, ims2 = [], []

# --- Plot first row ---
for i in range(10):
    im = axs[0, i].imshow(heatmaps_1[i], cmap="plasma", norm=norm1)
    ims1.append(im)

# --- Plot second row ---
for i in range(10):
    im = axs[1, i].imshow(heatmaps_2[i], cmap="plasma", norm=norm1)  #   vmin=vmin, vmax=vmax
    ims2.append(im)

# --- Axis formatting ---
for i, ax in enumerate(axs.flat):
    # ax.set_xticks(np.arange(len(Ns)))
    # ax.set_xticklabels(Ns)
    # ax.set_yticks(np.arange(len(Cs)))
    # ax.set_yticklabels(Cs)
    # ax.set_axis_off()
    ax.invert_yaxis()
    ax.set_xticks(ticks_reduced)
    ax.set_yticks(ticks_reduced)
    ax.set_xticklabels(x_tick_labels_reduced)
    ax.set_yticklabels(y_tick_labels_reduced)
    ax.set_xticklabels(x_tick_labels_reduced, rotation=90)
    if i < 10:
        ax.set_title(fr"σ = {i/10:.1f}")
    if not show_case in ["odd", "even"]:
        ax.tick_params(axis='both', labelsize=5)
row_titles = [
    "Noise introduced in the update rule",
    "Disturbance through link removal",
]
diffs = [0.1, 0.51]

# add titles above each row
for i, title in enumerate(row_titles):
    txt = fig.text(
        0.5,                 # x position (centered)
        0.92 - diffs[i],     # y position
        title,
        ha='center',
        va='center',
        fontsize=14,
        fontweight='bold'
    )
    print(txt.get_position())

cbar_ax = fig.add_axes([0.25, 0.92, 0.5, 0.02])

cbar1 = fig.colorbar(ims1[0], cax=cbar_ax, fraction=0.02, pad=0.02,

    orientation="horizontal")
cbar1.set_label("log(Asynchronous runs/total runs)")

# # --- Create two colorbars ---
# fig_cb, ax_cb = plt.subplots(figsize=(2, 6))
#
# cbar1 = fig_cb.colorbar(ims1[0], cax=ax_cb)
# cbar1.set_label("Asynchronus runs")  # optional label
# cbar1 = fig.colorbar(ims1[0], ax=axs[0, :], fraction=0.025, pad=0.04)
# cbar1.set_label("Asynchronous runs (set 1)")

# cbar2 = fig.colorbar(ims2[0], ax=axs[1, :], fraction=0.025, pad=0.04)
# cbar2.set_label("Asynchronous runs (set 2)")
# fig_cb, ax_cb = plt.subplots(figsize=(2, 6))
#
# cbar2 = fig_cb.colorbar(ims2[0], cax=ax_cb)
# cbar2.set_label("Asynchronus runs")  # optional label
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()