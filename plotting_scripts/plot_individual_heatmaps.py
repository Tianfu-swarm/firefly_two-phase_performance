import numpy as np

import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

import os

# --- Output directory ---
out_dir = os.path.expanduser("~/Downloads")

# --- File paths (add meaningful names!) ---
files = [
    ("lower_phase_noise_0.0", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.0.npz"),
    ("lower_phase_noise_0.1", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.1.npz"),
    ("lower_phase_noise_0.2", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.2.npz"),
    ("lower_phase_noise_0.3", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.3.npz"),
    ("lower_phase_noise_0.4", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.4.npz"),
    ("lower_phase_noise_0.5", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.5.npz"),
    ("lower_phase_noise_0.6", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.6.npz"),
    ("lower_phase_noise_0.7", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.7.npz"),
    ("lower_phase_noise_0.8", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.8.npz"),
    ("lower_phase_noise_0.9", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_0.9.npz"),
    # ("lower_phase_noise_1.0", "/Volumes/Data/other/2026_firefly_synchronization/r_com_range/heatmap_noise_lower_phase_1.0.npz"),
    ("removed_links_0.0", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.0.npz"),
    ("removed_links_0.1", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.1.npz"),
    ("removed_links_0.2", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.2.npz"),
    ("removed_links_0.3", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.3.npz"),
    ("removed_links_0.4", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.4.npz"),
    ("removed_links_0.5", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=5000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.5.npz"),
    ("removed_links_0.6", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.6.npz"),
    ("removed_links_0.7", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.7.npz"),
    ("removed_links_0.8", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.8.npz"),
    ("removed_links_0.9", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=0.9.npz"),
    # ("removed_links_1.0", "/Volumes/Data/other/2026_firefly_synchronization/compressed_async_lower_phase_runs_results_k_graph_experiment_local_N=50_200_C=10_70_T=10000_flash_proportion=0.5_qr_threshold=0.5_update_noise=1.0.npz"),
]
x_tick_labels = [50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
y_tick_labels = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]
ticks = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# --- Shared normalization ---
norm = LogNorm(vmin=1e-3, vmax=1)
for name, path in files:
    # Load data
    data = np.load(path)["arr"] / 1000
    # data is of shape 16 x 16
    # can you take only every second value of the x axis
    data_reduced = data[:, ::2]
    data_reduced = data_reduced[::2, :]
    ticks_reduced = ticks[:8]
    x_tick_labels_reduced = x_tick_labels[::2]
    y_tick_labels_reduced = y_tick_labels[::2]
    # print(data_reduced.shape)
    #
    # exit(12)
    # Avoid zeros for LogNorm
    data_reduced = np.where(data_reduced <= 0, 1e-3, data_reduced)
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(data_reduced, cmap="plasma", norm=norm)
    # ax.set_axis_off()
    ax.invert_yaxis()
    # set ticks
    ax.set_xticks(ticks_reduced)
    ax.set_yticks(ticks_reduced)
    ax.set_xticklabels(x_tick_labels_reduced)
    ax.set_yticklabels(y_tick_labels_reduced)
    # Colorbar
    # cbar = fig.colorbar(im, ax=ax)
    # cbar.set_label("Asynchronous runs")
    # Save
    save_path = os.path.join(out_dir, f"{name}.pdf")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")