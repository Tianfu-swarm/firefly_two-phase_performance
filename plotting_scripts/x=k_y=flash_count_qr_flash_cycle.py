import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

N = 100
clock_length = 10
T = 1000
noise_level = 0.0
connectivity = "k"  # "r" or "k"

if connectivity == "r":
    x_label = "r"  # Connectivity
    y_label = "F/N"  # max amplitude
    x_ticks = [0, 0.3, 0.6, 0.9, 1.2, 1.5]
    x_tick_pos = [0, 3, 6, 9, 12, 15]
    y_ticks = [0.0, 0.5, 1]
    y_tick_pos = [0, 50, 100]
elif connectivity == "k":
    x_label = "k/N"  # Connectivity
    y_label = "F/N"  # max amplitude
    x_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    x_tick_pos = [0, 4, 8, 12, 16, 20]
    y_ticks = [0.0, 0.5, 1]
    y_tick_pos = [0, 50, 100]
else:
    raise ValueError("connectivity not implemented")

def get_heatmap(data_path, N):
    data = pd.read_pickle(data_path)
    heatmap = np.zeros((N + 5, len(data.keys())))  # pad some to see the full synchronization
    
    for i, k in enumerate(data.keys()):
        for run in data[k].keys():
            heatmap[int(np.max(data[k][run])), i] += 1
    
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i, j] == 0.0:
                heatmap[i, j] = 0.0001
            else:
                heatmap[i, j] = np.log(heatmap[i, j])
                pass
    
    return heatmap, data

# Convert pt -> inches for matplotlib
pt_to_inch = 1 / 72.27
width_pt = 285
height_pt = 149
fig_width = width_pt * pt_to_inch*3
fig_height = height_pt * pt_to_inch*3
fig, axs = plt.subplots(nrows=6, ncols=6, sharex=False, sharey=False, figsize=(fig_width, fig_height))
fig.subplots_adjust(wspace=0.05, hspace=0.25)

vmin, vmax = None, None
heatmaps = {}

# First pass: collect all heatmaps and find global min/max
for i, flash_proportion in enumerate([0.1, 0.2, 0.33, 0.4, 0.5, 0.6]):
    for k, qr_threshold in enumerate([0.1, 0.2, 0.33, 0.4, 0.5, 0.6]):
        # try:
        if connectivity == "r":
            data_path = f'/Volumes/Data/other/2026_firefly_synchronization/qr_f_experiments_r_com_range/flash_proportion={flash_proportion}_qr_threshold={qr_threshold}_update_noise=0.0/N={N}_C={clock_length}_T={T}_r_com_range_flash_counts.pkl'
        if connectivity == "k":
            data_path = f'/Volumes/Data/other/2026_firefly_synchronization/qr_f_experiments_k_graph/flash_proportion={flash_proportion}_qr_threshold={qr_threshold}_update_noise=0.0/N={N}_C={clock_length}_T={T}_k_regular_graph_flash_counts.pkl'
                
        heatmap, data = get_heatmap(data_path, N)
        heatmaps[(i, k)] = (heatmap, data)
        local_min, local_max = heatmap.min(), heatmap.max()
        vmin = local_min if vmin is None else min(vmin, local_min)
        vmax = local_max if vmax is None else max(vmax, local_max)
        # except:
        #     print(f"File not found: {data_path}")

# Second pass: plot with shared scale
im = None
for i in range(6):
    for k in range(6):
        ax = axs[i, k]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        if (i, k) in heatmaps:
            heatmap, data = heatmaps[(i, k)]
            im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='plasma', vmin=vmin, vmax=vmax)
            
            # show y-axis only on left column
            if k == 0:
                ax.set_yticks(y_tick_pos)
                ax.set_yticklabels(y_ticks)
                ax.set_ylabel(y_label)
            
            # show x-axis only on bottom row
            if i == 5:
                ax.set_xticks(x_tick_pos)
                ax.set_xticklabels(x_ticks)
                ax.set_xlabel(x_label)

# Single colorbar on the right
# if im is not None:
#     fig_cb, ax_cb = plt.subplots(figsize=(0.5, 6))
#     fig.colorbar(im, cax=ax_cb, orientation='vertical')
#     fig_cb.tight_layout()
plt.show()


# im0 = axs[1].imshow(heatmap_r, aspect='auto', origin='lower', cmap='plasma')
# normalized_labels = [float(col) for col in data_r.keys()]
# axs[1].set_xticks(np.arange(len(data_r.keys())))
# axs[1].set_xticklabels(normalized_labels, rotation=45)
# axs[1].set_ylim(50, N)
# axs[1].set_yticks(np.linspace(50, N, 6))
# axs[1].set_yticklabels(np.linspace(0.5, 1, 6))  # , rotation=45
# axs[1].set_xlabel(f"Connectivity [r] | qr = {qr_threshold} | flash duration = {qr_threshold} | Noise level = {noise_level}")
# axs[1].set_ylabel(r"Maximal amplitude [N$_\text{flash}$ / N]")
# fig.colorbar(im0, ax=axs[1], label="log(Number of runs)", orientation='vertical')
