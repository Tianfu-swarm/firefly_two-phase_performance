import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

N = 100
clock_length = 34
T = 1000
noise_level = 0.0


def get_heatmap(data_path, N):
    data = pd.read_pickle(data_path)
    heatmap = np.zeros((N + 1, len(data.keys())))
    
    for i, k in enumerate(data.keys()):
        for run in data[k].keys():
            heatmap[int(np.max(data[k][run])), i] += 1
    
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i, j] == 0.0:
                heatmap[i, j] = 0.0001
            else:
                heatmap[i, j] = np.log(heatmap[i, j])
    
    return heatmap, data


fig, axs = plt.subplots(nrows=6, ncols=6, sharex=False, sharey=False, figsize=(8, 6))

vmin, vmax = None, None
heatmaps = {}

# First pass: collect all heatmaps and find global min/max
for i, flash_proportion in enumerate([0.1, 0.2, 0.33, 0.4, 0.5, 0.6]):
    for k, qr_threshold in enumerate([0.1, 0.2, 0.33, 0.4, 0.5, 0.6]):
        try:
            data_path = f'/Volumes/Data/other/2026_firefly_synchronization/old_experiments/N={N}_C={clock_length}_T={T}_flash_proportion={flash_proportion}_qr_threshold={qr_threshold}_update_noise={noise_level}_r_com_range_flash_counts.pkl'
            heatmap, data = get_heatmap(data_path, N)
            heatmaps[(i, k)] = (heatmap, data)
            local_min, local_max = heatmap.min(), heatmap.max()
            vmin = local_min if vmin is None else min(vmin, local_min)
            vmax = local_max if vmax is None else max(vmax, local_max)
        except:
            print(f"File not found: {data_path}")

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

# Single colorbar on the right
if im is not None:
    fig_cb, ax_cb = plt.subplots(figsize=(0.5, 6))
    fig.colorbar(im, cax=ax_cb, orientation='vertical')
    fig_cb.tight_layout()
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
