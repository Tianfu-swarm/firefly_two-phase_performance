import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

N = 100
clock_length = 34
T=1000
qr_threshold = 0.33
flash_proportion = 0.5
noise_level = 0.0  # Set None to deactivate:  0.05

r_path = f'/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={clock_length}_T={T}_flash_proportion={flash_proportion}_qr_threshold={flash_proportion}_update_noise={noise_level}_r_com_range_flash_counts.pkl'
r_path_2 = f'/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={clock_length}_T={T}_flash_proportion={qr_threshold}_qr_threshold={qr_threshold}_update_noise={noise_level}_r_com_range_flash_counts.pkl'

def get_heatmap(data_path, N):
    data = pd.read_pickle(data_path)
    heatmap = np.zeros((N+1, len(data.keys())))
    
    
    for i, k in enumerate(data.keys()):
        for run in data[k].keys():
            heatmap[int(np.max(data[k][run])), i] += 1
            # plt.plot(data[k][run], alpha=0.5)
            # plt.show()
    for i in range(heatmap.shape[0]):
        for j in range(heatmap.shape[1]):
            if heatmap[i, j] == 0.0:
                heatmap[i, j] = 0.0001
            else:
                heatmap[i, j] = np.log(heatmap[i, j])
    # for col_idx, col in enumerate(data.columns):
    #     counts = data[col].value_counts()
    #     for value, count in counts.items():
    #         heatmap[int(value), col_idx] = np.log(count)
            
    return heatmap, data


# heatmap_k, data_k = get_heatmap(r_path, N)
# heatmap_r, data_r = get_heatmap(r_path_2, N)

# Plot
fig, axs = plt.subplots(nrows=6, ncols=6, sharex=False, sharey=False, figsize=(8, 6))

for i, flash_proportion in enumerate([0.1, 0.2, 0.33, 0.4, 0.5, 0.6]):
    for k, qr_threshold in enumerate([0.1, 0.2, 0.33, 0.4, 0.5, 0.6]):
        try:
            data_path = f'/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={clock_length}_T={T}_flash_proportion={flash_proportion}_qr_threshold={qr_threshold}_update_noise={noise_level}_r_com_range_flash_counts.pkl'
            heatmap, data = get_heatmap(data_path, N)
        except:
            print(f"File not found: {data_path}")
            continue
    
        im0 = axs[i, k].imshow(heatmap, aspect='auto', origin='lower', cmap='plasma')
        normalized_labels = [float(col) for col in data.keys()]
        axs[i, k].set_xticks(np.arange(len(data.keys())))
        axs[i, k].set_xticklabels(normalized_labels, rotation=45)
        axs[i, k].set_ylim(0, N)
        axs[i, k].set_yticks(np.linspace(0, N, 11))
        axs[i, k].set_yticklabels([f"{x:.1f}" for x in np.linspace(0.0, 1, 11)])  # , rotation=45
        axs[i, k].set_xlabel(f"qr = {qr_threshold} | flash duration = {flash_proportion}")
        axs[i, k].set_ylabel(r"Max amp")
        fig.colorbar(im0, ax=axs[i, k], orientation='vertical')

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

plt.tight_layout()
plt.show()