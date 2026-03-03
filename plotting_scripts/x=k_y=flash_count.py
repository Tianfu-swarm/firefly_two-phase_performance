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
clock_length = 10
T=1000
flash_proportion = 0.5
noise_level = None  # Set None to deactivate:  0.05
if noise_level is not None:
    noise_str = f"_update_noise={noise_level}"  # set noise level
else:
    noise_str = ""  # set noise level
    noise_level = 0.0
k_path = f'/Volumes/Data/other/2026_firefly_synchronization/N={N}_clock_lnegth={clock_length}_T={T}_flash_proportion={flash_proportion}{noise_str}_k_regular_graph_flash_counts.csv'
r_path = f'/Volumes/Data/other/2026_firefly_synchronization/N={N}_clock_lnegth={clock_length}_T={T}_flash_proportion={flash_proportion}{noise_str}_r_com_range_flash_counts.csv'

def get_heatmap(data_path, N):
    data = pd.read_csv(data_path)
    heatmap = np.zeros((N+1, len(data.columns)))
    
    for col_idx, col in enumerate(data.columns):
        counts = data[col].value_counts()
        for value, count in counts.items():
            heatmap[int(value), col_idx] = np.log(count)
            
    return heatmap, data




heatmap_k, data_k = get_heatmap(k_path, N)
heatmap_r, data_r = get_heatmap(r_path, N)

# Plot
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(8, 6))

im0 = axs[0].imshow(heatmap_k, aspect='auto', origin='lower', cmap='plasma')
normalized_labels = [float(col) / N for col in data_k.columns]
axs[0].set_xticks(np.arange(len(data_k.columns)))
axs[0].set_xticklabels(normalized_labels, rotation=45)
axs[0].set_ylim(50, N)
axs[0].set_yticks(np.linspace(50, N, 6))
axs[0].set_yticklabels(np.linspace(0.5, 1, 6))  # , rotation=45
axs[0].set_xlabel(f"Connectivity [k / N] | Noise level = {noise_level}")
axs[0].set_ylabel(r"Maximal amplitude [N$_\text{flash}$ / N]")
fig.colorbar(im0, ax=axs[0], label="log(Number of runs)", orientation='vertical')

im0 = axs[1].imshow(heatmap_r, aspect='auto', origin='lower', cmap='plasma')
normalized_labels = [float(col) for col in data_r.columns]
axs[1].set_xticks(np.arange(len(data_r.columns)))
axs[1].set_xticklabels(normalized_labels, rotation=45)
axs[1].set_ylim(50, N)
axs[1].set_yticks(np.linspace(50, N, 6))
axs[1].set_yticklabels(np.linspace(0.5, 1, 6))  # , rotation=45
axs[1].set_xlabel("Connectivity [r] | Noise level = {noise_level}")
axs[1].set_ylabel(r"Maximal amplitude [N$_\text{flash}$ / N]")
fig.colorbar(im0, ax=axs[1], label="log(Number of runs)", orientation='vertical')



plt.tight_layout()
plt.show()