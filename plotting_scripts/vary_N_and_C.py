import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

T = 10000
flash_proportion = 0.5
noise_level = 1.0  # Set None to deactivate:  0.05
noise_str = f"_update_noise={noise_level}"  # set noise level
Ns = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
Cs = [10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70]  # , 54, 58, 62, 66, 70
use = "r"  # "k" or "r"
# load data
save_flash_counts = {}
for N in Ns:
    save_flash_counts[N] = {}
    for C in Cs:
        save_flash_counts[N][C] = -1
        if use == "k":
            path = f'/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={C}_T={T}_flash_proportion={flash_proportion}{noise_str}_k_regular_graph_flash_counts.csv'
        else:
            path = f'/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={C}_T={T}_flash_proportion={flash_proportion}{noise_str}_r_com_range_flash_counts.csv'
        
        try:
            data = pd.read_csv(path)
            print(f"Loaded data for N={N}, C={C}")
            # print(data.head())
            if use == "k":
                full_communication_data = data[str(N)]
            else:
                full_communication_data = data["1.5"]
            print(full_communication_data.shape)
            # get sum of all flash counts that are smaller than 50
            save_flash_counts[N][C] = (full_communication_data.values < (N * 0.9)).sum() / \
                                      full_communication_data.shape[0]
        except:
            print(f"File not found: {path}")
            continue

# Plot
# build matrix
heatmap = np.zeros((len(Cs), len(Ns)))

for i, C in enumerate(Cs):
    for j, N in enumerate(Ns):
        heatmap[i, j] = save_flash_counts.get(N, {}).get(C, np.nan)
        if heatmap[i, j] == 0.0:
            heatmap[i, j] = 0.0001

# plot
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(
    heatmap,
    origin="lower",
    aspect="auto",
    cmap="plasma",
    # vmin=0,
    # vmax=1,
    norm=LogNorm(vmin=1e-4, vmax=1)  # avoid log(0)
)

ax.set_xticks(np.arange(len(Ns)))
ax.set_xticklabels(Ns)

ax.set_yticks(np.arange(len(Cs)))
ax.set_yticklabels(Cs)

ax.set_xlabel("N")
ax.set_ylabel("C")

fig.colorbar(im, ax=ax, label="Asynchronous runs / Total runs")

plt.tight_layout()
plt.show()
