import numpy as np
import igraph as ig
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# matplotlib.use('QtAgg')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# N = 100
# clock_length = 34
# T=1000
# qr_threshold = 0.33
# flash_proportion = 0.6
# noise_level = 0.0  # Set None to deactivate:  0.05

# parameter = start value - step size - end value (inclusive)
# N = 50 - 10 - 200
# k = 0 - 5 - N
# seed = 0 - 1 - 99
for N in np.arange(50, 201, 10):
    for k in np.arange(0, N+1, 5):
        for seed in np.arange(0, 1, 1):
            print(f"setting: N={N}, k={k}, seed={seed}")
            path = (f'/Volumes/Data/other/2026_firefly_synchronization/pre_computed_graphs/'
                    f'N={N}_k={k}_seed={seed}.npz')
            # check if file exist:
            if os.path.exists(path):
                print(f"setting N={N}, k={k}, seed={seed} already exists...")
                continue
            if N == k:
                communication_graph = np.ones((N, N))
                np.fill_diagonal(communication_graph, 0)
            if k > (N / 2):  # we can use the complement to be quicker
                G = ig.Graph.K_Regular(N, N - 1 - k)
                communication_graph = np.ones((N, N))
                np.fill_diagonal(communication_graph, 0)
                communication_graph -= np.array(G.get_adjacency().data)
            else:
                G = ig.Graph.K_Regular(N, k)
                communication_graph = np.array(G.get_adjacency().data)
            
            np.savez_compressed(path, communication_graph=communication_graph)
            print(f"done setting N={N}, k={k}, seed={seed}")

# path = f'/Volumes/Data/other/2026_firefly_synchronization/N={N}_C={clock_length}_T={T}_flash_proportion={flash_proportion}_qr_threshold={flash_proportion}_update_noise={noise_level}_r_com_range_flash_counts.pkl'
#
# data = pd.read_pickle(path)
# heatmap = np.zeros((N + 1, len(data.keys())))
#
# for i, k in enumerate(data.keys()):
#     for run in data[k].keys():
#         heatmap[int(np.max(data[k][run])), i] += 1
#         plt.plot(data[k][run])
#         plt.show()