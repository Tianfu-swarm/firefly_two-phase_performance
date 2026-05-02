import numpy as np
import igraph as ig
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('QtAgg')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

seeds = list(range(1000))
Nk_pairs = list(zip([50, 60, 70, 80, 90, 100], [5, 6, 7, 8, 9, 10]))
n_rows = len(seeds)
n_cols = len(Nk_pairs)

# fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
for j, (N, k) in enumerate(Nk_pairs):
    counter = 0
    commu = []
    for i, seed in enumerate(seeds):
        # print(f"setting: N={N}, k={k}, seed={seed}")
        path = (f'/Volumes/Data/other/2026_firefly_synchronization/pre_computed_graphs/'
                f'N={N}_k={k}_seed={seed}.npz')
        graph = np.load(path)['communication_graph']

        g = ig.Graph.Adjacency((graph > 0).tolist(), mode=ig.ADJ_UNDIRECTED)
        
        communities = g.community_multilevel()
        commu.append(communities.modularity)
        
        if g.is_bipartite():
            counter += 1
    print(f"setting: N={N}, k={k} | \n"
          f"    #bipartite: {counter}\n"
          f"    community: {np.mean(commu)} ({np.std(commu)})")
#         layout = g.layout("fr")
#         ax = axs[i, j]  # row = seed, col = (N,k)
#         ig.plot(
#             g,
#             target=ax,
#             layout=layout,
#             vertex_size=10,  # smaller for grid
#             vertex_color="lightblue",
#             edge_width=0.5
#         )
#
#         # Titles only on top row (cleaner)
#         if i == 0:
#             ax.set_title(f"N={N}, k={k}", fontsize=10)
#         # Optional: label seeds on left
#         if j == 0:
#             ax.set_ylabel(f"seed={seed}", fontsize=10)
#         ax.set_xticks([])
#         ax.set_yticks([])
# plt.tight_layout()
# plt.show()



# from simulation import *
#
# save_dir = "/Volumes/Data/other/2026_firefly_synchronization"
# seed = 116
# N = 50  # or 60
# k = 5
# C = 34
#
# data = np.load(f"{save_dir}/pre_computed_graphs/N={N}_k={k}_seed={seed}.npz")
# communication_graph = data["communication_graph"]
# rng = np.random.default_rng(seed)
# phases = rng.integers(0, C, size=N)
# flash_counts, phase_history, groups_history, k, init_clock_state, seed = simulate_fireflies_k_regular_graph(N=N,
#                                                                                                             clock_length=C,
#                                                                                                             phases=phases,
#                                                                                                             communication_graph=communication_graph,
#                                                                                                             T=10000,
#                                                                                                             flash_proportion=0.5,
#                                                                                                             qr_threshold=0.5,
#                                                                                                             k=-1,
#                                                                                                             seed=seed,
#                                                                                                             update_noise=0.0)
#
# print(phase_history.shape)
# # indices that would sort agents by their final value
# sorted_idx = np.argsort(phase_history[-1])
# # reorder columns (agents)
# phase_history = phase_history[:, sorted_idx]
#
# # threshold at C/2
# phase_history = (phase_history >= (C / 2)).astype(int)
#
# plt.figure(figsize=(12, 4))
# plt.imshow(
#     phase_history.T,          # transpose → (50, 10000)
#     aspect='auto',   # important for long time axis
#     origin='lower'   # agent 0 at bottom
# )
#
# plt.colorbar(label="Value")
# plt.xlabel("Time step")
# plt.ylabel("Agent")
# plt.title("Agent Activity Over Time")
# plt.tight_layout()
# plt.show()