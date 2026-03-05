import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
import random
import igraph as ig

matplotlib.use('QtAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


C=10
N = 100
k_range = [0,1,2,3,4,5,6,7,8,9,10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

flash_counts = pd.read_csv(f'/Volumes/Data/other/2026_firefly_synchronization/N=100_clock_lnegth={C}_T=1000_flash_proportion=0.5_k_regular_graph_flash_counts.csv')
with open(f'/Volumes/Data/other/2026_firefly_synchronization/N=100_clock_lnegth={C}_T=1000_flash_proportion=0.5_k_regular_graph_init_state_failed.pkl', 'rb') as f:
    init_states_failed = pickle.load(f)
with open(
    f'/Volumes/Data/other/2026_firefly_synchronization/N=100_clock_lnegth={C}_T=1000_flash_proportion=0.5_k_regular_graph_init_state_sucess.pkl',
    'rb') as f:
    init_states_success = pickle.load(f)
with open(
    f'/Volumes/Data/other/2026_firefly_synchronization/N=100_clock_lnegth={C}_T=1000_flash_proportion=0.5_k_regular_graph_phase_history.pkl',
    'rb') as f:
    phase_history = pickle.load(f)
flash_counts.columns = flash_counts.columns.astype(int)

seeds_where_it_does_not_work = flash_counts[100][flash_counts[100] < 100].index
works = {}
for k in flash_counts.columns:
    # print(f"k={k}")
    works[k] = flash_counts[k][seeds_where_it_does_not_work]

works = pd.DataFrame(works)

y_min = int(np.nanmin(works.values))
y_max = int(np.nanmax(works.values))

# Create empty count matrix
heatmap = pd.DataFrame(
    0,
    index=np.arange(y_min, y_max + 1),
    columns=works.columns
)

# Fill counts
for k in works.columns:
    counts = works[k].value_counts()
    for y_val, count in counts.items():
        heatmap.loc[int(y_val), k] += count

# ---- Plot ----
plt.figure(figsize=(8,6))

im = plt.imshow(
    heatmap.values,
    aspect='auto',
    origin='lower'   # important so small y is at bottom
)

plt.colorbar(im)

plt.xticks(
    ticks=np.arange(len(heatmap.columns)),
    labels=heatmap.columns,
    rotation=45
)

plt.yticks(
    ticks=np.arange(len(heatmap.index)),
    labels=heatmap.index
)

plt.xlabel("k")
plt.ylabel("y value")
plt.title("Only failed settings for k = 100")

plt.tight_layout()
# plt.show()

color_map = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "orange",
    5: "purple",
    6: "brown",
    7: "pink",
    8: "gray",
    9: "cyan"
}


for k in k_range:
    print(f"{k}--------------------------------------------------------------------------------")
    # print(works[k].index[0])
    init_state = init_states_failed[k][works[k].index[0]]
    if np.sum(init_state) == 0:
        init_state = init_states_success[k][works[k].index[0]]
    # print(init_state)
    # Convert values to integers
    values_int = init_state.astype(int)
    
    out_come = phase_history[k][works[k].index[0]]
    
    # Map each value to a color
    colors = [color_map[v] for v in values_int]
    
    for seed_graph in range(1):
        if k == N:
            communication_graph = np.ones((N, N))
            np.fill_diagonal(communication_graph, 0)
        else:
            random.seed(seed_graph)
            np.random.seed(seed_graph)
            ig.set_random_number_generator(random)
            G = ig.Graph.K_Regular(N, k)
            communication_graph = np.array(G.get_adjacency().data)
            hist = G.path_length_hist()
            print(hist)
            print(f"modularity score: {G.modularity(membership=out_come)}")
            # print(f"maximal clique: {len(G.maximal_cliques())}{G.maximal_cliques()}")
            # leiden = G.community_optimal_modularity() # NOTE This takes forever!
            leiden = G.community_multilevel()
            print(leiden)
        
            fig, ax = plt.subplots()
            plt.title(f"K={k}, seed={seed_graph}")
            ig.plot(G,
                    vertex_color=colors,
                    layout="mds",  # fruchterman_reingold
                    target=ax)
            plt.show()