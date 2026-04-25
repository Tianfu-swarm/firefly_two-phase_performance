import numpy as np
from collections import deque


def is_k_regular(A, k, undirected=True, no_self_loops=True):
    if undirected and not np.allclose(A, A.T):
        return False
    
    if no_self_loops and not np.all(np.diag(A) == 0):
        return False

    n = A.shape[0]

    # --- k-regular check ---

    degrees = A.sum(axis=1)

    if not np.all(degrees == k):
        return False

    # --- connectivity check (BFS) ---
    visited = np.zeros(n, dtype=bool)
    queue = deque([0])
    visited[0] = True
    while queue:
        node = queue.popleft()
        neighbors = np.where(A[node] != 0)[0]
        for nb in neighbors:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)

    return np.all(visited)

save_dir = "/home/till/PycharmProjects/firefly_two-phase_performance/results"

for N in [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]:
    for k_per in [0.1,.2,.3,.4,.5,.6,.7,.8,0.9]:
        k = int(N - N * k_per)
        for seed_graph in range(1000):
            data = np.load(f"{save_dir}/pre_computed_graphs/N={N}_k={k}_seed={seed_graph}.npz")
            communication_graph = data["communication_graph"]
            if is_k_regular(communication_graph, k, no_self_loops=False):
                pass
            else:
                print(f"N={N}, k={k}, seed={seed_graph}")
