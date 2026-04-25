import numpy as np


def is_k_regular(A, k, undirected=True, no_self_loops=True):
    if undirected and not np.allclose(A, A.T):
        return False
    
    if no_self_loops and not np.all(np.diag(A) == 0):
        return False
    
    degrees = A.sum(axis=1)
    
    return np.all(degrees == k)


save_dir = "/home/till/PycharmProjects/firefly_two-phase_performance/results"

for N in [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160.170, 180, 190, 200]:
    for k_per in [0.9]:
        k = int(N - N * k_per)
        for seed_graph in range(1000):
            data = np.load(f"{save_dir}/pre_computed_graphs/N={N}_k={k}_seed={seed_graph}.npz")
            communication_graph = data["communication_graph"]
            if is_k_regular(communication_graph, k):
                pass
            else:
                print(f"N={N}, k={k}, seed={seed_graph}")
