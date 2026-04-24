import numpy as np
import random
import os
import igraph as ig
from concurrent.futures import ProcessPoolExecutor, as_completed

# BASE_PATH = '/home/till/PycharmProjects/firefly_two-phase_performance/results/pre_computed_graphs'
BASE_PATH = '/Volumes/Data/other/2026_firefly_synchronization/pre_computed_graphs'

def generate_graph(args):
    N, k, seed = args
    
    print(f"setting: N={N}, k={k}, seed={seed}")
    
    path = f'{BASE_PATH}/N={N}_k={k}_seed={seed}.npz'
    
    if os.path.exists(path):
        return f"skipped N={N}, k={k}, seed={seed}"
    
    rng = random.Random(int(seed))  # ensure Python int
    ig.set_random_number_generator(rng)
    
    if N == k:
        communication_graph = np.ones((N, N))
        np.fill_diagonal(communication_graph, 0)
    elif k > (N / 2):  # complement trick
        G = ig.Graph.K_Regular(N, N - 1 - k)
        communication_graph = np.ones((N, N))
        np.fill_diagonal(communication_graph, 0)
        communication_graph -= np.array(G.get_adjacency().data)
    else:
        G = ig.Graph.K_Regular(N, k)
        communication_graph = np.array(G.get_adjacency().data)
    
    np.savez_compressed(path, communication_graph=communication_graph)
    
    return f"done N={N}, k={k}, seed={seed}"


if __name__ == "__main__":
    # generate normal graphs
    # tasks = [
    #     (N, k, seed)
    #     for N in np.arange(50, 201, 10)
    #     for k in np.arange(0, N + 1, 5)
    #     for seed in np.arange(0, 1000, 1)
    # ]
    #
    # os.makedirs(BASE_PATH, exist_ok=True)
    #
    # with ProcessPoolExecutor(max_workers=10) as executor:
    #     futures = [executor.submit(generate_graph, t) for t in tasks]
    #
    #     for future in as_completed(futures):
    #         print(future.result())
    
    # generate for the reduced case
    tasks = [
        (N, int(N - missing), seed)
        for N in np.arange(50, 201, 10)
        for missing in [0.9*N]  # 0.05*N, 0.1*N,0.2*N,0.3*N, 0.4*N, 0.6*N, 0.7*N, 0.8*N,
        for seed in np.arange(0, 1000, 1)
    ]
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(generate_graph, t) for t in tasks]

        for future in as_completed(futures):
            print(future.result())
