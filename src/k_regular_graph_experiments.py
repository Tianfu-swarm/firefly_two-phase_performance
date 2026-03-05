import numpy as np
import os
import platform
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import random
import pickle
from tqdm import tqdm
import argparse

from simulation import simulate_fireflies_k_regular_graph
from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":
    # for interactive plots
    if platform.system() == "Darwin":
        matplotlib.use('QtAgg')
        plt.rcParams.update({'font.size': 20})
    elif platform.system() == "Linux":
        def is_headless():
            return os.environ.get("DISPLAY", "") == ""
    
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--save_dir', type=str, default="/Volumes/Data/other/2026_firefly_synchronization")
    arg_parser.add_argument('--n_seeds', type=int, default=1000)
    arg_parser.add_argument('--graph_seeds', type=int, default=1)
    # simulation params
    arg_parser.add_argument('--N', type=int, default=100)  # number of fireflies
    arg_parser.add_argument('--C', type=int, default=10)  # clock length (number of discrete phases)
    arg_parser.add_argument('--T', type=int, default=1000)  # number of time steps to simulate
    arg_parser.add_argument('--flash_proportion', type=float, default=0.5)  # how long to flash
    arg_parser.add_argument('--update_noise', type=float, default=0.0)  # how long to flash
    arg_parser.add_argument('--k_range', nargs="+", type=int, default=[0, 10, 20, 30, 100])
    
    args = arg_parser.parse_args()
    
    # ensure k values are valid (k must be less than N for k-regular graph)
    args.k_range = [k for k in args.k_range if k <= args.N]
    
    run_params = []
    save_flash_counts = {}
    save_phase_history = {}
    save_init_state_failed = {}
    save_init_state_success = {}
    avg_num_neighbors = {}
    for k in args.k_range:
        save_flash_counts[k] = np.zeros(args.n_seeds)
        avg_num_neighbors[k] = []
        save_phase_history[k] = np.zeros((args.n_seeds, args.N))
        save_init_state_failed[k] = np.zeros((args.n_seeds, args.N))
        save_init_state_success[k] = np.zeros((args.n_seeds, args.N))
        for seed_graph in range(args.graph_seeds):
            if k == args.N:
                communication_graph = np.ones((args.N, args.N))
                np.fill_diagonal(communication_graph, 0)
            else:
                random.seed(seed_graph)
                np.random.seed(seed_graph)
                ig.set_random_number_generator(random)
                try:
                    G = ig.Graph.K_Regular(args.N, k)
                except:
                    print(f"Cannot generate k-regular graph with k={k} and N={args.N} (should be k > N). Skipping this k.")
                    continue
                communication_graph = np.array(G.get_adjacency().data)

            avg_num_neighbors[k].append(float(np.mean(np.sum(communication_graph, axis=1) / (args.N - 1))))
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed)
                phases = rng.integers(0, args.C, size=args.N)

                run_params.append((args.N, args.C, phases, communication_graph, args.T, args.flash_proportion, k, seed))

    avg_num_neighbors_df = pd.DataFrame(avg_num_neighbors)
    avg_num_neighbors_df.to_csv(
        f"{args.save_dir}/N={args.N}_clock_lnegth={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_k_regular_graph_avg_neighbors.csv",
        index=False)
    print(f"done setting up the parameters ...")
    print(f"Running: N={args.N} | C={args.C} | T={args.T} | flash_proportion={args.flash_proportion} | update_noise={args.update_noise} | k_range={args.k_range}")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        futures = [
            executor.submit(simulate_fireflies_k_regular_graph, N, clock_length, phases, communication_graph, args.T,
                            flash_proportion, k, seed, args.update_noise)
            for (N, clock_length, phases, communication_graph, args.T, flash_proportion, k, seed) in run_params
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            flash_counts, phase_history, groups_history, k, init_clock_state, seed = future.result()
            save_flash_counts[k][seed] = np.max(flash_counts)
            if np.max(flash_counts) <= args.N * 0.80 and k > args.N * 0.1:
                save_phase_history[k][seed] = phase_history
                save_init_state_failed[k][seed] = init_clock_state
            else:
                save_init_state_success[k][seed] = init_clock_state

    save_flash_counts = pd.DataFrame(save_flash_counts)

    noise_str = f"_update_noise={args.update_noise}"

    save_flash_counts.to_csv(
        f"{args.save_dir}/N={args.N}_clock_lnegth={args.C}_T={args.T}_flash_proportion={args.flash_proportion}{noise_str}_k_regular_graph_flash_counts.csv",
        index=False)

    with open(
        f"{args.save_dir}/N={args.N}_clock_lnegth={args.C}_T={args.T}_flash_proportion={args.flash_proportion}{noise_str}_k_regular_graph_phase_history.pkl",
        'wb') as f:
        pickle.dump(save_phase_history, f)

    with open(
        f"{args.save_dir}/N={args.N}_clock_lnegth={args.C}_T={args.T}_flash_proportion={args.flash_proportion}{noise_str}_k_regular_graph_init_state_failed.pkl",
        'wb') as f:
        pickle.dump(save_init_state_failed, f)

    with open(
        f"{args.save_dir}/N={args.N}_clock_lnegth={args.C}_T={args.T}_flash_proportion={args.flash_proportion}{noise_str}_k_regular_graph_init_state_sucess.pkl",
        'wb') as f:
        pickle.dump(save_init_state_success, f)
