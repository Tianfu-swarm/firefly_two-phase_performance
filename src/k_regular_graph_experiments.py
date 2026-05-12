import numpy as np
import os
import pandas as pd
import igraph as ig
import random
import pickle
from tqdm import tqdm
import argparse

from simulation import simulate_fireflies_k_regular_graph
from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--save_dir', type=str, default="/Volumes/Data/other/2026_firefly_synchronization")
    arg_parser.add_argument('--n_seeds', type=int, default=1000)
    arg_parser.add_argument('--graph_seeds', type=int, default=100)
    # simulation params
    arg_parser.add_argument('--N', type=int, default=100)  # number of fireflies
    arg_parser.add_argument('--C', type=int, default=10)  # clock length (number of discrete phases)
    arg_parser.add_argument('--T', type=int, default=1000)  # number of time steps to simulate
    arg_parser.add_argument('--flash_proportion', type=float, default=0.5)  # how long to flash
    arg_parser.add_argument('--qr_threshold', type=float, default=0.5)  # how long to flash
    arg_parser.add_argument('--update_noise', type=float, default=0.0)  # how long to flash
    arg_parser.add_argument('--k_range', nargs="+", type=int, default=[0, 10, 20, 30, 100])
    
    args = arg_parser.parse_args()
    
    # make a quick check if the results already exist and skip if they do
    flash_counts_path = (f"{args.save_dir}/"
                         f"flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}/"
                         f"N={args.N}_C={args.C}_T={args.T}_k_regular_graph_flash_counts.pkl")
    if os.path.isfile(flash_counts_path):
        print(f"{flash_counts_path} already exists. skipping...")
        exit(0)
    
    # ensure k values are valid (k must be less than N for k-regular graph)
    args.k_range = [k for k in args.k_range if k <= args.N]
    
    graph_dir = os.path.dirname(args.save_dir)
    
    run_params = []
    save_flash_counts = {}
    # save_phase_history = {}
    # save_init_state_failed = {}
    # save_init_state_success = {}
    # avg_num_neighbors = {}
    for k in args.k_range:
        save_flash_counts[k] = {}
        # avg_num_neighbors[k] = []
        # save_phase_history[k] = np.zeros((args.n_seeds, args.N))
        # save_init_state_failed[k] = np.zeros((args.n_seeds, args.N))
        # save_init_state_success[k] = np.zeros((args.n_seeds, args.N))
        for seed_graph in range(args.graph_seeds):
            data = np.load(f"{graph_dir}/pre_computed_graphs/N={args.N}_k={k}_seed={seed_graph}.npz")
            communication_graph = data["communication_graph"]
            # avg_num_neighbors[k].append(float(np.mean(np.sum(communication_graph, axis=1) / (args.N - 1))))
            for seed in range(args.n_seeds):
                final_seed = int(args.n_seeds * seed_graph + seed)
                rng = np.random.default_rng(final_seed)
                phases = rng.integers(0, args.C, size=args.N)
                
                run_params.append(
                    (args.N, args.C, phases, communication_graph, args.T, args.flash_proportion, args.qr_threshold, k,
                     final_seed))
    
    print(f"done setting up the parameters ...")
    print(
        f"Running: N={args.N} | C={args.C} | T={args.T} | flash_proportion={args.flash_proportion} | qr_threshold={args.qr_threshold} | update_noise={args.update_noise} | k_range={args.k_range}")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        futures = [
            executor.submit(simulate_fireflies_k_regular_graph, N, clock_length, phases, communication_graph, args.T,
                            flash_proportion, qr_threshold, k, seed, args.update_noise)
            for (N, clock_length, phases, communication_graph, args.T, flash_proportion, qr_threshold, k, seed) in
            run_params
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            flash_counts, phase_history, groups_history, k, init_clock_state, seed = future.result()
            save_flash_counts[k][seed] = flash_counts
            # if np.max(flash_counts) <= args.N * 0.90 and k > args.N * 0.1:
            #     save_phase_history[k][seed] = phase_history
            #     save_init_state_failed[k][seed] = init_clock_state
            # else:
            #     save_init_state_success[k][seed] = init_clock_state
    
    # save_flash_counts = pd.DataFrame(save_flash_counts)
    
    flash_counts_dir_path = os.path.dirname(flash_counts_path)
    os.makedirs(f"{flash_counts_dir_path}", exist_ok=True)
    
    with open(f"{flash_counts_path}", 'wb') as f:
        pickle.dump(save_flash_counts, f)
    # save_flash_counts.to_csv(
    #     f"{args.save_dir}/"
    #     f"flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}/"
    #     f"N={args.N}_C={args.C}_T={args.T}_k_regular_graph_flash_counts.csv",
    #     index=False)
    
    # with open(
    #     f"{args.save_dir}/flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}/N={args.N}_C={args.C}_T={args.T}_k_regular_graph_phase_history.pkl",
    #     'wb') as f:
    #     pickle.dump(save_phase_history, f)
    
    # with open(
    #     f"{args.save_dir}/flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}/N={args.N}_C={args.C}_T={args.T}_k_regular_graph_init_state_failed.pkl",
    #     'wb') as f:
    #     pickle.dump(save_init_state_failed, f)
    
    # with open(
    #     f"{args.save_dir}/flash_proportion={args.flash_proportion}_qr_threshold={args.qr_threshold}_update_noise={args.update_noise}/N={args.N}_C={args.C}_T={args.T}_k_regular_graph_init_state_sucess.pkl",
    #     'wb') as f:
    #     pickle.dump(save_init_state_success, f)
