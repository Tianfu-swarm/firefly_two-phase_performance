import numpy as np
import os
import pandas as pd
import igraph as ig
import random
import pickle
from tqdm import tqdm
import argparse

from simulation import simulate_fireflies_k_regular_graph_transition
from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--save_dir', type=str, default="/Volumes/Data/other/2026_firefly_synchronization")
    arg_parser.add_argument('--n_seeds', type=int, default=1)
    arg_parser.add_argument('--graph_seeds', type=int, default=2)
    # simulation params
    arg_parser.add_argument('--N', type=int, default=100)  # number of fireflies
    arg_parser.add_argument('--C', type=int, default=10)  # clock length (number of discrete phases)
    arg_parser.add_argument('--T', type=int, default=1000)  # number of time steps to simulate
    arg_parser.add_argument('--t_switch', type=int, default=100)  # number of time steps to simulate
    arg_parser.add_argument('--flash_proportion', type=float, default=0.5)  # how long to flash
    arg_parser.add_argument('--update_noise', type=float, default=0.0)  # how long to flash
    arg_parser.add_argument('--reduce_full_k_by', nargs="+", type=int, default=[10])
    
    args = arg_parser.parse_args()
    
    # make a quick check if the results already exist and skip if they do
    flash_counts_path = f"{args.save_dir}/N={args.N}_clock_lnegth={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_update_noise={args.update_noise}_k_regular_graph_transition_flash_counts.pkl"
    if os.path.isfile(flash_counts_path):
        print(f"{flash_counts_path} already exists. skipping...")
        exit(0)
    
    # ensure k values are valid (k must be less than N for k-regular graph)
    args.k_reduced_range = [args.N - k_reduced for k_reduced in args.reduce_full_k_by if k_reduced <= args.N]
    # full communication to start
    communication_graph_1 = np.ones((args.N, args.N))
    
    run_params = []
    save_flash_counts = {}
    save_phase_history = {}
    save_init_state_failed = {}
    save_init_state_success = {}
    for k in args.k_reduced_range:
        save_flash_counts[k] = {}
        save_phase_history[k] = np.zeros((args.graph_seeds, args.N))
        save_init_state_failed[k] = np.zeros((args.graph_seeds, args.N))
        save_init_state_success[k] = np.zeros((args.graph_seeds, args.N))
        for seed_graph in range(args.graph_seeds):
            random.seed(seed_graph)
            np.random.seed(seed_graph)
            ig.set_random_number_generator(random)
            try:
                G = ig.Graph.K_Regular(args.N, k)
            except:
                print(f"Cannot generate k-regular graph with k={k} and N={args.N} (should be k > N). Skipping this k.")
                continue
            print(f"done generating the graph for k={k} and seed={seed_graph}")
            communication_graph_2 = np.array(G.get_adjacency().data)
            
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed)
                group_size_fill = args.N - 3 * (args.N // 4)
                phases = np.concatenate([np.ones(args.N // 4, dtype=int) * i * (args.C // 4) for i in range(3)])
                phases = np.concatenate([phases, np.ones(group_size_fill, dtype=int) * 3 * (args.C // 4)])
                # print(f"phases (shape: {phases.shape}): {phases}")
                # print(f"phase shift: {args.C // 4}")
                #
                run_params.append(
                    (args.N, args.C, phases, communication_graph_1, communication_graph_2, args.t_switch, args.T,
                     args.flash_proportion, k, seed_graph))

    print(f"done setting up the parameters ...")
    print(
        f"Running: N={args.N} | C={args.C} | T={args.T} | flash_proportion={args.flash_proportion} | update_noise={args.update_noise} | k_reduced_range={args.k_reduced_range}")
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        
        futures = [
            executor.submit(simulate_fireflies_k_regular_graph_transition, N, clock_length, phases,
                            communication_graph_1, communication_graph_2, t_switch, T,
                            flash_proportion, k, seed, args.update_noise)
            for
            (N, clock_length, phases, communication_graph_1, communication_graph_2, t_switch, T, flash_proportion, k,
             seed) in run_params
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures)):
            flash_counts, phase_history, groups_history, k, init_clock_state, seed = future.result()
            save_flash_counts[k][seed] = flash_counts
            if np.max(flash_counts) <= args.N * 0.80 and k > args.N * 0.1:
                save_phase_history[k][seed] = phase_history
                save_init_state_failed[k][seed] = init_clock_state
            else:
                save_init_state_success[k][seed] = init_clock_state
    
    with open(
        f"{args.save_dir}/N={args.N}_clock_lnegth={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_update_noise={args.update_noise}_k_regular_graph_transition_flash_counts.pkl",
        'wb') as f:
        pickle.dump(save_flash_counts, f)
    
    with open(
        f"{args.save_dir}/N={args.N}_clock_lnegth={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_update_noise={args.update_noise}_k_regular_graph_transition_phase_history.pkl",
        'wb') as f:
        pickle.dump(save_phase_history, f)
    
    with open(
        f"{args.save_dir}/N={args.N}_clock_lnegth={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_update_noise={args.update_noise}_k_regular_graph_transition_init_state_failed.pkl",
        'wb') as f:
        pickle.dump(save_init_state_failed, f)
    
    with open(
        f"{args.save_dir}/N={args.N}_clock_lnegth={args.C}_T={args.T}_flash_proportion={args.flash_proportion}_update_noise={args.update_noise}_k_regular_graph_transition_init_state_sucess.pkl",
        'wb') as f:
        pickle.dump(save_init_state_success, f)
