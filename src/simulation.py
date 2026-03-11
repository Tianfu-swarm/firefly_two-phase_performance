import numpy as np


def simulate_fireflies_k_regular_graph(N=150,
                                       clock_length=10,
                                       phases=np.zeros((150, 1)),
                                       communication_graph=np.zeros((150, 150)),
                                       T=1000,
                                       flash_proportion=0.5,
                                       k=-1,
                                       seed=42,
                                       update_noise=None):
    """
    Simulation for the k regular graph
    :param N: Number of fireflies
    :param clock_length: The internal clock length for each firefly
    :param phases: N x 1 matrix with the clock counter for each firefly
    :param communication_graph: N x N matrix with 1 for communication link and 0 no communication link
    :param T: The number of time steps the simulation runs
    :param flash_proportion: The length of flashing (usually 0.5)
    :param k: Just a path through for logging
    :param seed: Just a path through for logging
    :param update_noise: default None, there is no noise when updating the phase, otherwise specify a float value
    that is used for making the noise level
    :return:
    """
    rng = np.random.default_rng(seed)
    flash_length = int(clock_length * flash_proportion)
    flash_start = int(clock_length - flash_length)
    trigger_phase = flash_start + 1  # NOTE this is a constant
    
    init_clock_state = phases.copy()  # NOTE this is only forwarded for logging purposes
    
    phase_history = np.zeros((T, N), dtype=int)
    groups_history = np.zeros((T, 1), dtype=int)
    
    flashing = (phases >= flash_start)
    
    flash_counts = np.zeros(T, dtype=int)
    
    for t in range(T):
        # logging
        phase_history[t] = phases
        values, counts = np.unique(phase_history[t], return_counts=True)
        groups_history[t] = len(counts)
        flash_counts[t] = flashing.sum()
        
        phases = (phases + 1) % clock_length
        flashing = (phases >= flash_start)
        idxs = np.where((phases) == trigger_phase)[0]
        
        neighbor_flash_count = communication_graph[idxs] @ flashing  # (len(idxs), 1)
        neighbor_total = communication_graph[idxs].sum(axis=1)  # total neighbors
        majority_flashing = (neighbor_flash_count > neighbor_total / 2)  # (len(idxs), 1)
        
        # update phase -- we do not need to modulo bc it only affects fireflies in the middle of their clock cycle
        if update_noise is None:
            phases[idxs] += majority_flashing.flatten().astype(int)
        else:
            # noisy update: with prob update_noise do the opposite of the normal update, with prob 1-update_noise
            #  do the normal update
            noise = rng.random(len(idxs)) <= update_noise
            phases[idxs[noise != majority_flashing]] += 1
    
    return flash_counts, phase_history[-1, :], groups_history, k, init_clock_state, seed


def simulate_fireflies_k_regular_graph_transition(N=150,
                                                  clock_length=10,
                                                  phases=np.zeros((150, 1)),
                                                  communication_graph_1=np.zeros((150, 150)),
                                                  communication_graph_2=np.zeros((150, 150)),
                                                  t_switch=500,
                                                  T=1000,
                                                  flash_proportion=0.5,
                                                  k=-1,
                                                  seed=42,
                                                  update_noise=None):
    """
    Simulation for the k regular graph
    :param N: Number of fireflies
    :param clock_length: The internal clock length for each firefly
    :param phases: N x 1 matrix with the clock counter for each firefly
    :param communication_graph_1: N x N matrix with 1 for communication link and 0 no communication link
    :param communication_graph_2: N x N matrix with 1 for communication link and 0 no communication link
    :param t_switch: The time step at which the communication graph switches from communication_graph_1 to communication_graph_2
    :param T: The number of time steps the simulation runs
    :param flash_proportion: The length of flashing (usually 0.5)
    :param k: Just a path through for logging
    :param seed: Just a path through for logging
    :param update_noise: default None, there is no noise when updating the phase, otherwise specify a float value
    that is used for making the noise level
    :return:
    """
    rng = np.random.default_rng(seed)
    flash_length = int(clock_length * flash_proportion)
    flash_start = int(clock_length - flash_length)
    trigger_phase = flash_start + 1  # NOTE this is a constant
    
    init_clock_state = phases.copy()  # NOTE this is only forwarded for logging purposes
    
    phase_history = np.zeros((T, N), dtype=int)
    groups_history = np.zeros((T, 1), dtype=int)
    
    flashing = (phases >= flash_start)
    
    flash_counts = np.zeros(T, dtype=int)
    
    for t in range(T):
        # logging
        phase_history[t] = phases
        values, counts = np.unique(phase_history[t], return_counts=True)
        groups_history[t] = len(counts)
        flash_counts[t] = flashing.sum()
        
        phases = (phases + 1) % clock_length
        flashing = (phases >= flash_start)
        idxs = np.where((phases) == trigger_phase)[0]
        
        if t < t_switch:
            neighbor_flash_count = communication_graph_1[idxs] @ flashing  # (len(idxs), 1)
            neighbor_total = communication_graph_1[idxs].sum(axis=1)  # total neighbors
        else:
            neighbor_flash_count = communication_graph_2[idxs] @ flashing  # (len(idxs), 1)
            neighbor_total = communication_graph_2[idxs].sum(axis=1)  # total neighbors
        majority_flashing = (neighbor_flash_count > neighbor_total / 2)  # (len(idxs), 1)
        
        # update phase -- we do not need to modulo bc it only affects fireflies in the middle of their clock cycle
        if update_noise is None:
            phases[idxs] += majority_flashing.flatten().astype(int)
        else:
            # noisy update: with prob update_noise do the opposite of the normal update, with prob 1-update_noise
            #  do the normal update
            noise = rng.random(len(idxs)) <= update_noise
            phases[idxs[noise != majority_flashing]] += 1
    
    return flash_counts, phase_history[-1, :], groups_history, k, init_clock_state, seed


def simulate_fireflies_r_communication_range(N=150,
                                             clock_length=10,
                                             phases=np.zeros((150, 1)),
                                             communication_graph=np.zeros((150, 150)),
                                             T=1000,
                                             flash_proportion=0.5,
                                             r=-1,
                                             seed=42,
                                             update_noise=None):
    """
    Simulation for the k regular graph
    :param N: Number of fireflies
    :param clock_length: The internal clock length for each firefly
    :param phases: N x 1 matrix with the clock counter for each firefly
    :param communication_graph: N x N matrix with 1 for communication link and 0 no communication link
    :param T: The number of time steps the simulation runs
    :param flash_proportion: The length of flashing (usually 0.5)
    :param r: Just a path through for logging (communication range)
    :param seed: Just a path through for logging
    :param update_noise: default None, there is no noise when updating the phase, otherwise specify a float value
    that is used for making the noise level
    :return:
    """
    rng = np.random.default_rng(seed)
    flash_length = int(clock_length * flash_proportion)
    flash_start = int(clock_length - flash_length)
    trigger_phase = flash_start + 1  # NOTE this is a constant
    
    init_clock_state = phases.copy()  # NOTE this is only forwarded for logging purposes
    
    phase_history = np.zeros((T, N), dtype=int)
    groups_history = np.zeros((T, 1), dtype=int)
    
    flashing = (phases >= flash_start)
    
    flash_counts = np.zeros(T, dtype=int)
    
    for t in range(T):
        # logging
        phase_history[t] = phases
        values, counts = np.unique(phase_history[t], return_counts=True)
        groups_history[t] = len(counts)
        flash_counts[t] = flashing.sum()
        
        phases = (phases + 1) % clock_length
        flashing = (phases >= flash_start)
        idxs = np.where((phases) == trigger_phase)[0]
        
        neighbor_flash_count = communication_graph[idxs] @ flashing  # (len(idxs), 1)
        neighbor_total = communication_graph[idxs].sum(axis=1)  # total neighbors
        majority_flashing = (neighbor_flash_count > neighbor_total / 2)  # (len(idxs), 1)
        
        # update phase -- we do not need to modulo bc it only affects fireflies in the middle of their clock cycle
        if update_noise is None:
            phases[idxs] += majority_flashing.flatten().astype(int)
        else:
            # noisy update: with prob update_noise do the opposite of the normal update, with prob 1-update_noise
            #  do the normal update
            noise = rng.random(len(idxs)) <= update_noise
            phases[idxs[noise != majority_flashing]] += 1
    
    # return flash_counts, phase_history[-1, :], groups_history, r
    return flash_counts, phase_history[-1, :], groups_history, r, init_clock_state, seed
