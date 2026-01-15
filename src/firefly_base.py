import numpy as np

'''
Firefly clock cycle from 1 to L = CYCLE_LENGTH = 50
Firefly flashes from 1 to L/2 = 25, is dark from L/2 + 1 = 26 to L = 50
'''
CYCLE_LENGTH = 50
N = 150
TIME_STEPS = 5000

def calc_distance_matrix(locations):
    distance_matrix = np.array([np.linalg.norm(loc - locations[:], axis=1) for loc in locations])
    distance_matrix[np.diag_indices(distance_matrix.shape[0])] = float('inf')
    return distance_matrix 

def calc_neighbors(distance_matrix, r):
    neighbors = []
    for i in range(distance_matrix.shape[0]):
        neighbors.append(np.nonzero(distance_matrix[i] <= r)[0])
    return neighbors

def step_time(clocks):
    clocks += 1
    clocks[np.nonzero(clocks > CYCLE_LENGTH)] = 1

def sync_clock(clocks, neighbors):
    started_flashing = np.nonzero(clocks == 1)[0]
    for i in started_flashing:
        n_i = neighbors[i]
        if np.sum(np.logical_and(clocks[n_i] <= CYCLE_LENGTH/2, clocks[n_i] > 1)) > n_i.shape[0]/2:
        #if np.sum(clocks[n_i] <= CYCLE_LENGTH/2) > n_i.shape[0]/2:
            clocks[i] += 1

def init_sim(n, cycle_length):
    locations = np.random.rand(n,2) #x- and y-coordinates of fireflies, samples from [0,1) (!)
    d_mat = calc_distance_matrix(locations) #pair-wise distance matrix
    initial_clocks = np.random.randint(1, cycle_length+1,n) #150 clock values between 1 and 50 (CYCLE_LENGTH) 
    return locations, d_mat, initial_clocks