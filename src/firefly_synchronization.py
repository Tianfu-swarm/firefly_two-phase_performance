import numpy as np
import matplotlib.pyplot as plt
import gif
import firefly_base as fb

'''
Basic plotting of synchronization (Task a)
'''
RADII = [0.05,0.1,0.5,1.4]

#avg_neighbors = {k: [] for k in RADII}
#for i in range(10):


locations, d_mat, initial_clocks = fb.init_sim(fb.N, fb.CYCLE_LENGTH) #same initialization for runs with different radius values

for r in RADII:
    clocks = initial_clocks.copy()
    neighbors = fb.calc_neighbors(d_mat, r)
    #avg_neighbors[r].append(sum(map(len, neighbors)) / len(neighbors))

    print("Avg number of neigbors: " + str(sum(map(len, neighbors)) / len(neighbors)))

    num_flashing = []
    history = np.empty((fb.TIME_STEPS, fb.N), np.uint8)
    for t in range(fb.TIME_STEPS):
        num_flashing.append(np.sum(clocks <= fb.CYCLE_LENGTH/2))
        history[t] = clocks
        fb.sync_clock(clocks, neighbors)
        fb.step_time(clocks)

#avg_res = {k: sum(v)/len(v) for k, v in avg_neighbors.items()}
#print(avg_res)


    plt.figure(figsize=(14,7))
    plt.plot(list(range(fb.TIME_STEPS)), num_flashing, linewidth=0.5)
    plt.title('Currently flashing fireflies over time for r=' + str(r))
    plt.xlabel('Timestep')
    plt.ylabel('Flashing firelies')
    plt.ylim(0,155)
    plt.yticks([0,20,40,60,75,90,110,130,150])
    plt.savefig('../output/currently_flashing_r' + str(r) + '.png')
    plt.show()