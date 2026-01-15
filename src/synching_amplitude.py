import numpy as np
import matplotlib.pyplot as plt
import firefly_base as fb

'''
Investigating effect of changing vicinity radius r (Task b)
'''
RADII = np.arange(0.025,1.425,0.025)
SAMPLES = 50

amps = np.empty((len(RADII), SAMPLES))
for s in range(SAMPLES):
    #same initialization for runs with different radius values
    locations, d_mat, initial_clocks = fb.init_sim(fb.N, fb.CYCLE_LENGTH)

    for idx,r in enumerate(RADII):
        clocks = initial_clocks.copy()
        neighbors = fb.calc_neighbors(d_mat, r)
        #avg_neighbors[r].append(sum(map(len, neighbors)) / len(neighbors))

        num_flashing = []
        for t in range(fb.TIME_STEPS):
            if t >= 4950:
                num_flashing.append(np.sum(clocks <= fb.CYCLE_LENGTH/2))
            fb.sync_clock(clocks, neighbors)
            fb.step_time(clocks)

        amp = (max(num_flashing) - min(num_flashing)) / 2
        amps[idx, s] = amp


avgs = np.average(amps, axis=1)

plt.plot(RADII, avgs)
plt.title('Average amplitude during last flash cycle')
plt.xlabel('Vicinity (radius r)')
plt.ylabel('Amplitude')
plt.ylim(bottom=0)
plt.savefig('./exercise2/plots/amplitudes.png')


