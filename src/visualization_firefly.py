import numpy as np
import matplotlib.pyplot as plt
import gif
import firefly_base as fb

'''
Creating a gif to visualize the firefly synchronization behavior (Bonus)
'''
R = 0.5
STEPS = fb.TIME_STEPS
TARGET_DURATION = 60 #in seconds


#SIM
locations, d_mat, clocks = fb.init_sim(fb.N, fb.CYCLE_LENGTH) 
neighbors = fb.calc_neighbors(d_mat, R)

history = np.empty((STEPS, fb.N), np.uint8)
for t in range(STEPS):
    history[t] = clocks
    fb.sync_clock(clocks, neighbors)
    fb.step_time(clocks)



#GIF PLOTTING
@gif.frame
def plot(i):
    flashing = np.where(history[i] <= fb.CYCLE_LENGTH/2)
    dark = np.where(history[i] > fb.CYCLE_LENGTH/2)
    plt.scatter(locations[flashing][:,0], locations[flashing][:,1], color="gold", s=10)
    plt.scatter(locations[dark][:,0], locations[dark][:,1],color='gray',s=10)
    plt.title(str(i))
    plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,1.1))

frames = [plot(i) for i in range(STEPS)]

gif.save(frames, './exercise2/gifs/synching_fireflies_d' + str(TARGET_DURATION) + '_r' + str(R) + '.gif', duration=TARGET_DURATION*1000/STEPS)
