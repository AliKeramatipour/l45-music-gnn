import matplotlib.pyplot as plt
import numpy as np

def piano_roll(array,plot_range = (0,3000)):
    array = array[plot_range[0]:plot_range[1]]
    plt.plot(range(array.shape[0]), np.multiply(np.where(array>0, 1, 0), range(1, 129)), marker='.', markersize=1, linestyle='')
    plt.title("Piano roll")
    plt.xlabel("Timessteps")
    plt.ylabel("Notes")
    plt.show()