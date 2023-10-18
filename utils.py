import random
import numpy as np
import pickle

def set_seed(seed):
   np.random.seed(seed=seed)
   random.seed(seed)

def save_random_state(path,mab,rlist,l0,r0):
    np_state = np.random.get_state()
    py_state = random.getstate()
   
    info = {'np_state': np_state,
           'py_state': py_state,
           'mab.dists': mab.get_dist(),
           'mab.K': mab.get_K(),
           'mab.epoch': mab.get_epoch(),
           'mab.timestep': mab.get_timestep(),
           'rlist': rlist,
           'l0': l0,
           'r0': r0
           }
    with open(path, "wb") as tf:
        pickle.dump(info,tf)

def load_random_state(path):
    with open(path, "rb") as tf:
        info = pickle.load(tf)
    return info