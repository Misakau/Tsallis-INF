import argparse
import random
import cvxpy as cp
import numpy as np
from MAB import MAB
from utils import set_seed,save_random_state,load_random_state


parser = argparse.ArgumentParser(
        description="1/2-Tsallis-Inf"
    )
parser.add_argument("--T", type=int, default=10)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--K", type=int, default=8)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--recover", action="store_true", default=False)
parser.add_argument("--recover_path", type=str, default='myinfo.pkl')

args = parser.parse_args()

def choose_w(L_0, k, t, alpha):
    w = cp.Variable(k, pos=True)
    a = np.ones(k) / alpha * np.sqrt(t) # lr = 1/sqrt(t), alpha = 0.5
    obj = cp.Maximize(w @ L_0.T + (w**alpha) @ a.T)
    cons = [w @ np.ones(k).T == 1]
    prob = cp.Problem(obj,cons)
    prob.solve(solver='SCS')
    try: 
        return w.value.copy()
    except:
        print(L_0)
        print(prob.status)
        print(prob.value)
    return w.value.copy()

EPOCH = args.epoch
T = args.T
K = args.K
ALPHA = 0.5
Ds = np.array([0.4,0.5,0.2,0.7,0.1,0.25,0.3,0.1])
RECOVER = args.recover
REC_PATH = args.recover_path

Rlist = {}
Last_L_0 = np.zeros(K)
Last_R_0 = 0
# set seed
set_seed(args.seed)

mab = MAB(K,Ds)

# recover if needed

if RECOVER == True:
    info = load_random_state(REC_PATH)
    np.random.set_state(info['np_state'])
    random.setstate(info['py_state'])
    mab.load(info['mab.K'],info['mab.dists'],
             info['mab.epoch'],info['mab.timestep']
            )
    Rlist = info['rlist'].copy()
    Last_L_0 = info['l0']
    Last_R_0 = info['r0']

# 1/2-Tsallis-Inf
st_epoch = mab.get_epoch()
for epoch in range(st_epoch, EPOCH):
    # Initialize L_0
    L_0 = np.zeros(K)
    R_0 = 0
    # Recover L_0 and R_0 if needed
    if epoch == st_epoch and RECOVER == True:
        L_0 = Last_L_0.copy()
        R_0 = Last_R_0
    st_t = mab.get_timestep()

    if st_t == T:
        mab.reset()
        continue

    for t in range(st_t, T): # the real time slot is t+1
        w = choose_w(-L_0,K,t+1,ALPHA)
        I_t = np.random.multinomial(1,w,1)
        idx = I_t.argmax()
        r = mab.pick(idx)
        loss = 1 - r
        R_0 += r
        l_t = I_t / w * loss
        L_0 += l_t.reshape(-1)
        if epoch == 0:
            Rlist[t] = [R_0]
        else:
            Rlist[t].append(R_0)
        # Save last state
        save_random_state(REC_PATH,mab,Rlist,L_0,R_0)

    mab.reset()

Rlist = np.array([Rlist[k] for k in range(T)])

sample_mean = Rlist.mean(axis=1)
Mean = np.array([mab.regt(t) for t in range(1,T+1)])
regret = Mean - sample_mean

import matplotlib.pyplot as plt

plt.figure()
plt.title(f'1/2-Tsallis-Inf with stochastic MAB (K={args.K} T={args.T})\nBernoulli={Ds}')
plt.xlabel('Time Slots')
plt.ylabel('Expected Regret')
plt.plot(np.arange(1,T+1),regret)
plt.savefig("Evaluation.png")
#plt.legend()