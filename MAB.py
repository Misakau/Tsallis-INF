import random
import numpy as np

from utils import set_seed,save_random_state,load_random_state

class MAB:
   def __init__(self,K, mu):
      self.K = K
      self.dists = mu.reshape(-1)
      self.epoch = 0
      self.timestep = 0
      assert mu.shape[0] == K
   
   def best_idx(self):
      return np.argmax(self.dists)

   def regt(self,t):
      return t*self.dists[self.best_idx()]

   def pick(self, idx):
      assert idx < self.K and idx >= 0
      p = self.dists[idx]
      #ret = np.zeros(self.K)
      r = random.uniform(0,1)
      ret = 1 if r < p else 0
      self.timestep += 1
      return ret
   
   def init(self):
      self.epoch = 0
      self.timestep = 0

   def reset(self):
      self.epoch += 1
      self.timestep = 0

   def get_dist(self):
      return self.dists
   
   def get_K(self):
      return self.K
   
   def get_timestep(self):
      return self.timestep
   
   def get_epoch(self):
      return self.epoch
   
   def load(self,K,dists,epoch,timestep):
      self.K = K
      self.dists = dists
      self.epoch = epoch
      self.timestep = timestep
   
if __name__ == '__main__':
   set_seed(1)
   mab = MAB(K=5,mu = np.array([0.4,0.5,0.2,0.7,0.1]))
   mab.reset()
   print(mab.get_dist())
   print(mab.regt(9))
   print(mab.get_K())
   s = 0
   for i in range(50000):
      s += mab.pick(3)
   print(s/50000)
   save_random_state("myDictionary.pkl",mab)
   info = load_random_state("myDictionary.pkl")