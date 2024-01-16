import torch
import numpy.linalg as linalg
import numpy as np
from collections import defaultdict

def get_solution(J, ab):
    u_star  = - torch.linalg.inv(J) @ ab
    return u_star



def find_L(Js):
    return float((torch.svd(Js)[1].mean(0)).max().item())

class Algorithm:
    def __init__(self, Js, abs, mu, lr_0=None, init=None, noise=False):
        self.num_samples = Js.shape[0]
        self.lr_0= lr_0
        self.lr = None
        if init == None:
            self.u = 100 * torch.randn_like(abs.mean(0))
            self.h = self.u + torch.randn_like(self.u)
        else:
            self.u, self.h = init
        self.Js = Js
        self.J = Js.mean(0)
        self.mu = mu
        self.L = find_L(self.Js)
        self.abs = abs
        self.transhold = 1
        self.solution = get_solution(self.J, abs.mean(0))
        self.noise = noise

    def update(self):
        raise NotImplementedError

    def run(self, n_steps=4000, sampler="uni", batch_size=1, scheduler="constant", trashold=None):
        if scheduler == "stich":
            if trashold is not None:
                self.transhold = trashold
            else:
                self.transhold = n_steps // 2
            self.lr = [ 2 / (self.a * ( 2 * self.d/self.a + i - self.transhold)) if i >= self.transhold else (1 / self.d) for i in range(n_steps)]
        elif scheduler == "constant":
            self.lr = [ self.lr_0 for i in range(n_steps)]
        if sampler == "uni":
            idxs = [np.random.choice(self.num_samples, batch_size, replace=True) for i in range(n_steps)]
        else:
            idxs=[None for i in range(n_steps)]
        results = defaultdict(list)
        for i in range(n_steps):
            self.update(i, idxs[i])
            results["u"].append(self.u)
            results["Dist2Sol"].append(float(torch.norm(self.solution - self.u)**2))
        return results
    
    def gradient(self, x, idx):
        if self.noise == True:
            return (self.Js[idx, :, :] @ x + self.abs[idx, :] ).mean(0) + 1/(self.abs.shape[1]**0.5) * torch.randn_like(self.abs.mean(0))
        else:
            return (self.Js[idx, :, :] @ x + self.abs[idx, :] ).mean(0)
class SGD(Algorithm):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)
        if self.lr_0 == None:
            self.lr_0 = self.mu / (2*self.L**2)
        self.a = self.mu
        self.d = 2 * (self.L**2) / self.mu

    def update(self, i, idx=None):
        if idx == None:
            idx = range(self.num_samples)
        self.u = self.u - self.lr[i] * self.gradient(self.u, idx) 


class Popov(Algorithm):
    def __init__(self, *args, **kwargs):
        super(Popov, self).__init__(*args, **kwargs)
        if self.lr_0 == None:
            self.lr_0 = 1/ (10 * (3**0.5) * self.L)
        self.a = self.mu
        self.d = 2 * (3**0.5) * self.L
    def update(self, i, idx=None):
        if idx == None:
            idx = range(self.num_samples)
        grad = self.gradient(self.h, idx)
        self.u = self.u - self.lr[i] * grad
        self.h = self.u - self.lr[i] * grad 