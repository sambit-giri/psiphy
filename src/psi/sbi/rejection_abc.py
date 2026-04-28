import numpy as np 
from tqdm import tqdm

class rABC:
	def __init__(self, simulator, distance, observation, prior, bounds, N=100, eps=0.1):
		self.simulator = simulator
		self.distance  = distance
		#self.verbose = verbose
		self.y_obs = observation
		self.param_names = [kk for kk in prior]
		self.param_bound = bounds
		self.bounds = np.array([bounds[kk] for kk in self.param_names])
		self.N = N
		self.eps = eps
		self.accepted_param = []

	def sample_prior(self, kk):
		return self.param_bound[kk][0]+(self.param_bound[kk][1]-self.param_bound[kk][0])*np.random.uniform()

	def run(self, N=None):
		if N is not None: self.N = N
		params  = np.array([[self.sample_prior(kk) for kk in self.param_names] for i in range(self.N)]).squeeze()
		sim_out = np.array([self.simulator(i) for i in tqdm(params)])
		dists   = np.array([self.distance(self.y_obs, ss) for ss in tqdm(sim_out)])
		for pp,dd in zip(params,dists):
			if dd<self.eps: self.accepted_param.append(pp)
		#accepted_param = params[dists<self.eps]
		#self.accepted_param = self.accepted_param + accepted_param