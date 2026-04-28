import numpy as np 
import pickle
from tqdm import tqdm
from time import time 

import emcee
from sklearn.gaussian_process import GaussianProcessRegressor

class ABC_gpL:
	def __init__(self, simulator, distance, obs, 
				theta_sampler=None, theta_range={}, n_train_init=100,
				mcmc_sampler=None, mcmc_sampler_info=None
				):

		self.mcmc_sampler_info = mcmc_sampler_info
		if mcmc_sampler is None: mcmc_sampler = emcee.EnsembleSampler
		self.simulator = simulator
		self.distance  = distance
		self.obs = obs

		self.n_train_init = n_train_init
		self.theta_range  = theta_range
		if theta_sampler is None or theta_sampler == 'uniform':
			self.theta_sampler = lambda n=1: np.array([np.random.random(n)*(theta_range[ke][1]-theta_range[ke][0])+theta_range[ke][0] for ke in theta_range.keys()]).T
		else:
			self.theta_sampler = theta_sampler
		if self.mcmc_sampler_info is None:
			self.mcmc_sampler_info = {'ndim': len(self.theta_range.keys())}
			self.mcmc_sampler_info['nwalkers'] = self.mcmc_sampler_info['ndim']*2
			self.mcmc_sampler_info['pool'] = None
			self.mcmc_sampler_info['backend'] = None

		self.theta_train    = np.array([])
		self.dist_train     = np.array([])
		self.distance_model = None

	def prepare_distance_model(self, model=None, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=3):
		if model is not None:
			self.distance_model = model
		else:
			self.distance_model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer)

	def create_dataset(self, n_train=None):
		nEnd = self.n_train_init
		if n_train is not None: 
			if n_train>len(self.dist_train):
				nEnd = n_train
			else:
				print('We already have {} samples in the log-likelihood training set.'.format(len(self.dist_train)))

		nStart = len(self.dist_train)

		if nEnd>nStart:
			theta_train_ = self.theta_sampler(nEnd-nStart)
			self.theta_train = theta_train_ if self.theta_train.size==0 else np.vstack((self.theta_train, theta_train_))

		if self.theta_train.shape[0]>self.dist_train.shape[0]:
			sim_train_   = self.simulator(self.theta_train[self.dist_train.shape[0]:self.theta_train.shape[0]])
			dist_train_  = self.distance(sim_train_, self.obs)
			self.dist_train  = np.append(self.dist_train, dist_train_)

	def learn_distance(self, n_train=None):
		self.create_dataset(n_train=n_train)

		if self.distance_model is None:
			self.prepare_distance_model()

		self.distance_model.fit(self.theta_train, self.dist_train)
		print('Score:', self.distance_model.score(self.theta_train, self.dist_train))

	def learn_distance_BO(self, tol=0.01):
		self.create_dataset(n_train=n_train)

		if self.distance_model is None:
			self.prepare_distance_model()

		self.distance_model.fit(self.theta_train, self.dist_train)
		print('Score:', self.distance_model.score(self.theta_train, self.dist_train))

	def run_mcmc(self, n_samples=None):
		assert self.distance_model is not None
		if n_samples is not None:
			self.mcmc_sampler_info['n_samples'] = n_samples

		def log_prior(theta):
			for i,ke in enumerate(self.theta_range.keys()):
				if theta[i]<self.theta_range[ke][0] or theta[i]>self.theta_range[ke][1]:
					return -np.inf
			return 0.0

		def log_probability(theta):
			lp = log_prior(theta)
			if not np.isfinite(lp):
				return -np.inf
			if theta.ndim==1: theta = theta[None,:]
			d_mean, d_std = self.distance_model.predict(theta, return_std=True)
			return lp - d_mean**2 #/d_std**2

		# self.log_prior = log_prior
		# self.log_probability = log_probability

		pos_fn = lambda n=1: np.array([np.random.random(n)*(self.theta_range[ke][1]-self.theta_range[ke][0])+self.theta_range[ke][0] for ke in self.theta_range.keys()]).T
		pos = pos_fn(self.mcmc_sampler_info['nwalkers'])
		nwalkers, ndim = pos.shape
		n_samples = self.mcmc_sampler_info['n_samples'] if 'n_samples' in self.mcmc_sampler_info.keys() else 5**ndim

		print(self.mcmc_sampler_info['n_samples'])
		sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(), 
								pool=self.mcmc_sampler_info['pool'], 
								backend=self.mcmc_sampler_info['backend']
								)
		sampler.run_mcmc(pos, self.mcmc_sampler_info['n_samples'], progress=True)
		self.sampler = sampler

		# tau = sampler.get_autocorr_time()
		# print('Autocorrelation time:', tau)

		flat_samples = sampler.get_chain(discard=100, thin=1, flat=True)
		return flat_samples

