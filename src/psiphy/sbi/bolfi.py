import numpy as np
import warnings, random
warnings.filterwarnings("ignore")

import pickle
from multiprocessing import Pool, cpu_count
from glob import glob
from time import time, sleep
from scipy.stats import gaussian_kde

try:
    import emcee
except ImportError:
    emcee = None

try:
    from dynesty import NestedSampler, DynamicNestedSampler
    from dynesty.utils import resample_equal
    from dynesty import plotting as dyplot
except ImportError:
    NestedSampler = DynamicNestedSampler = resample_equal = dyplot = None

from ..utils import distances
from ..utils import bayesian_opt as bopt
from ..utils import helpers as hf

try:
    from skopt import gp_minimize
    from skopt import dump, load
    from skopt import callbacks
    from skopt.callbacks import CheckpointSaver
except ImportError:
    gp_minimize = dump = load = callbacks = CheckpointSaver = None

try:
    from GPyOpt.methods import BayesianOptimization
except ImportError:
    BayesianOptimization = None
# BayesianOptimization = bopt.BayesianOptimization_GPyOpt

def dict_get_remove(a, key, default=None, remove=True):
	out = a.get(key, default)
	if remove and key in a.keys(): 
		a.pop(key)
	return out, a

def _dynesty_run(log_probability, prior_range, **kwargs):
	space = prior_range if isinstance(prior_range,list) else [(prior_range[ke][0], prior_range[ke][1]) for ke in prior_range.keys()]
	mins  = np.array([ii[0] for ii in space])
	maxs  = np.array([ii[1] for ii in space])
	def prior_transform(theta):
		return mins + (maxs-mins)*theta

	n_jobs = kwargs.get('n_jobs', 1)
	nlive  = kwargs.get('nlive', 1024)      # number of live points
	bound  = kwargs.get('bound', 'multi')   # use MutliNest algorithm for bounds
	sample = kwargs.get('sample', 'unif')   # uniform sampling
	tol  = kwargs.get('tol', 0.01)          # the stopping criterion
	ndim = len(mins)

	if n_jobs>1:
		with Pool() as pool:
			# sampler = NestedSampler(log_probability, prior_transform, ndim, bound=bound, sample=sample, nlive=nlive, pool=pool, queue_size=8)
			sampler = DynamicNestedSampler(log_probability, prior_transform, ndim, bound=bound, sample=sample, pool=pool, queue_size=8)
			t0 = time()
			sampler.run_nested(nlive_init=nlive, dlogz_init=tol, print_progress=True) 
			t1 = time()
	else:
		# sampler = NestedSampler(log_probability, prior_transform, ndim, bound=bound, sample=sample, nlive=nlive)
		sampler = DynamicNestedSampler(log_probability, prior_transform, ndim, bound=bound, sample=sample)
		t0 = time()
		sampler.run_nested(nlive_init=nlive, dlogz_init=tol, print_progress=True) 
		t1 = time()
	timedynesty = (t1-t0)
	print("Time taken to run 'dynesty' (in static mode) is {:.2f} seconds".format(timedynesty))
	res = sampler.results # get results dictionary from sampler
	# print(res.summary())
	# draw posterior samples
	weights = np.exp(res['logwt'] - res['logz'][-1])
	samples_dynesty = resample_equal(res.samples, weights)
	return {'weighted_samples': samples_dynesty, 'samples': res.samples, 'weights': weights}

def _emcee_run(n_samples, nwalkers, ndim, log_probability, pos, filename, reset_sampler, n_jobs=1, **kwargs_emcee):
	print(filename)
	moves = kwargs_emcee.get('moves', emcee.moves.WalkMove())
	print('Move used:', moves)

	backend = emcee.backends.HDFBackend(filename) 
	if reset_sampler or not glob(filename): 
		backend.reset(nwalkers, ndim)
		if n_jobs>1:
			with Pool(n_jobs) as pool:
				sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend, pool=pool, moves=moves)
				sampler.run_mcmc(pos, n_samples, progress=True);
		else:
			sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend, moves=moves)
			sampler.run_mcmc(pos, n_samples, progress=True);
	else:
		old_iter = backend.iteration
		print('Previous run contains {} iterations.'.format(old_iter))
		if n_samples>old_iter:
			print('Continuing the chains...')
			sleep(0.5)
			if n_jobs>1:
				with Pool(n_jobs) as pool:
					sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend, pool=pool, moves=moves)
					sampler.run_mcmc(None, n_samples-old_iter, progress=True);
			else:
				sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend, moves=moves)
				sampler.run_mcmc(None, n_samples-old_iter, progress=True);
		else:
			sampler = emcee.EnsembleSampler(backend.shape[0], backend.shape[1], log_probability, backend=backend, moves=moves)
	return sampler

def read_sampler_emcee(filename=None, sampler=None):
	if filename is not None:
		backend = emcee.backends.HDFBackend(filename) 
		sampler = emcee.EnsembleSampler(backend.shape[0], backend.shape[1], None, backend=backend)
	# if sampler is None:
	# 	sampler = self.sampler 
	return sampler 

def get_chain_emcee(discard=0, filename=None, sampler=None):
	sampler = read_sampler_emcee(filename=filename, sampler=sampler)
	tau = sampler.get_autocorr_time(quiet=True)
	print('autocorr_time:', tau)
	if discard=='tau': discard = 3*max(tau.astype(int))
	flat_samples = sampler.get_chain(discard=discard, flat=True) 
	flat_logprob = sampler.get_log_prob(discard=discard, flat=True) 
	# print(flat_samples.shape, flat_logprob.shape)
	return flat_samples, flat_logprob


class BOLFI: 
	''' 
	Bayesian optimisation for Likelihood-free Inference. 
	''' 
	
	def __init__(self, distance, prior_range, obs=None, distance_kernel='exp', verbose=True, package='GPyOpt', learn_log_dist=False): 
		# self.simulator = simulator 
		self.distance    = distance 
		self.prior_range = prior_range 
		if distance_kernel in [None, 'exp']: self.distance_kernel = lambda u: np.exp(-u/2)  
		self.verbose = verbose 
		self.obs = obs 
		self.package = package 
		self.learn_log_dist = learn_log_dist
	
	def save_likelihood_model(self, filename): 
		try: self.J_save = {'model': self.J_full.models[-1],  
		                     'X': self.J_full.x_iters,  
		                     'y': self.J_full.func_vals} 
		except: self.J_save = {'model': self.J_full.model,  
		                        'X': self.J_full.X,  
		                        'y': self.J_full.Y} 
		pickle.dump(self.J_save, open(filename,'wb')) 
		print('Saved likelihood model:', filename) 

	def load_likelihood_model(self, filename): 
		self.J_save = pickle.load(open(filename, 'rb')) 
		try: self.cook_likelihood(self.J_full.models[-1])#, package='skopt') 
		except: self.cook_likelihood(self.J_full.model)#, package='GPyOpt') 
		print('Previous likelihood model loaded.') 
	
	def cook_likelihood(self, gpmodel):
		if self.package=='GPyOpt': 
			self.gp_Jmu = lambda x: gpmodel.predict(x if x.ndim==2 else x[None,:])[0].squeeze()
			self.gp_Jsigma = lambda x: gpmodel.predict(x if x.ndim==2 else x[None,:])[1].squeeze()
		else: 
			self.gp_Jmu = lambda x: gpmodel.predict(x if x.ndim==2 else x[:,None])
			self.gp_Jsigma = lambda x: gpmodel.predict(x if x.ndim==2 else x[:,None], return_std)[1]

		if self.learn_log_dist:
			self.gp_logL = lambda x: self.distance_kernel(np.exp(self.gp_Jmu(x))) 
		else:
			self.gp_logL = lambda x: self.distance_kernel(self.gp_Jmu(x)) 
	  
	def learn_likelihood(self, obs=None, gpmodel=None, reset_model=False,  
		n_calls=100, n_random_starts=None, n_initial_points=None,  
		initial_point_generator='random', acq_func='EI', acq_optimizer='auto',  
		random_state=None, verbose=False, callback=None, n_points=10000,  
		n_restarts_optimizer=5, xi=0.01, kappa=1.96, noise='gaussian',  
		n_jobs=1, model_queue_size=None, batch_size=1, filename=None, 
		acquisition_optimizer_type='lbfgs', **kwargs): 

		if obs is not None: self.obs = obs 
		if self.obs is None: 
			print('Provide obs to learn the likelihood.') 
			return None 

		x0, y0   = None, None 
		callback = None 
		if filename is not None and not reset_model: 
			# checkpoint_saver = CheckpointSaver(filename, compress=9) # keyword arguments will be passed to `skopt.dump` 
			# callback = [checkpoint_saver] 
			if glob(filename): 
				self.load_likelihood_model(filename) 
				x0 = self.J_save['X'] 
				y0 = self.J_save['y'] 
				if gpmodel is None: gpmodel = self.J_save['model'] 

		space = self.prior_range if isinstance(self.prior_range,list) else [(self.prior_range[ke][0], self.prior_range[ke][1]) for ke in self.prior_range.keys()] 
		if n_initial_points is None: n_initial_points = 5**len(space) 

		print('Learning synthetic likelihood using Bayesian Optimisation...') 
		if self.learn_log_dist:
			f = lambda x: np.log(self.distance(x)) #self.simulator(x), self.obs) 
		else:
			f = lambda x: self.distance(x) #self.simulator(x), self.obs) 

		if self.package=='skopt': 
			res = gp_minimize(f,          # the function to minimize 
					space,              # the bounds on each dimension of x 
					base_estimator=gpmodel,  
					n_calls=n_calls,  
					n_random_starts=n_random_starts,  
					n_initial_points=n_initial_points,  
					initial_point_generator=initial_point_generator,  
					acq_func=acq_func,  
					acq_optimizer=acq_optimizer,  
					x0=x0, y0=y0,  
					random_state=random_state,  
					verbose=verbose,  
					callback=callback,  
					n_points=n_points,  
					n_restarts_optimizer=n_restarts_optimizer,  
					xi=xi,  
					kappa=kappa,  
					noise=noise,  
					n_jobs=n_jobs,  
					# model_queue_size=model_queue_size, 
					) 
			self.J_full = res  
			self.cook_likelihood(self.J_full.models[-1]) 
		else: 
			domain = [{'name': 'var_{}'.format(i), 'type': 'continuous', 'domain': (space[i][0],space[i][1])} for i in range(len(space))] 
			res = BayesianOptimization( 
						f, 
						domain=domain, 
						constraints=None, 
						cost_withGradients=None, 
						model_type='GP', 
						X=x0, 
						Y=y0, 
						initial_design_numdata=n_initial_points, 
						initial_design_type=initial_point_generator, 
						acquisition_type=acq_func, 
						normalize_Y=False, 
						exact_feval=False, 
						acquisition_optimizer_type=acquisition_optimizer_type, 
						model_update_interval=1, 
						evaluator_type='sequential', #'thompson_sampling' 
						batch_size=batch_size, 
						num_cores=n_jobs, 
						verbosity=True, 
						verbosity_model=True, 
						maximize=False, 
						de_duplication=False, 
						**kwargs
						) 
			res.run_optimization(max_iter=n_calls, verbosity=verbose) 
			self.J_full = res  
			self.cook_likelihood(self.J_full.model) 
		print('...done') 
		if filename is not None: self.save_likelihood_model(filename)

	def sample_posterior(self, n_samples=5000, method='MCMC', **kwargs):
		# print(kwargs)
		log_prior = kwargs.get('log_prior', None)
		if method=='MCMC':
			print('MCMC sampling using emcee...')
			filename, kwargs = dict_get_remove(kwargs, 'filename', None) # filename = kwargs.get('filename', None)
			nwalkers, kwargs = dict_get_remove(kwargs, 'nwalkers', 16)   # nwalkers = kwargs.get('nwalkers', 16)
			reset_sampler, kwargs = dict_get_remove(kwargs, 'reset_sampler', True) # reset_sampler = kwargs.get('reset_sampler', True)
			discard, kwargs = dict_get_remove(kwargs, 'discard', 0)      # discard = kwargs.get('discard', 0)

			sampler = self.sample_MCMC(n_samples, log_prior=log_prior, nwalkers=nwalkers, filename=filename, 
									reset_sampler=reset_sampler, n_jobs=4, **kwargs)
			self.sampler_info = sampler
			samples, logprobs = get_chain_emcee(discard=discard, sampler=sampler)
		elif method=='NestedSampling':
			res = self.sample_NestedSampling(**kwargs)
			self.sampler_info = res 
			return res['samples']
		elif method=='IS':
			print('Importance Sampling...')
			proposal, kwargs = dict_get_remove(kwargs, 'proposal', 'uniform')
			samples = self.sample_IS(n_samples, log_prior=log_prior, proposal=proposal)
		print('...done')
		return samples 

	def sample_MCMC(self, n_samples, log_prior=None, nwalkers=16,
		filename=None, reset_sampler=True, n_jobs=4, **kwargs_emcee):

		space = self.prior_range if isinstance(self.prior_range,list) else [(self.prior_range[ke][0], self.prior_range[ke][1]) for ke in self.prior_range.keys()]
		mins  = np.array([ii[0] for ii in space])
		maxs  = np.array([ii[1] for ii in space])
		pos = mins+(maxs-mins) * np.random.uniform(0,1,size=(nwalkers, len(mins)))
		nwalkers, ndim = pos.shape

		if log_prior is None: 
			log_prior = lambda x: np.log(np.product(mins<=x,axis=1)*np.product(x<=maxs,axis=1)) if x.ndim==2 else np.log(np.product(mins<=x)*np.product(x<=maxs))
		log_probability = lambda x: np.asscalar(self.gp_logL(x)+log_prior(x)) if np.isfinite(log_prior(x)) else -np.inf 

		if filename is None: filename = 'dummy_bolfi_743.h5'
		sampler = _emcee_run(n_samples, nwalkers, ndim, log_probability, pos, filename, reset_sampler, n_jobs=1, **kwargs_emcee)
		return sampler

	def sample_NestedSampling(self, log_prior=None, **kwargs):
		space = self.prior_range if isinstance(self.prior_range,list) else [(self.prior_range[ke][0], self.prior_range[ke][1]) for ke in self.prior_range.keys()]
		mins  = np.array([ii[0] for ii in space])
		maxs  = np.array([ii[1] for ii in space])

		if log_prior is None: 
			log_prior = lambda x: np.log(np.product(mins<=x,axis=1)*np.product(x<=maxs,axis=1)) if x.ndim==2 else np.log(np.product(mins<=x)*np.product(x<=maxs))
		log_probability = lambda x: np.asscalar(self.gp_logL(x)+log_prior(x)) if np.isfinite(log_prior(x)) else -np.inf 

		out = _dynesty_run(log_probability, self.prior_range, **kwargs)
		return out

	def sample_IS(self, n_samples, log_prior=None, proposal='uniform'):
		space = self.prior_range if isinstance(self.prior_range,list) else [(self.prior_range[ke][0], self.prior_range[ke][1]) for ke in self.prior_range.keys()]
		mins  = np.array([ii[0] for ii in space])
		maxs  = np.array([ii[1] for ii in space])

		if log_prior is None: 
			log_prior = lambda x: np.log(np.product(mins<=x,axis=1)*np.product(x<=maxs,axis=1)) if x.ndim==2 else np.log(np.product(mins<=x)*np.product(x<=maxs))
		log_probability = lambda x: self.gp_logL(x)+log_prior(x) 
		func = lambda x: np.exp(log_probability(x))
		out  = importance_sampling(func, n_samples, self.prior_range, proposal=proposal)
		return out


def _grid_creator(mins, maxs, n_samples):
	out = np.linspace(mins[0],maxs[0],n_samples)[:,None]
	for mn,mx in zip(mins[1:],maxs[1:]):
		# print(mn, mx, out.shape)
		nx = np.linspace(mn,mx,n_samples)
		out1 = []
		for i1 in out:
			for i2 in nx:
				out1.append(np.append(i1,i2))  
		out = np.array(out1)
	return out 

def importance_sampling(func, n_samples, prior_range, proposal='uniform'):
	space = prior_range if isinstance(prior_range,list) else [(prior_range[ke][0], prior_range[ke][1]) for ke in prior_range.keys()]
	mins  = np.array([ii[0] for ii in space])
	maxs  = np.array([ii[1] for ii in space])
	print('Initialising samples...')
	if proposal=='uniform':
		pos0  = mins+(maxs-mins) * np.random.uniform(0,1,size=(n_samples,len(mins)))
		proposal = lambda x: np.product(mins<=x,axis=1)*np.product(x<=maxs,axis=1)/n_samples if x.ndim==2 else np.product(mins<=x)*np.product(x<=maxs)/n_samples
	elif proposal in ['gaussian', 'normal']:
		pos0  = mins+(maxs-mins) * np.random.normal(0.5,1/3,size=(n_samples,len(mins)))
		proposal = lambda x: np.product(np.exp(-((x-mins)/(maxs-mins)-0.5)**2/(1/3)**2),axis=1) if x.ndim==2 else np.product(np.exp(-((x-mins)/(maxs-mins)-0.5)**2/(1/3)**2))
	elif proposal=='grid':
		pos0 = _grid_creator(mins, maxs, n_samples)
		proposal = lambda x: np.product(mins<=x,axis=1)*np.product(x<=maxs,axis=1)/n_samples if x.ndim==2 else np.product(mins<=x)*np.product(x<=maxs)/n_samples
	print('...done')
	wts  = func(pos0)/proposal(pos0); wts /= wts.sum()
	pos1 = np.array(random.choices(pos0, weights=wts, k=n_samples))
	return {'i': pos0, 'f':pos1, 'w': wts}


def sequential_importance_sampling(func, n_samples, prior_range, proposal='uniform', max_iter=10, kernel='EmpiricalCovariance'):
	if kernel=='EmpiricalCovariance':
		from sklearn.covariance import EmpiricalCovariance
		cov_est = EmpiricalCovariance()
	elif kernel=='LedoitWolf':
		from sklearn.covariance import LedoitWolf
		cov_est = LedoitWolf()
	else:
		cov_est = kernel

	space = prior_range if isinstance(prior_range,list) else [(prior_range[ke][0], prior_range[ke][1]) for ke in prior_range.keys()]
	mins  = np.array([ii[0] for ii in space])
	maxs  = np.array([ii[1] for ii in space])
	# print('Initialising samples...')
	if proposal=='uniform':
		pos0  = mins+(maxs-mins) * np.random.uniform(0,1,size=(n_samples,len(mins)))
		proposal = lambda x: np.product(mins<=x,axis=1)*np.product(x<=maxs,axis=1)/n_samples if x.ndim==2 else np.product(mins<=x)*np.product(x<=maxs)/n_samples
	elif proposal in ['gaussian', 'normal']:
		pos0  = mins+(maxs-mins) * np.random.normal(0.5,1/3,size=(n_samples,len(mins)))
		proposal = lambda x: np.product(np.exp(-((x-mins)/(maxs-mins)-0.5)**2/(1/3)**2),axis=1) if x.ndim==2 else np.product(np.exp(-((x-mins)/(maxs-mins)-0.5)**2/(1/3)**2))
	elif proposal=='grid':
		pos0 = _grid_creator(mins, maxs, n_samples)
		proposal = lambda x: np.product(mins<=x,axis=1)*np.product(x<=maxs,axis=1)/n_samples if x.ndim==2 else np.product(mins<=x)*np.product(x<=maxs)/n_samples
	# print('...done')
	out = {'i': pos0}

	wts  = func(pos0)/proposal(pos0); wts /= wts.sum()
	with tqdm(range(max_iter)) as rounds:
		for i in rounds:
			rounds.set_description('Round = {}/{}'.format(i+1,max_iter))
			pos1 = np.array(random.choices(pos0, weights=wts, k=n_samples))
			cov  = 2*cov_est.fit(pos1).covariance_ #numpy.cov(pos0, aweights=wts0) #
			# print(cov)
			pos2 = np.array([np.random.multivariate_normal(po, cov) for po in pos1])
			ker2 = gaussian_kde(pos2.T)
			wts  = func(pos2)/ker2(pos2.T); wts /= wts.sum()
			pos0 = pos2.copy()
	pos1 = np.array(random.choices(pos0, weights=wts, k=n_samples))
	out['f'] = pos1
	out['w'] = wts
	return out


def SMC_sampling(func, n_samples, prior_range, proposal='uniform', kernel='EmpiricalCovariance'):
	'''
	UNDER CONSTRUCTION
	'''
	if kernel=='EmpiricalCovariance':
		from sklearn.covariance import EmpiricalCovariance
		cov_est = EmpiricalCovariance()
	elif kernel=='LedoitWolf':
		from sklearn.covariance import LedoitWolf
		cov_est = LedoitWolf()
	else:
		cov_est = kernel

	space = prior_range if isinstance(prior_range,list) else [(prior_range[ke][0], prior_range[ke][1]) for ke in prior_range.keys()]
	mins  = np.array([ii[0] for ii in space])
	maxs  = np.array([ii[1] for ii in space])

	if proposal=='uniform':
		pos0 = mins+(maxs-mins) * np.random.uniform(0,1,size=(n_samples,len(mins)))
		proposal = lambda x: np.product(mins<=x,axis=1)*np.product(x<=maxs,axis=1)/n_samples if x.ndim==2 else np.product(mins<=x)*np.product(x<=maxs)/n_samples
	elif proposal in ['gaussian', 'normal']:
		pos0  = mins+(maxs-mins) * np.random.normal(0.5,1/3,size=(n_samples,len(mins)))
		proposal = lambda x: np.product(np.exp(-((x-mins)/(maxs-mins)-0.5)**2/(1/3)**2),axis=1) if x.ndim==2 else np.product(np.exp(-((x-mins)/(maxs-mins)-0.5)**2/(1/3)**2))
	wts0  = proposal(pos0); wts0 /= wts0.sum()

	max_iter = 5
	with tqdm(range(max_iter)) as rounds:
		for i in rounds:
			rounds.set_description('Round = {}/{}'.format(i+1,max_iter))
			pos1 = np.array(random.choices(pos0, weights=wts0, k=n_samples))
			cov  = 2*cov_est.fit(pos1).covariance_ #numpy.cov(pos0, aweights=wts0) #
			print(cov)
			pos2 = np.array([np.random.multivariate_normal(po, cov) for po in pos1])
			wts2 = func(pos2)/wts0; wts2 /= wts2.sum()
			pos0, wts0 = pos2.copy(), wts2.copy()
	return pos1, wts





