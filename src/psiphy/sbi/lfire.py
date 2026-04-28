import numpy as np
from sklearn.model_selection import KFold
try:
    from scipy.integrate import simpson as simps
except ImportError:
    from scipy.integrate import simps
import warnings 
warnings.filterwarnings("ignore")
from ..utils import distances
from ..utils import helpers as hf
from ..utils import bayesian_opt as bopt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

def _grid_bounds(bounds, n_grid=20):
	def add_dim_to_grid(bound, n_grid=100, init_grid=None):
		if init_grid is None:
			final_grid = np.linspace(bound[0], bound[1], n_grid)
		else:
			final_grid = [np.append(gg,nn) if type(gg)==np.ndarray else [gg,nn] for gg in init_grid for nn in np.linspace(bound[0], bound[1], n_grid)]
		return np.array(final_grid)
	ndim_param = bounds.shape[0]
	grid = add_dim_to_grid(bounds[0], n_grid=n_grid, init_grid=None)
	if ndim_param==1: return grid
	for bound in bounds[1:]:
		grid = add_dim_to_grid(bound, n_grid=n_grid, init_grid=grid)
	return grid


class LFIRE_core:
	def __init__(self, simulator, observation, prior, bounds, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, thetas=None, verbose=True, penalty='l1', n_jobs=4, clfy=None):
		#self.N_init  = N_init
		self.simulator = simulator
		#self.distance  = distance
		self.verbose = verbose
		self.penalty = penalty
		self.y_obs = observation
		self.param_names = [kk for kk in prior]
		self.param_bound = bounds
		self.bounds = np.array([bounds[kk] for kk in self.param_names])
		self.n_m = n_m
		self.n_theta = n_theta
		self.n_grid_out = n_grid_out
		self.n_jobs = n_jobs
		self.clfy   = clfy

		if sim_out_den is not None: 
			self.sim_out_den = sim_out_den
			self.n_theta = sim_out_den.shape[0]
		else: 
			self.sim_denominator()  

		if thetas is None: self.theta_grid()
		else: self.thetas = thetas 

	def sample_prior(self, kk):
		return self.param_bound[kk][0]+(self.param_bound[kk][1]-self.param_bound[kk][0])*np.random.uniform()

	def theta_grid(self):
		self.thetas = _grid_bounds(self.bounds, n_grid=self.n_grid_out)

	def sim_denominator(self):
		print('Simulating the marginalisation data set.')
		params  = np.array([[self.sample_prior(kk) for kk in self.param_names] for i in range(self.n_m)]).squeeze()
		self.sim_out_den = np.array([self.simulator(i) for i in params])

	def sim_numerator(self, theta, n_theta):
		self.sim_out_num = np.array([self.simulator(theta) for i in range(self.n_theta)])

	def ratio(self, theta, sim_out_num=None, get_score=False):
		n_m = self.sim_out_den.shape[0]
		if sim_out_num is None:
			self.sim_numerator(theta, self.n_theta)
		else:
			self.sim_out_num = sim_out_num
		sim_out_num, sim_out_den = self.sim_out_num, self.sim_out_den
		X = np.vstack((sim_out_num,sim_out_den))
		y = np.hstack((np.ones(sim_out_num.shape[0]),np.zeros(sim_out_den.shape[0])))

		clfy = LogisticRegressionCV(penalty=self.penalty, solver='saga', n_jobs=self.n_jobs) if self.clfy is None else self.clfy
		clfy.fit(X, y)

		sim_out_true = np.array([self.y_obs])
		#y_pred = clfy.predict(sim_out_true), 
		y_pred_prob  = clfy.predict_proba(sim_out_true).squeeze()
		n_theta, n_m = sim_out_num.shape[0], sim_out_den.shape[0]
		rr = (n_m/n_theta)*y_pred_prob[1]/y_pred_prob[0]
		rr = 1 if rr>1 else rr
		if get_score: return rr, clfy.score(X,y)
		return rr

	def run(self, thetas=None, n_grid_out=100, sim_out_num=None):
		if thetas is not None: self.thetas = thetas
		self.posterior = np.zeros(self.thetas.shape[0])
		for i, theta in enumerate(self.thetas):
			r0 = self.ratio(theta, sim_out_num=sim_out_num)
			self.posterior[i] = r0
			if self.verbose:
				if np.array(theta).size==1: theta = [theta]
				msg = ','.join(['{0:.3f}'.format(th) for th in theta]) 
				print('Pr({0:}) = {1:.5f}'.format(msg,r0))
				print('Completed: {0:.2f} %'.format(100*(i+1)/self.thetas.shape[0]))


class LFIRE_TrainingSetAuto:
	def __init__(self, simulator, observation, prior, bounds, n_init=10, n_step=1, n_max=100, n_grid_out=25, thetas=None, verbose=True, penalty='l1', n_jobs=4, clfy=None, lfire=None):
		self.n_init = n_init
		self.n_m = None
		self.n_theta = None
		self.n_step  = n_step
		self.n_max   = n_max
		self.n_grid_out = n_grid_out

		self.simulator = simulator
		self.verbose = verbose
		self.penalty = penalty
		self.y_obs  = observation
		self.prior  = prior
		self.bounds = bounds
		self.thetas = thetas
		self.n_jobs = n_jobs
		self.clfy   = clfy

		self.lfire = LFIRE if lfire is None else lfire

	def run(self, thetas=None, n_grid_out=100, sim_out_num=None):
		self.clfy_score_mean = {}
		#self.Ns = []
		Ns_m = np.arange(self.n_init, self.n_max, self.n_step).astype(int)
		Ns_theta = np.arange(self.n_init, self.n_max, self.n_step).astype(int)
		for ni_m in Ns_m:
			for ni_t in Ns_theta:
				self.lfi = self.lfire(self.simulator, self.y_obs, self.prior, self.bounds, sim_out_den=None, n_m=ni_m, n_theta=ni_t, n_grid_out=self.n_grid_out, thetas=thetas, verbose=self.verbose, penalty=self.penalty, n_jobs=self.n_jobs, clfy=self.clfy)
				if thetas is not None: self.lfi.thetas = thetas
				self.lfi.posterior = np.zeros(self.lfi.thetas.shape[0])
				self.clfy_score = np.zeros(self.lfi.thetas.shape[0])
				for i, theta in enumerate(self.lfi.thetas):
					r0, s0 = self.lfi.ratio(theta, sim_out_num=sim_out_num, get_score=True)
					self.lfi.posterior[i] = r0
					self.clfy_score[i] = s0
					if self.verbose:
						if np.array(theta).size==1: theta = [theta]
						msg = ','.join(['{0:.3f}'.format(th) for th in theta]) 
						print('Pr({0:}) = {1:.5f}'.format(msg,r0))
						print('Completed: {0:.2f} %'.format(100*(i+1)/self.lfi.thetas.shape[0]))
				print(ni_m,ni_t,self.clfy_score.mean())
				self.clfy_score_mean[ni_m,ni_t] = self.clfy_score.mean()



class LFIRE_BayesianOpt:
	def __init__(self, simulator, observation, prior, bounds, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, thetas=None, n_init=10, max_iter=1000, tol=1e-5, verbose=True, penalty='l1', n_jobs=4, clfy=None, lfire=None, simulate_corner=True, exploitation_exploration=None, sigma_tol=0.001, model_pdf=None, params=None):
		self.n_init     = n_init
		self.max_iter   = max_iter
		self.tol        = tol
		self.n_m        = n_m 
		self.n_theta    = n_theta
		self.n_grid_out = n_grid_out

		self.simulator = simulator
		self.verbose   = verbose
		self.penalty   = penalty
		self.y_obs  = observation
		self.prior  = prior
		self.bounds = bounds
		self.thetas = thetas
		self.n_jobs = n_jobs
		self.clfy   = clfy
		self.exploitation_exploration = exploitation_exploration
		self.sigma_tol = sigma_tol

		self.lfire = LFIRE if lfire is None else lfire
		if model_pdf is None:
			kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-3, 10.0), nu=1.5)
			model_pdf = GaussianProcessRegressor(kernel=kernel)
		self.gpr = model_pdf

		self.lfi = self.lfire(self.simulator, self.y_obs, self.prior, self.bounds, sim_out_den=sim_out_den, n_m=self.n_m, n_theta=self.n_theta, n_grid_out=self.n_grid_out, thetas=thetas, verbose=self.verbose, penalty=self.penalty, n_jobs=self.n_jobs, clfy=self.clfy)
		self.theta_out = self.lfi.thetas

		params_corner = self.corner_to_theta() if simulate_corner else None
		if params is None:
			self.params   = np.array([[self.lfi.sample_prior(kk) for kk in self.lfi.param_names] for i in range(self.n_init if params_corner is None else self.n_init-params_corner.shape[0])]).squeeze()
		else:
			self.params   = params[:self.n_init,:]
		if params_corner is not None: self.params = np.concatenate((params_corner, self.params), axis=0)

		self.JS_dist = []
		self.posterior_theta = []

	def corner_to_theta(self):
		pa = _grid_bounds(self.lfi.bounds, n_grid=2)
		return pa

	def _adjust_shape(self, abc):
		return abc.reshape(-1,1) if abc.ndim==1 else abc

	def run(self, max_iter=None, tol=None):
		if max_iter is not None: self.max_iter = max_iter
		if tol is not None: self.tol = tol
		# Initial grid
		self.posterior_params = np.zeros(self.params.shape[0])
		if len(self.posterior_theta)==0: #start_iter<self.n_init:
			print('Initializing in a coarse parameter space.')
			for i, theta in enumerate(self.params):
				r0 = self.lfi.ratio(theta)
				self.posterior_params[i] = r0
				if self.verbose:
					if np.array(theta).size==1: theta = [theta]
					msg = ','.join(['{0:.3f}'.format(th) for th in theta]) 
					print('Pr({0:}) = {1:.5f}'.format(msg,r0))
					print('Completed: {0:.2f} %'.format(100*(i+1)/self.max_iter))

		X, y = self._adjust_shape(self.params), self.posterior_params
		self.gpr.fit(X, y)
		posterior_theta_next, sigma_theta = self.gpr.predict(self._adjust_shape(self.theta_out), return_std=True)
		posterior_theta_next[posterior_theta_next<0] = 0
		posterior_theta_next[posterior_theta_next>1] = 1
		self.posterior_theta.append(posterior_theta_next)

		# Next points
		print('Further sampling the parameter space with Bayesian Optimisation.')
		start_iter = self.params.size
		condition1 = False
		for n_iter in range(start_iter,self.max_iter):
			if condition1: 
				print('Stopped as extreme tolerance reached.')
				break

			if self.sigma_tol is not None:
				self.exploitation_exploration = 1./self.sigma_tol if np.any(sigma_theta>self.sigma_tol) else 1.
			#X_next = bopt.propose_location(bopt.expected_improvement, self._adjust_shape(self.params), self.posterior_params, self.gpr, self.lfi.bounds, n_restarts=10).T
			X_next = bopt.propose_location(bopt.GP_UCB_posterior_space, self._adjust_shape(self.params), self.posterior_params, self.gpr, self.lfi.bounds, n_restarts=10, xi=self.exploitation_exploration).T
			self.params = np.vstack((self._adjust_shape(self.params), X_next))
			r_next = self.lfi.ratio(self.params[-1])
			self.posterior_params = np.hstack((self.posterior_params, r_next))
	
			posterior_theta_old  = self.posterior_theta[-1]
			X, y = self._adjust_shape(self.params), self.posterior_params
			self.gpr.fit(X, y)
			posterior_theta_next, sigma_theta = self.gpr.predict(self._adjust_shape(self.theta_out), return_std=True)
			posterior_theta_next[posterior_theta_next<0] = 0
			posterior_theta_next[posterior_theta_next>1] = 1
			self.posterior_theta.append(posterior_theta_next)
			js = distances.jensenshannon(posterior_theta_old, posterior_theta_next)
			self.JS_dist.append(js)

			if self.verbose:
				msg = ','.join(['{0:.3f}'.format(th) for th in self.params[-1]]) 
				print('Pr({0:}) = {1:.5f}'.format(msg,r_next))
				print('JS = {0:.5f}'.format(js))
				print('Completed: {0:.2f} %'.format(100*(n_iter+1)/self.max_iter))

			condition1 = js<self.tol

		self.posterior = self.posterior_theta[-1]
		self.thetas    = self.theta_out
		self.param_names = self.lfi.param_names


class LFIRE_BayesianOpt_ShrinkSpace:
	def __init__(self, simulator, observation, prior, bounds, sim_out_den=None, n_m=100, n_theta=100, n_grid_out=100, thetas=None, n_init=10, max_iter=1000, shrink_condition={'CI':95, 'n':5}, tol=1e-5, verbose=True, penalty='l1', n_jobs=4, clfy=None, lfire=None, simulate_corner=True, exploitation_exploration=1):
		self.n_init     = n_init
		self.tol        = tol
		self.n_m        = n_m 
		self.n_theta    = n_theta
		self.n_grid_out = n_grid_out

		self.shrink_CI = shrink_CI
		self.max_iter  = [max_iter for i in range(len(shrink_CI)+1)] if isinstance(max_iter, (int)) else max_iter
		self.max_iter_tot = np.array(self.max_iter).sum()

		self.simulator = simulator
		self.verbose   = verbose
		self.penalty   = penalty
		self.y_obs  = observation
		self.prior  = prior
		self.bounds = bounds
		self.thetas = thetas
		self.n_jobs = n_jobs
		self.clfy   = clfy
		self.exploitation_exploration = exploitation_exploration

		self.lfire = LFIRE if lfire is None else lfire
		self.gpr = GaussianProcessRegressor()

		self.lfi = self.lfire(self.simulator, self.y_obs, self.prior, self.bounds, sim_out_den=None, n_m=self.n_m, n_theta=self.n_theta, n_grid_out=self.n_grid_out, thetas=thetas, verbose=self.verbose, penalty=self.penalty, n_jobs=self.n_jobs, clfy=self.clfy)
		self.theta_out = self.lfi.thetas

		params_corner = self.corner_to_theta() if simulate_corner else None
		self.params   = np.array([[self.lfi.sample_prior(kk) for kk in self.lfi.param_names] for i in range(self.n_init if params_corner is None else self.n_init-params_corner.shape[0])]).squeeze()
		if params_corner is not None: self.params = np.concatenate((params_corner, self.params), axis=0)

		self.JS_dist = []
		self.posterior_theta = []

	def corner_to_theta(self):
		pa = _grid_bounds(self.lfi.bounds, n_grid=2)
		return pa

	def _adjust_shape(self, abc):
		return abc.reshape(-1,1) if abc.ndim==1 else abc

	def run(self, max_iter=None, tol=None):
		if max_iter is not None: self.max_iter = max_iter
		if tol is not None: self.tol = tol
		# Initial grid
		self.posterior_params = np.zeros(self.params.shape[0])
		if len(self.posterior_theta)==0: #start_iter<self.n_init:
			print('Initializing in a coarse parameter space.')
			for i, theta in enumerate(self.params):
				r0 = self.lfi.ratio(theta)
				self.posterior_params[i] = r0
				if self.verbose:
					if np.array(theta).size==1: theta = [theta]
					msg = ','.join(['{0:.3f}'.format(th) for th in theta]) 
					print('Pr({0:}) = {1:.5f}'.format(msg,r0))
					print('Completed: {0:.2f} %'.format(100*(i+1)/self.max_iter_tot))

		X, y = self._adjust_shape(self.params), self.posterior_params
		self.gpr.fit(X, y)
		posterior_theta_next = self.gpr.predict(self._adjust_shape(self.theta_out))
		posterior_theta_next[posterior_theta_next<0] = 0
		posterior_theta_next[posterior_theta_next>1] = 1
		self.posterior_theta.append(posterior_theta_next)

		# Next points
		print('Further sampling the parameter space with Bayesian Optimisation.')
		start_iter = len(self.params)
		condition1 = False
		for n_iter in range(start_iter,self.max_iter[0]):
			if condition1: break
			#X_next = bopt.propose_location(bopt.expected_improvement, self._adjust_shape(self.params), self.posterior_params, self.gpr, self.lfi.bounds, n_restarts=10).T
			X_next = bopt.propose_location(bopt.GP_UCB_posterior_space, self._adjust_shape(self.params), self.posterior_params, self.gpr, self.lfi.bounds, n_restarts=10, xi=self.exploitation_exploration).T
			self.params = np.vstack((self._adjust_shape(self.params), X_next))
			r_next = self.lfi.ratio(self.params[-1])
			self.posterior_params = np.hstack((self.posterior_params, r_next))
	
			posterior_theta_old  = self.posterior_theta[-1]
			X, y = self._adjust_shape(self.params), self.posterior_params
			self.gpr.fit(X, y)
			posterior_theta_next = self.gpr.predict(self._adjust_shape(self.theta_out))
			posterior_theta_next[posterior_theta_next<0] = 0
			posterior_theta_next[posterior_theta_next>1] = 1
			self.posterior_theta.append(posterior_theta_next)
			js = distances.jensenshannon(posterior_theta_old, posterior_theta_next)
			self.JS_dist.append(js)

			if self.verbose:
				msg = ','.join(['{0:.3f}'.format(th) for th in self.params[-1]]) 
				print('Pr({0:}) = {1:.5f}'.format(msg,r_next))
				print('JS = {0:.5f}'.format(js))
				print('Completed: {0:.2f} %'.format(100*(n_iter+1)/self.max_iter_tot))

			condition1 = js<self.tol

		self.posterior = self.posterior_theta[-1]
		self.thetas    = self.theta_out
		self.param_names = self.lfi.param_names



# LFIRE = LFIRE_core
try:
    import emcee
    _emcee_EnsembleSampler = emcee.EnsembleSampler
except ImportError:
    emcee = None
    class _emcee_EnsembleSampler:
        pass

class LFIRE(_emcee_EnsembleSampler):
	"""docstring for LFIRE"""
	def __init__(self, **arg):
		super(LFIRE, self).__init__(**arg)
		#self.arg = arg


