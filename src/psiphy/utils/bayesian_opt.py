import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy import integrate
from . import sampling_space as smp
from . import helpers as hf

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25, xi=1):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    It maximises the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        bounds: The lower and upper bounds of your X space.
        n_restarts: Number of times to restart minimser from a different random start point.
        xi : Tuning parameter, such as Exploitation-exploration trade-off parameter.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 2*Y_sample.max()#1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr, xi=xi)
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)


def propose_location_nSphere(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25, xi=1, inside_nsphere=True, batch=1):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    It maximises the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        bounds: The lower and upper bounds of your X space.
        n_restarts: Number of times to restart minimser from a different random start point.
        xi : Tuning parameter, such as Exploitation-exploration trade-off parameter.
    Returns:
        Location of the acquisition function maximum.
    '''
    if n_restarts<5*batch:
        print('(n_restarts) parameter is changed to 5x(batch)')
        n_restarts = 5*batch

    dim = X_sample.shape[1]
    min_val = 2*Y_sample.max()#1
    min_x = None

    min_vals, min_xs = [], []

    bound_min = bounds.min(axis=1)
    bound_max = bounds.max(axis=1)
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        if inside_nsphere:
            check = check_inside_nsphere(X, bound_min, bound_max)
            if not check: 
                #print(X, 'check failed')
                return np.inf
        #print(X, 'check passed')
        val = -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr, xi=xi)
        return val.reshape(1) if val.size==1 else val
    
    # Find the best optimum by starting from n_restart different random points.
    if inside_nsphere: start_points = smp.MCS_nsphere(n_params=dim, samples=n_restarts, mins=bound_min, maxs=bound_max)
    else: start_points = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim))
    for x0 in start_points:
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B') 
        if check_inside_nsphere(res.x, bound_min, bound_max):
            min_vals.append(res.fun[0])
            min_xs.append(res.x)       
        # if res.fun < min_val:
        #     min_val = res.fun[0]
        #     min_x = res.x   
    args = _argmin(np.array(min_vals), count=batch)        
    min_val = np.array(min_vals)[args]
    min_x   = np.array(min_xs)[args]
    min_x   = min_x.T if batch>1 else min_x.reshape(-1, 1)
    return min_x



def GP_UCB_posterior_space(X, X_sample, Y_sample, gpr, xi=100):
    '''
    Computes the Upper Confidence Bound (UCB) at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    With this acquisition function, we want to find the space maximises.
    
    Args:
        X: Points at which UCB shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    ucb = mu + xi*sigma
    #ucb = xi*sigma

    return ucb

def negativeGP_LCB(X, X_sample, Y_sample, gpr, xi=100):
    '''
    Computes the Lower Confidence Bound (UCB) at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    With this acquisition function, we want to find the space maximises.
    
    Args:
        X: Points at which UCB shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    lcb = mu - xi*sigma
    #lcb = xi*sigma

    return -lcb

def check_inside_nsphere(X, bound_min, bound_max):
    check = np.sum(((X-bound_min)/(bound_max-bound_min)-0.5)**2, axis=1) if X.ndim>1 else np.sum(((X-bound_min)/(bound_max-bound_min)-0.5)**2)
    return check<0.25

def draw_nsphere_surface(bound_min, bound_max, N=100):
    dims = len(bound_min)
    ri = 0.5*np.ones((N))
    for i in range(1,dims-1): ri = np.vstack((ri,np.linspace(0,np.pi,N)))
    ri = np.vstack((ri,np.linspace(0,2*np.pi,N)))
    ri = ri.T
    xi = np.array([hf.spherical_to_cartesian(thet) for thet in ri])
    yi = (xi+0.5)*(bound_max-bound_min)+bound_min
    return yi

def _argmin(x, count=1, axis=None):
    if count==1: return np.argmin(x, axis=axis)
    args = np.argsort(x, axis=axis)
    return args[:count]


def ExpIntVar(X, model, y_opt=0.0, xi=0.01, return_grad=False):
    """
    UNDER CONSTRUCTION
    This is for the scikit-optimize version.

    Use the expected improvement to calculate the acquisition values.
    The conditional probability `P(y=f(x) | x)` form a gaussian with a certain
    mean and standard deviation approximated by the model.
    The EI condition is derived by computing ``E[u(f(x))]``
    where ``u(f(x)) = 0``, if ``f(x) > y_opt`` and ``u(f(x)) = y_opt - f(x)``,
    if``f(x) < y_opt``.
    This solves one of the issues of the PI condition by giving a reward
    proportional to the amount of improvement got.
    Note that the value returned by this function should be maximized to
    obtain the ``X`` with maximum improvement.
    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        Values where the acquisition function should be computed.
    model : sklearn estimator that implements predict with ``return_std``
        The fit estimator that approximates the function through the
        method ``predict``.
        It should have a ``return_std`` parameter that returns the standard
        deviation.
    y_opt : float, default 0
        Previous minimum value which we would like to improve upon.
    xi : float, default=0.01
        Controls how much improvement one wants over the previous best
        values. Useful only when ``method`` is set to "EI"
    return_grad : boolean, optional
        Whether or not to return the grad. Implemented only for the case where
        ``X`` is a single sample.
    Returns
    -------
    values : array-like, shape=(X.shape[0],)
        Acquisition function values computed at X.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if return_grad:
            mu, std, mu_grad, std_grad = model.predict(
                X, return_std=True, return_mean_grad=True,
                return_std_grad=True)

        else:
            mu, std = model.predict(X, return_std=True)

    # check dimensionality of mu, std so we can divide them below
    if (mu.ndim != 1) or (std.ndim != 1):
        raise ValueError("mu and std are {}-dimensional and {}-dimensional, "
                         "however both must be 1-dimensional. Did you train "
                         "your model with an (N, 1) vector instead of an "
                         "(N,) vector?"
                         .format(mu.ndim, std.ndim))

    values = np.zeros_like(mu)
    mask = std > 0
    improve = y_opt - xi - mu[mask]
    scaled = improve / std[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = std[mask] * pdf
    values[mask] = exploit + explore

    if return_grad:
        if not np.all(mask):
            return values, np.zeros_like(std_grad)

        # Substitute (y_opt - xi - mu) / sigma = t and apply chain rule.
        # improve_grad is the gradient of t wrt x.
        improve_grad = -mu_grad * std - std_grad * improve
        improve_grad /= std ** 2
        cdf_grad = improve_grad * pdf
        pdf_grad = -improve * cdf_grad
        exploit_grad = -mu_grad * cdf - pdf_grad
        explore_grad = std_grad * pdf + pdf_grad

        grad = exploit_grad + explore_grad
        return values, grad

    return values


try:
    from GPyOpt.methods import BayesianOptimization
    from GPyOpt.util.arguments_manager import ArgumentsManager
    from GPyOpt.core.task.space import Design_space, bounds_to_space
    from GPyOpt.core.task.objective import SingleObjective
    from GPyOpt.core.task.cost import CostModel
    from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
    from GPyOpt.acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB, AcquisitionEI_MCMC, AcquisitionMPI_MCMC, AcquisitionLCB_MCMC, AcquisitionLP, AcquisitionEntropySearch
    from GPyOpt.acquisitions import AcquisitionBase
    from GPyOpt.util.general import get_quantiles
    _gpyopt_available = True
except ImportError:
    _gpyopt_available = False
    BayesianOptimization = ArgumentsManager = Design_space = bounds_to_space = None
    SingleObjective = CostModel = AcquisitionOptimizer = None
    AcquisitionEI = AcquisitionMPI = AcquisitionLCB = None
    AcquisitionEI_MCMC = AcquisitionMPI_MCMC = AcquisitionLCB_MCMC = None
    AcquisitionLP = AcquisitionEntropySearch = None
    get_quantiles = None
    class AcquisitionBase:  # dummy bases so subclasses below don't fail at definition
        pass
    class ArgumentsManager:
        pass
    class BayesianOptimization:
        pass

class AcquisitionEIV(AcquisitionBase):
    """
    Expected Integrated Variance acquisition function.

    Leclercq (2018) approximation adopted. The class will be generalised later.

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.
    .. Note:: allows to compute the Improvement per unit of cost
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, n_grid=50):
        self.optimizer = optimizer
        super(AcquisitionEIV, self).__init__(model, space, optimizer)
        self.prior_range = self.space
        self.n_grid = n_grid

        if cost_withGradients is not None:
            print('The set cost function is ignored! EIV acquisition does not make sense with cost.')  

        self.X = self._grid_creator(self.prior_range, self.n_grid)
        self.mX, self.sX = self.model.predict(self.X)   
        self.nX = self.X.shape[0]

    def _compute_acq(self, x):
        """
        Computes the Expected Integrated Variance 
        """
        c  = self.model.predict_covariance(np.vstack((self.X,x)))[-1,:-1]
        t2 = c**2/self.sX**2
        vt = (1/4)*np.exp(-self.mX)*(self.sX**2-t2)
        f_acqu = vt.sum()/self.nX # integrate.nquad(lambda x0,x1: module.gp_logL(np.array([x0,x1])), prior_range)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x) 
        f_acqu = -m + self.exploration_weight * s       
        df_acqu = -dmdx + self.exploration_weight * dsdx
        return f_acqu, df_acqu

    def _grid_creator(self, prior_range, n_samples):
        mins, maxs = np.array(prior_range).T
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


class ArgumentsManager_GPyOpt(ArgumentsManager):
    def acquisition_creator(self, acquisition_type, model, space, acquisition_optimizer, cost_withGradients):

        """
        Acquisition chooser from the available options. Extra parameters can be passed via **kwargs.
        """
        acquisition_type = acquisition_type
        model = model
        space = space
        acquisition_optimizer = acquisition_optimizer
        cost_withGradients = cost_withGradients
        acquisition_jitter = self.kwargs.get('acquisition_jitter',0.01)
        acquisition_weight = self.kwargs.get('acquisition_weight',2)

        # --- Choose the acquisition
        if acquisition_type is  None or acquisition_type =='EI':
            return AcquisitionEI(model, space, acquisition_optimizer, cost_withGradients, acquisition_jitter)

        elif acquisition_type =='EI_MCMC':
            return AcquisitionEI_MCMC(model, space, acquisition_optimizer, cost_withGradients, acquisition_jitter)

        elif acquisition_type =='MPI':
            return AcquisitionMPI(model, space, acquisition_optimizer, cost_withGradients, acquisition_jitter)

        elif acquisition_type =='MPI_MCMC':
            return AcquisitionMPI_MCMC(model, space, acquisition_optimizer, cost_withGradients, acquisition_jitter)

        elif acquisition_type =='LCB':
            return AcquisitionLCB(model, space, acquisition_optimizer, None, acquisition_weight)

        elif acquisition_type =='LCB_MCMC':
            return AcquisitionLCB_MCMC(model, space, acquisition_optimizer, None, acquisition_weight)

        elif acquisition_type =='EIV':
            print('Expected Integrated Variance')
            prior_range = self.kwargs.get('prior_range')
            return AcquisitionEIV(model, space, acquisition_optimizer, None, 50)

        else:
            raise Exception('Invalid acquisition selected.')

class BayesianOptimization_GPyOpt(BayesianOptimization):
    def __init__(self, f, domain = None, constraints = None, cost_withGradients = None, model_type = 'GP', X = None, Y = None,
        initial_design_numdata = 5, initial_design_type='random', acquisition_type ='EI', normalize_Y = True,
        exact_feval = False, acquisition_optimizer_type = 'lbfgs', model_update_interval=1, evaluator_type = 'sequential',
        batch_size = 1, num_cores = 1, verbosity=False, verbosity_model = False, maximize=False, de_duplication=False, **kwargs):
        self.modular_optimization = False
        self.initial_iter = True
        self.verbosity = verbosity
        self.verbosity_model = verbosity_model
        self.model_update_interval = model_update_interval
        self.de_duplication = de_duplication
        self.kwargs = kwargs

        # --- Handle the arguments passed via kwargs
        self.problem_config = ArgumentsManager_GPyOpt(kwargs)

        # --- CHOOSE design space
        self.constraints = constraints
        self.domain = domain
        self.space = Design_space(self.domain, self.constraints)

        # --- CHOOSE objective function
        self.maximize = maximize
        if 'objective_name' in kwargs:
            self.objective_name = kwargs['objective_name']
        else:
            self.objective_name = 'no_name'
        self.batch_size = batch_size
        self.num_cores = num_cores
        if f is not None:
            self.f = self._sign(f)
            self.objective = SingleObjective(self.f, self.batch_size,self.objective_name)
        else:
            self.f = None
            self.objective = None

        # --- CHOOSE the cost model
        self.cost = CostModel(cost_withGradients)

        # --- CHOOSE initial design
        self.X = X
        self.Y = Y
        self.initial_design_type  = initial_design_type
        self.initial_design_numdata = initial_design_numdata
        self._init_design_chooser()

        # --- CHOOSE the model type. If an instance of a GPyOpt model is passed (possibly user defined), it is used.
        self.model_type = model_type
        self.exact_feval = exact_feval  # note that this 2 options are not used with the predefined model
        self.normalize_Y = normalize_Y

        if 'model' in self.kwargs:
            if isinstance(kwargs['model'], GPyOpt.models.base.BOModel):
                self.model = kwargs['model']
                self.model_type = 'User defined model used.'
                print('Using a model defined by the used.')
            else:
                self.model = self._model_chooser()
        else:
            self.model = self._model_chooser()

        # --- CHOOSE the acquisition optimizer_type

        # This states how the discrete variables are handled (exact search or rounding)
        kwargs.update({ 'model' : self.model })
        self.acquisition_optimizer_type = acquisition_optimizer_type
        self.acquisition_optimizer = AcquisitionOptimizer(self.space, self.acquisition_optimizer_type, **kwargs)  ## more arguments may come here

        # --- CHOOSE acquisition function. If an instance of an acquisition is passed (possibly user defined), it is used.
        self.acquisition_type = acquisition_type

        if 'acquisition' in self.kwargs:
            if isinstance(kwargs['acquisition'], GPyOpt.acquisitions.AcquisitionBase):
                self.acquisition = kwargs['acquisition']
                self.acquisition_type = 'User defined acquisition used.'
                print('Using an acquisition defined by the used.')
            else:
                self.acquisition = self._acquisition_chooser()
        else:
            self.acquisition = self.acquisition = self._acquisition_chooser()


        # --- CHOOSE evaluator method
        self.evaluator_type = evaluator_type
        self.evaluator = self._evaluator_chooser()

        # --- Create optimization space
        super(BayesianOptimization,self).__init__(  model                  = self.model,
                                                    space                  = self.space,
                                                    objective              = self.objective,
                                                    acquisition            = self.acquisition,
                                                    evaluator              = self.evaluator,
                                                    X_init                 = self.X,
                                                    Y_init                 = self.Y,
                                                    cost                   = self.cost,
                                                    normalize_Y            = self.normalize_Y,
                                                    model_update_interval  = self.model_update_interval,
                                                    de_duplication         = self.de_duplication)










