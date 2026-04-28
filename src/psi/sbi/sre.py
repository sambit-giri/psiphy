import numpy as np 
import emcee
from scipy.special import logit

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


# class emceeSRE(emcee.EnsembleSampler):
# 	def __init__(self, *args):
# 		super().__init__(*args)

class emceeSRE:
	def __init__(self):
		pass
	def ratio_estimation(self, classifier, X, y, test_size=0.1, normalizer=None):
		print('Learning ratio estimator...')
		X, y = X.astype(np.float32), y.astype(np.float64).squeeze()
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
		model = classifier if normalizer is None else Pipeline([('normalize', normalizer), ('classifier', classifier)])
		model.fit(X_train.astype(np.float32), y_train)
		scr = model.score(X_test,y_test) 
		print('r2 score: {:.3f}'.format(scr))
		log_ratio_estimator = lambda X_new: logit(model.predict_proba(X_new)[:,model.classes_==1])
		print('...done')
		self.classifier = model
		return log_ratio_estimator

	def set_obs(self, obs):
		self.obs = obs[None,:] if obs.ndim==1 else obs

	def set_simulator(self, simulator):
		self.simulator = simulator

	def learn_logL_with_classifier(self, classifier, mins, maxs, Nsamples=None, test_size=0.1):
		assert len(mins) == len(maxs)
		Ndim = len(mins)
		if Nsamples is None: Nsamples = 10**Ndim
		theta0 = mins+(maxs-mins) * np.random.uniform(0,1,size=(Nsamples, len(mins)))
		theta1 = mins+(maxs-mins) * np.random.uniform(0,1,size=(Nsamples, len(mins)))
		try:
			out1 = self.simulator(theta1)
		except:
			print('Set the simulator function.')
			return None 
		X1 = np.hstack((theta1,out1)); y1 = np.ones(X1.shape[0])[:,None]
		X0 = np.hstack((theta0,out1)); y0 = np.zeros(X0.shape[0])[:,None]
		X  = np.vstack((X0,X1))
		y  = np.vstack((y0,y1))
		log_ratio_estimator = self.ratio_estimation(classifier, X, y, test_size=test_size)
		self.log_ratio_estimator = log_ratio_estimator
		self.logL_estimator = lambda theta: log_ratio_estimator(np.hstack((self.obs,theta[None,:] if theta.ndim==1 else theta)).astype(np.float32) ).squeeze()




