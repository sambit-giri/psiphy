"""
Kernel Density Estimation
-------------------------
Written around the KDE code in scikit-learn package.

"""
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import cross_val_score

def bandwidth_kde_silverman(X):
	"""
	h_ii = (4/(d+2))**(1/(d+4)) n**(-1/(d+4)) sigma_i 
	"""
	n, d = X.shape
	h_ii = (4/(d+2))**(1/(d+4))*n**(-1/(d+4))*X.std(axis=0)
	return h_ii

def bandwidth_kde_scott(X):
	"""
	h_ii = n**(-1/(d+4)) sigma_i 
	"""
	n, d = X.shape
	h_ii = n**(-1/(d+4))*X.std(axis=0)
	return h_ii


def bandwidth_kdeCV(X, kde=None, bw=10**np.linspace(-2,1), cv=1, verbose=True, kernel='gaussian', metric='euclidean', atol=0, leaf_size=40):
	"""
	Estimate the bandwidth using cross validation.
	"""
	if kde is None: kde = KernelDensity(kernel=kernel, metric=metric, atol=atol, leaf_size=leaf_size)

	Jh = np.array([])
	for i,h in enumerate(bw):
		kde.bandwidth = h
		kde.fit(X)
		log_pdf = kde.score_samples(X)
		pdf     = np.exp(log_pdf)
		jh_ = cross_val_loss_kdeCV_loo(X, kde, fx=pdf) if cv==1 else cross_val_loss_kdeCV_kFold(X, kde, fx=pdf, n_splits=cv)
		Jh  = np.append(Jh, jh_)
		if verbose: print('Completed: {0:.2f} %'.format(100*(i+1)/bw.size))

	return bw[Jh.argmin()]


def cross_val_loss_kdeCV_loo(X, kde, fx=None, verbose=True):
	"""
	J(h) = \int \hat{f}^2_n(x)dx - 2\int \hat{f}(x)f(x)dx
	\hat{J}(h) = \int \hat{f}^2_n(x)dx - \frac{2}{n}\sum \hat{f}_{-i}(x_i)
	"""
	from sklearn.model_selection import LeaveOneOut

	if fx is None:
		kde.fit(X)
		fx = np.exp(kde.score_samples(X))

	loo = LeaveOneOut()

	if verbose: print('Leave One Out CV| splits: {0:d}'.format(X.shape[0]))

	fx_loo_sum = np.array([])

	for train_index, test_index in loo.split(X):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		kde.fit(X_train)
		fx_loo = np.exp(kde.score_samples(X_test))
		fx_loo_sum = np.append(fx_loo_sum, fx_loo)

	return np.sum(fx**2)-np.sum(fx_loo_sum)*2/fx_loo_sum.size


def cross_val_loss_kdeCV_kFold(X, kde, fx=None, n_splits=5, verbose=True):
	"""
	J(h) = \int \hat{f}^2_n(x)dx - 2\int \hat{f}(x)f(x)dx
	\hat{J}(h) = \int \hat{f}^2_n(x)dx - \frac{2}{n}\sum \hat{f}_{-i}(x_i)
	"""
	from sklearn.model_selection import KFold

	if fx is None:
		kde.fit(X)
		fx = np.exp(kde.score_samples(X))

	kf = KFold(n_splits=n_splits)
	if verbose: print('{0:d}-fold CV| splits: {1:d}'.format(n_splits,kf.get_n_splits(X)))

	fx_kfold_sum = np.array([])
	for train_index, test_index in kf.split(X):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		kde.fit(X_train)
		fx_kfold = np.exp(kde.score_samples(X_test))
		fx_kfold_sum = np.append(fx_kfold_sum, fx_kfold)

	return np.sum(fx**2)-np.sum(fx_kfold_sum)*2/fx_kfold_sum.size



