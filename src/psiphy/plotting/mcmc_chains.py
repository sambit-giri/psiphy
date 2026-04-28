import matplotlib.pyplot as plt
import numpy as np

from getdist import plots, MCSamples
import getdist

from scipy import interpolate
from scipy.ndimage import gaussian_filter

def plot_dist_corner_getdist(samples, labels=None, names=None, filled=True, sample_labels=None):
	if isinstance(samples, (list,tuple)):
		if sample_labels is None: sample_labels = ['samples {0:d}'.format(i) for i in range(len(samples))]
		samps = [MCSamples(samples=samples0, names=names, labels=labels, label=str(slabels0)) for samples0, slabels0 in zip(samples,sample_labels)]
		g = plots.get_subplot_plotter()
		g.triangle_plot(samps, filled=filled)
	else:
		samps = MCSamples(samples=samples, names=names, labels=labels)
		g = plots.get_subplot_plotter()
		g.triangle_plot(samps, filled=filled)
	plt.show()
	return None


def walk_parameter(param, param_name=None, step_name=None, step_ticks=None, linestyle=None, linewidth=2, color=None):
	if param.ndim==1:
		if param_name is None: param_name = '\\theta'
		if step_name is None: step_name = ''
		if step_ticks is None: step_ticks = np.arange(param.size)
		else: assert step_ticks.size==param.size
		plt.plot(step_ticks, param, linestyle=linestyle, linewidth=linewidth, c=color)
		plt.xlabel(step_name)
		plt.ylabel(param_name)
	else:
		for s,pa in enumerate(param.T):
			plt.subplot(param.shape[1],1,s+1)
			pa_name = param_name[s] if len(param_name)==param.shape[1] else param_name
			walk_parameter(pa, param_name=pa_name, step_name=step_name, step_ticks=step_ticks, linestyle=linestyle, linewidth=linewidth, color=None)

def plot_corner_dist(samples, labels=None, flavor='hist', bins_1d=60, bins_2d=60, cmap=plt.cm.viridis, color='k', shading='gouraud', linestyle='-', linewidth=2, normed=True, verbose=False, CI=[68,95], CI_plotparam=None, smooth_dist=2.5):
	n_samples = samples.shape[1]
	if labels is None: labels = ['$\\theta_%d$'%i for i in range(n_samples)]
	else: assert len(labels)==n_samples
	fig, axes = plt.subplots(ncols=n_samples, nrows=n_samples, figsize=(10,8))
	fig.subplots_adjust(left=0.15, bottom=0.12, right=0.90, top=0.96, wspace=0.1, hspace=0.1)
	if np.array(bins_1d).size==1: bins_1d = [bins_1d for i in range(n_samples)]
	if np.array(bins_2d).size==1: bins_2d = [bins_2d for i in range(n_samples)]

	for i in range(n_samples):
		for j in range(n_samples):
			# print(i+1,j+1)
			if j>i: axes[i,j].set_visible(False)
			else:
				if i==j: 
					density_1D(samples[:,i], axes=axes[i,j], bins=bins_1d[i], linestyle=linestyle, linewidth=linewidth, normed=normed, smooth_dist=smooth_dist, color=color)
				else: 
					im = density_2D(samples[:,j], samples[:,i], CI=CI, axes=axes[i,j], flavor=flavor, nbins=bins_2d[i], cmap=cmap, shading=shading, CI_plotparam=CI_plotparam, smooth_dist=smooth_dist)
			if j==0: 
				if i!=0: 
					axes[i,j].set_ylabel(labels[i])
				else: 
					axes[i,j].set_yticks([])
			else: 
				axes[i,j].set_yticks([])
			if i==n_samples-1: 
				axes[i,j].set_xlabel(labels[j])
			else: 
				axes[i,j].set_xticks([])
	#fig.set_size_inches(18.5, 10.5)
	fig.subplots_adjust(right=0.9)
	cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
	fig.colorbar(im, cax=cbar_ax)
	### Estimate the CI
	if verbose:
		print_CI_samples(samples, bins_1d=bins_1d, CI=CI, labels=labels, smooth_dist=smooth_dist)

def plot_corner_multiple_dist(multi_samples, labels=None, flavor='hist', bins_1d=60, bins_2d=60, shading='gouraud', linestyle='-', linewidth=2, normed=True, verbose=False, CI=[68,95], smooth_dist=2.5):
	assert isinstance(multi_samples, (list,tuple))

	cmap_options = ['Blues', 'Oranges', 'Greys', 'Purples', 'Greens', 'Reds']
	colr_options = ['blue', 'orange', 'grey', 'purple', 'green', 'red']

	CI_plotparam_options = {'cmap': 'Reds', 'colors': None, 'alpha': 1}

	n_samples = multi_samples[0].shape[1]
	if labels is None: labels = ['$\\theta_%d$'%i for i in range(n_samples)]
	else: assert len(labels)==n_samples

	fig, axes = plt.subplots(ncols=n_samples, nrows=n_samples, figsize=(10,8))
	fig.subplots_adjust(left=0.15, bottom=0.12, right=0.90, top=0.96, wspace=0.1, hspace=0.1)

	if np.array(bins_1d).size==1: bins_1d = [bins_1d for i in range(n_samples)]
	if np.array(bins_2d).size==1: bins_2d = [bins_2d for i in range(n_samples)]

	for ss, samples in enumerate(multi_samples):
		print('Distribution:', ss)
		cmap, color = cmap_options[ss], colr_options[ss]
		CI_plotparam = {'cmap': None, 'colors': color, 'alpha': 1}

		for i in range(n_samples):
			for j in range(n_samples):
				# print(i+1,j+1)
				if j>i: axes[i,j].set_visible(False)
				else:
					if i==j: 
						density_1D(samples[:,i], axes=axes[i,j], bins=bins_1d[i], linestyle=linestyle, linewidth=linewidth, normed=normed, smooth_dist=smooth_dist, color=color)
					else: 
						im = contour_2D(samples[:,j], samples[:,i], CI=CI, axes=axes[i,j], flavor=flavor, nbins=bins_2d[i], cmap=cmap, shading=shading, CI_plotparam=CI_plotparam, smooth_dist=smooth_dist)
				if j==0: 
					if i!=0: 
						axes[i,j].set_ylabel(labels[i])
					else: 
						axes[i,j].set_yticks([])
				else: 
					axes[i,j].set_yticks([])
				if i==n_samples-1: 
					axes[i,j].set_xlabel(labels[j])
				else: 
					axes[i,j].set_xticks([])
		# fig.set_size_inches(18.5, 10.5)
		fig.subplots_adjust(right=0.9)
		# cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
		# fig.colorbar(im, cax=cbar_ax)
		### Estimate the CI
		if verbose:
			print_CI_samples(samples, bins_1d=bins_1d, CI=CI, labels=labels, smooth_dist=smooth_dist)

def print_CI_samples(samples, bins_1d=60, CI=[68,95], labels=None, smooth_dist=2.5):
	if labels is None: labels = ['$\\theta_%d$'%i for i in range(n_samples)]
	n_samples = samples.shape[1]
	if np.array(bins_1d).size==1: bins_1d = [bins_1d for i in range(n_samples)]
	print('Here are the credible intervals for each parameter')
	for ii in range(n_samples):
		print('Parameter ', labels[ii])
		print('------------------')
		for cc in CI:
			bla = get_CI_HDR_1D(samples[:,ii], percent=cc, bins=bins_1d[ii], smooth_dist=smooth_dist)
			print('%d percent:'%cc)
			for bb in zip(bla[0],bla[1]):
				print('['+str(bb[0])+', '+str(bb[1])+']')

	
def density_1D(x, axes=None, bins=60, linestyle='-', linewidth=2, color=None, show_mean=False, show_std=False, show_title=False, normed=True, smooth_dist=2.5):
	ht = np.histogram(x, bins=bins)
	if axes is None: 
		axes = plt
		axes_plt = True
	else: axes_plt = False
	xax, yax = ht[1][1:]/2+ht[1][:-1]/2., gaussian_filter(1.*ht[0]/ht[0].max(), sigma=smooth_dist)
	if normed: 
		axes.plot(xax, yax, linestyle=linestyle, linewidth=linewidth, c=color)
		if axes_plt: axes.ylim(0,1)
		else: axes.set_ylim(0,1)
	else: 
		axes.plot(xax, yax, linestyle=linestyle, linewidth=linewidth, c=color)
		if axes_plt: axes.ylim(0,1)
		else: axes.set_ylim(0,1)
	if show_mean: axes.plot(x.mean()*np.ones(10), np.linspace(0,ht[0].max(),10), '--', c='k')
	if show_std: 
		axes.plot(x.mean()*np.ones(10)-x.std(), np.linspace(0,ht[0].max(),10), ':', c='k', linewidth=linewidth)
		axes.plot(x.mean()*np.ones(10)+x.std(), np.linspace(0,ht[0].max(),10), ':', c='k', linewidth=linewidth)
	if show_title: axes.set_title('%.2f$_\mathrm{-%.2f}^\mathrm{+%.2f}$'%(x.mean(),x.std(),x.std()))
	print(x.mean(),x.std())

def contour_2D(x, y, CI=[68,95], axes=plt, flavor='hex', nbins='scott', cmap=plt.cm.BuGn_r, shading='gouraud', CI_plotparam=None, smooth_dist=2.5):
	if CI_plotparam is None: CI_plotparam={'cmap': 'Reds', 'colors': None, 'alpha': 1}
	if flavor.lower()=='scatter':
		# Everything sarts with a Scatterplot
		#axes.set_title('Scatterplot')
		im = axes.plot(x, y, 'ko')
	# As you can see there is a lot of overplottin here!
	elif flavor.lower()=='hex':
		# Thus we can cut the plotting window in several hexbins
		#axes.set_title('Hexbin')
		im = axes.hexbin(x, y, gridsize=nbins, cmap=cmap)
	elif flavor.lower()=='hist':
		# 2D Histogram
		#axes.set_title('2D Histogram')
		ht = np.histogram2d(x, y, bins=nbins, density=True)
		xx, yy = ht[1][1:]/2+ht[1][:-1]/2., ht[2][1:]/2+ht[2][:-1]/2.
		f = interpolate.interp2d(xx, yy, ht[0].T, kind='cubic')
		#f = interpolate.RectBivariateSpline(xx, yy, ht[0].T, kx=3, ky=3)
		#xi, yi = xi[int(xbins*0.05):-int(xbins*0.05),int(ybins*0.05):-int(ybins*0.05)], yi[int(xbins*0.05):-int(xbins*0.05),int(ybins*0.05):-int(ybins*0.05)]
		#xx, yy = xx[1::2], yy[1::2]		
		xi, yi = np.meshgrid(xx, yy)
		zi = ht[0].T#f(xx.flatten(),yy.flatten())
		zi = gaussian_filter(1.*zi, sigma=smooth_dist); zi = (zi-zi.min())/(zi.max()-zi.min())
		# im = axes.pcolormesh(xi, yi, zi, cmap=cmap)
		#im = axes.pcolormesh(xi, yi, 1.*zi.reshape(xi.shape)/zi.max(), cmap=cmap, shading=shading)
		#axes.contour(xi, yi, zi, cmap='viridis', levels=[zi.max()-2*zi.std(),zi.max()-zi.std()], linestyles='--')
		lstyles = ['-', '--', ':', '-.'] 
		levels_CI = np.array([get_CI_HDR_2D(x, y, percent=CI[idx]) for idx in range(len(CI))])
		lstyle_CI = np.array([lstyles[idx] for idx in range(len(CI))])
		levelsarg = np.argsort(levels_CI)
		#axes.contour(xi, yi, zi, levels=levels_CI[levelsarg], linestyles=lstyle_CI[levelsarg], cmap=CI_plotparam['cmap'], colors=CI_plotparam['colors'], alpha=CI_plotparam['alpha']) 
		levelsrt = levels_CI[levelsarg]
		alphas   = np.linspace(0.9, 0.3, len(levelsrt))
		for li in range(len(levelsrt)):
			lev = [levelsrt[li], levelsrt[li+1]] if li<len(levelsrt)-1 else [levelsrt[li], 1]
			axes.contourf(xi, yi, zi, levels=lev, cmap=CI_plotparam['cmap'], colors=CI_plotparam['colors'], alpha=alphas[-li-1]) 
		#axes.contourf(xi, yi, zi, levels=[levels_CI[levelsarg]], cmap=CI_plotparam['cmap'], colors=CI_plotparam['colors'], alpha=CI_plotparam['alpha']) 
	elif flavor.lower() in ['kde', 'shading', 'contour']:
		from scipy.stats import kde
		#from sklearn.neighbors.kde import KernelDensity
		# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
		data = np.vstack([x, y]).T
		#k = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
		k = kde.gaussian_kde(data.T, bw_method='scott')
		if np.array(nbins).size==2: xbins, ybins = nbins
		else: xbins, ybins = nbins, nbins
		xi, yi = np.mgrid[x.min():x.max():xbins*1j, y.min():y.max():ybins*1j]
		xi, yi = xi[int(xbins*0.05):-int(xbins*0.05),int(ybins*0.05):-int(ybins*0.05)], yi[int(xbins*0.05):-int(xbins*0.05),int(ybins*0.05):-int(ybins*0.05)]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))
		if flavor.lower()=='kde':
			# plot a density
			#axes.set_title('Calculate Gaussian KDE')
			im = axes.pcolormesh(xi, yi, zi.reshape(xi.shape)/zi.max(), cmap=cmap)
		elif flavor.lower()=='shading':
			# add shading
			#axes.set_title('2D Density with shading')
			im = axes.pcolormesh(xi, yi, zi.reshape(xi.shape)/zi.max(), shading=shading, cmap=cmap)
		elif flavor.lower()=='contour':
			# contour
			#axes.set_title('Contour')
			im = axes.pcolormesh(xi, yi, zi.reshape(xi.shape)/zi.max(), shading=shading, cmap=cmap)
			axes.contour(xi, yi, zi.reshape(xi.shape)/zi.max())
	elif flavor=='flat':
		if np.array(nbins).size==2: xbins, ybins = nbins
		else: xbins, ybins = nbins, nbins
		xi, yi = np.mgrid[x.min():x.max():xbins*1j, y.min():y.max():ybins*1j]
		xx, yy = np.linspace(x.min(),x.max(),xbins), np.linspace(y.min(),y.max(),ybins)
		from scipy.interpolate import interp2d
		f = interp2d(x, y, np.ones(x.size), kind='linear')
		zi = f(xx, yy)
		zz = np.zeros(zi.shape)
		zz[zi>0.5] = 1
		im = axes.pcolormesh(xi, yi, zz, shading=shading, cmap=cmap)
	return None


def density_2D(x, y, CI=[68,95], axes=plt, flavor='hex', nbins='scott', cmap=plt.cm.BuGn_r, shading='gouraud', CI_plotparam=None, smooth_dist=2.5):
	if CI_plotparam is None: CI_plotparam={'cmap': 'Reds', 'colors': None, 'alpha': 1}
	if flavor.lower()=='scatter':
		# Everything sarts with a Scatterplot
		#axes.set_title('Scatterplot')
		im = axes.plot(x, y, 'ko')
	# As you can see there is a lot of overplottin here!
	elif flavor.lower()=='hex':
		# Thus we can cut the plotting window in several hexbins
		#axes.set_title('Hexbin')
		im = axes.hexbin(x, y, gridsize=nbins, cmap=cmap)
	elif flavor.lower()=='hist':
		# 2D Histogram
		#axes.set_title('2D Histogram')
		ht = np.histogram2d(x, y, bins=nbins, density=True)
		xx, yy = ht[1][1:]/2+ht[1][:-1]/2., ht[2][1:]/2+ht[2][:-1]/2.
		f = interpolate.interp2d(xx, yy, ht[0].T, kind='cubic')
		#f = interpolate.RectBivariateSpline(xx, yy, ht[0].T, kx=3, ky=3)
		#xi, yi = xi[int(xbins*0.05):-int(xbins*0.05),int(ybins*0.05):-int(ybins*0.05)], yi[int(xbins*0.05):-int(xbins*0.05),int(ybins*0.05):-int(ybins*0.05)]
		#xx, yy = xx[1::2], yy[1::2]		
		xi, yi = np.meshgrid(xx, yy)
		zi = ht[0].T#f(xx.flatten(),yy.flatten())
		zi = gaussian_filter(1.*zi, sigma=smooth_dist); zi = (zi-zi.min())/(zi.max()-zi.min())
		im = axes.pcolormesh(xi, yi, zi, cmap=cmap)
		#im = axes.pcolormesh(xi, yi, 1.*zi.reshape(xi.shape)/zi.max(), cmap=cmap, shading=shading)
		#axes.contour(xi, yi, zi, cmap='viridis', levels=[zi.max()-2*zi.std(),zi.max()-zi.std()], linestyles='--')
		lstyles = ['-', '--', ':', '-.'] 
		levels_CI = np.array([get_CI_HDR_2D(x, y, percent=CI[idx]) for idx in range(len(CI))])
		lstyle_CI = np.array([lstyles[idx] for idx in range(len(CI))])
		levelsarg = np.argsort(levels_CI)
		axes.contour(xi, yi, zi, levels=levels_CI[levelsarg], linestyles=lstyle_CI[levelsarg], cmap=CI_plotparam['cmap'], colors=CI_plotparam['colors'], alpha=CI_plotparam['alpha']) 
	elif flavor.lower() in ['kde', 'shading', 'contour']:
		from scipy.stats import kde
		#from sklearn.neighbors.kde import KernelDensity
		# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
		data = np.vstack([x, y]).T
		#k = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
		k = kde.gaussian_kde(data.T, bw_method='scott')
		if np.array(nbins).size==2: xbins, ybins = nbins
		else: xbins, ybins = nbins, nbins
		xi, yi = np.mgrid[x.min():x.max():xbins*1j, y.min():y.max():ybins*1j]
		xi, yi = xi[int(xbins*0.05):-int(xbins*0.05),int(ybins*0.05):-int(ybins*0.05)], yi[int(xbins*0.05):-int(xbins*0.05),int(ybins*0.05):-int(ybins*0.05)]
		zi = k(np.vstack([xi.flatten(), yi.flatten()]))
		if flavor.lower()=='kde':
			# plot a density
			#axes.set_title('Calculate Gaussian KDE')
			im = axes.pcolormesh(xi, yi, zi.reshape(xi.shape)/zi.max(), cmap=cmap)
		elif flavor.lower()=='shading':
			# add shading
			#axes.set_title('2D Density with shading')
			im = axes.pcolormesh(xi, yi, zi.reshape(xi.shape)/zi.max(), shading=shading, cmap=cmap)
		elif flavor.lower()=='contour':
			# contour
			#axes.set_title('Contour')
			im = axes.pcolormesh(xi, yi, zi.reshape(xi.shape)/zi.max(), shading=shading, cmap=cmap)
			axes.contour(xi, yi, zi.reshape(xi.shape)/zi.max())
	elif flavor=='flat':
		if np.array(nbins).size==2: xbins, ybins = nbins
		else: xbins, ybins = nbins, nbins
		xi, yi = np.mgrid[x.min():x.max():xbins*1j, y.min():y.max():ybins*1j]
		xx, yy = np.linspace(x.min(),x.max(),xbins), np.linspace(y.min(),y.max(),ybins)
		from scipy.interpolate import interp2d
		f = interp2d(x, y, np.ones(x.size), kind='linear')
		zi = f(xx, yy)
		zz = np.zeros(zi.shape)
		zz[zi>0.5] = 1
		im = axes.pcolormesh(xi, yi, zz, shading=shading, cmap=cmap)
	return im
		
def plot_2d_power(ps, xticks, yticks, axes):
	xticks, yticks = np.round(xticks, decimals=2), np.round(yticks, decimals=2)
	im = axes.imshow(ps, origin='lower')
	locs, labels = axes.get_yticks()
	new_labels = yticks[locs.astype(int)[1:-1]]
	axes.set_yticks(locs[1:-1], new_labels)
	#axes.set_ylabel(ylabel)
	locs, labels = plt.xticks()
	new_labels = xticks[locs.astype(int)[1:-1]]
	axes.set_xticks(locs[1:-1], new_labels)
	#axes.set_xlabel(xlabel)
	#plt.colorbar()
	#plt.show()
	return im

def get_CI_HDR_2D(x, y, percent=95., nbins=60, smooth_dist=2.5):
	"""
	Hyndman (1996)
	"""
	ht = np.histogram2d(x, y, bins=nbins, density=True)
	xx, yy = ht[1][1:]/2+ht[1][:-1]/2., ht[2][1:]/2+ht[2][:-1]/2.
	#f = interpolate.interp2d(xx, yy, ht[0].T, kind='cubic')		
	xi, yi = np.meshgrid(xx, yy)
	zi = ht[0].T
	zi = gaussian_filter(1.*zi, sigma=smooth_dist); zi = (zi-zi.min())/(zi.max()-zi.min())
	p_sorted = np.sort(zi.flatten())
	i, pp    = 0, 1.
	while(pp>percent/100.):
		pp = p_sorted[p_sorted>=p_sorted[i]].sum()/p_sorted.sum()
		#print(i,pp)
		i = i+1
	pci =  p_sorted[i-1]
	return pci


def get_CI_HDR_1D(x, percent=95., bins=60, smooth_dist=2.5):
	"""
	Hyndman (1996)
	"""
	ht = np.histogram(x, bins=bins)
	xax, yax = ht[1][1:]/2+ht[1][:-1]/2., gaussian_filter(1.*ht[0]/ht[0].max(), sigma=smooth_dist)
	p_sorted = np.sort(yax)
	i, pp    = 0, 1.
	while(pp>percent/100.):
		pp = p_sorted[p_sorted>=p_sorted[i]].sum()/p_sorted.sum()
		#print(i,pp)
		i = i+1
	pci =  p_sorted[i-1]
	intervals = find_intervals(xax, yax, pci)
	return intervals[0], intervals[1], pci

def find_intervals(xax, yax, y0):
	min_interval = []
	max_interval = []
	greater  = yax[0]>y0
	if greater: min_interval.append('Prior minima')
	for xx,yy in zip(xax, yax):
		gg = yy>y0
		if int(gg) + int(greater) == 1:
			if gg: min_interval.append(xx)
			else: max_interval.append(xx)
			#print(greater, gg)
			greater = gg
	if greater: max_interval.append('Prior maxima')
	return min_interval, max_interval
			



"""
def get_CI_peak(xax, yax, peakind, percent=95.):
	dxax  = (xax[1:]-xax[:-1])[0]
	integ = yax[peakind]*dxax
	ii    = 0
	while (integ<percent and ii<xax.size):
		if peakind+ii<xax.size: integ = integ + yax[peakind+ii]*dxax
		if peakind+ii>=0: integ = integ + yax[peakind-ii]*dxax
		ii = ii+1


@article{doi:10.1080/00031305.1996.10474359,
author = { Rob J.   Hyndman },
title = {Computing and Graphing Highest Density Regions},
journal = {The American Statistician},
volume = {50},
number = {2},
pages = {120-126},
year  = {1996},
publisher = {Taylor & Francis},
doi = {10.1080/00031305.1996.10474359}
}


"""

