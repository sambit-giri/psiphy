import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian


# ── LFIRE grid-posterior plots ───────────────────────────────────────────────

def credible_limit(zi, level, method='naive'):
    if method == 'naive':
        zbins = np.linspace(zi.min(), zi.max(), 2000)
        sm = 0
        for i, zb in enumerate(zbins):
            sm = np.sum(zi[zi < zb]) / np.sum(zi)
            if sm > (1 - level / 100):
                break
    return zb


def plot_lfire(lfi, smooth=5, true_values=None, CI=[95], cmap='Blues', CI_param=None, figsize=(10, 8)):
    if np.ndim(lfi.thetas) == 1:
        fig, axes = plt.subplots(nrows=1, ncols=1)
        xx, cube = lfi.thetas, lfi.posterior
        if smooth:
            cube = gaussian(cube, smooth)
        axes.plot(xx, cube)
        axes.set_xlabel(lfi.param_names[0])
        plt.show()
        return None
    else:
        N = lfi.thetas.shape[1]
        fig, axes = plt.subplots(nrows=N, ncols=N, figsize=figsize)
        for i in range(N):
            for j in range(N):
                if j > i:
                    axes[i, j].axis('off')
                elif i == j:
                    plot_1Dmarginal_lfire(lfi, i, ax=axes[i, j], smooth=smooth, true_values=true_values)
                    if i + 1 < N:
                        axes[i, j].set_xlabel('')
                        axes[i, j].set_xticks([])
                    if j > 0:
                        axes[i, j].set_yticks([])
                else:
                    im = plot_2Dmarginal_lfire(lfi, i, j, ax=axes[i, j], smooth=smooth,
                                               true_values=true_values, CI=CI, cmap=cmap, CI_param=CI_param)
                    if i + 1 < N:
                        axes[i, j].set_xlabel('')
                        axes[i, j].set_xticks([])
                    if j > 0:
                        axes[i, j].set_ylabel('')
                        axes[i, j].set_yticks([])

    fig.subplots_adjust(right=0.88)
    cb_ax = fig.add_axes([0.89, 0.1, 0.02, 0.8])
    fig.colorbar(im, cax=cb_ax)
    plt.show()


def plot_1Dmarginal(thetas, posterior, param_names=None, idx=0, ax=None, bins=100,
                    verbose=False, smooth=False, true_values=None):
    N = thetas.shape[1]
    inds = np.arange(N)
    inds = np.delete(inds, idx)
    X = np.array([thetas[:, i] for i in inds])
    X = np.vstack((thetas[:, idx].reshape(1, -1), X)).T
    y = posterior
    dm = [int(np.round(y.shape[0] ** (1 / X.shape[1]))) for i in range(X.shape[1])]
    cube = y.reshape(dm)
    if idx != 0:
        cube = np.swapaxes(cube, 0, idx)
    while cube.ndim > 1:
        if verbose:
            print('Reducing dimension: {0:} -> {1:}'.format(cube.ndim, cube.ndim - 1))
        cube = cube.sum(axis=-1)
    xx = np.unique(X[:, 0])
    if smooth:
        cube = gaussian(cube, smooth)
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(xx, (cube - cube.min()) / (cube.max() - cube.min()))
    if param_names is not None:
        ax.set_xlabel(param_names[idx])


def plot_2Dmarginal(thetas, posterior, param_names=None, idx=0, idy=1, ax=None, bins=100,
                    verbose=False, smooth=False, true_values=None, CI=[95]):
    N = thetas.shape[1]
    inds = np.arange(N)
    inds = np.delete(inds, max([idx, idy]))
    inds = np.delete(inds, min([idx, idy]))
    X = np.array([thetas[:, i] for i in inds])
    if X.size == 0:
        X = np.vstack((thetas[:, idx].reshape(1, -1), thetas[:, idy].reshape(1, -1))).T
    else:
        X = np.vstack((thetas[:, idx].reshape(1, -1), thetas[:, idy].reshape(1, -1), X)).T
    y = posterior
    dm = [int(np.round(y.shape[0] ** (1 / X.shape[1]))) for i in range(X.shape[1])]
    cube = y.reshape(dm)
    cube = np.swapaxes(cube, 0, idx)
    cube = np.swapaxes(cube, 1, idy)
    while cube.ndim > 2:
        if verbose:
            print('Reducing dimension: {0:} -> {1:}'.format(cube.ndim, cube.ndim - 1))
        cube = cube.sum(axis=-1)
    yy = np.unique(X[:, 0])
    xx = np.unique(X[:, 1])
    xi, yi = np.meshgrid(xx, yy)
    if smooth:
        cube = gaussian(cube, smooth)
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    zi = (cube - cube.min()) / (cube.max() - cube.min())
    im = ax.pcolormesh(xi, yi, zi, cmap='Blues')
    if true_values is not None:
        ax.scatter(true_values[param_names[idy]], true_values[param_names[idx]], marker='*', c='r')
    if CI is not None:
        for cc in CI:
            ll = credible_limit(zi, cc, method='naive')
            ax.contour(xi, yi, zi, levels=[ll], linewidths=0.5, colors='k')
    if param_names is not None:
        ax.set_xlabel(param_names[idy])
        ax.set_ylabel(param_names[idx])
    return im


def plot_1Dmarginal_lfire(lfi, idx, ax=None, bins=100, verbose=False, smooth=False, true_values=None):
    N = lfi.thetas.shape[1]
    thetas = lfi.thetas
    inds = np.arange(N)
    inds = np.delete(inds, idx)
    X = np.array([thetas[:, i] for i in inds])
    X = np.vstack((thetas[:, idx].reshape(1, -1), X)).T
    y = lfi.posterior
    dm = [int(np.round(y.shape[0] ** (1 / X.shape[1]))) for i in range(X.shape[1])]
    cube = y.reshape(dm)
    if idx != 0:
        cube = np.swapaxes(cube, 0, idx)
    while cube.ndim > 1:
        if verbose:
            print('Reducing dimension: {0:} -> {1:}'.format(cube.ndim, cube.ndim - 1))
        cube = cube.sum(axis=-1)
    xx = np.unique(X[:, 0])
    if smooth:
        cube = gaussian(cube, smooth)
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(xx, (cube - cube.min()) / (cube.max() - cube.min()))
    ax.set_xlabel(lfi.param_names[idx])


def plot_2Dmarginal_lfire(lfi, idx, idy, ax=None, bins=100, verbose=False, smooth=False,
                          true_values=None, CI=[95], cmap='Blues', CI_param=None):
    if CI_param is None:
        colr_options = ['blue', 'orange', 'grey', 'purple', 'green', 'red']
        colr_options.reverse()
        CI_param = {
            'linestyle': ['-', '--', '-.', ':', '-.'],
            'color': colr_options,
            'linewidths': 2,
        }
    while len(CI_param['linestyle']) < len(CI):
        CI_param['linestyle'].append(CI_param['linestyle'][-1])
    while len(CI_param['color']) < len(CI):
        CI_param['color'].append(CI_param['color'][-1])

    N = lfi.thetas.shape[1]
    thetas = lfi.thetas
    inds = np.arange(N)
    inds = np.delete(inds, max([idx, idy]))
    inds = np.delete(inds, min([idx, idy]))
    X = np.array([thetas[:, i] for i in inds])
    if X.size == 0:
        X = np.vstack((thetas[:, idx].reshape(1, -1), thetas[:, idy].reshape(1, -1))).T
    else:
        X = np.vstack((thetas[:, idx].reshape(1, -1), thetas[:, idy].reshape(1, -1), X)).T
    y = lfi.posterior
    dm = [int(np.round(y.shape[0] ** (1 / X.shape[1]))) for i in range(X.shape[1])]
    cube = y.reshape(dm)
    cube = np.swapaxes(cube, 0, idx)
    cube = np.swapaxes(cube, 1, idy)
    while cube.ndim > 2:
        if verbose:
            print('Reducing dimension: {0:} -> {1:}'.format(cube.ndim, cube.ndim - 1))
        cube = cube.sum(axis=-1)
    yy = np.unique(X[:, 0])
    xx = np.unique(X[:, 1])
    xi, yi = np.meshgrid(xx, yy)
    if smooth:
        cube = gaussian(cube, smooth)
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    zi = (cube - cube.min()) / (cube.max() - cube.min())
    im = ax.pcolormesh(xi, yi, zi, cmap=cmap)
    if true_values is not None:
        ax.scatter(true_values[lfi.param_names[idy]], true_values[lfi.param_names[idx]], marker='*', c='r')
    if CI is not None:
        for ci, cc in enumerate(CI):
            ll = credible_limit(zi, cc, method='naive')
            ax.contour(xi, yi, zi, levels=[ll], linewidths=CI_param['linewidths'],
                       colors=CI_param['color'][ci], linestyles=CI_param['linestyle'][ci])
    ax.set_xlabel(lfi.param_names[idy])
    ax.set_ylabel(lfi.param_names[idx])
    return im


# ── Sample-based corner plots with HDR credible intervals ────────────────────

def walk_parameter(param, param_name=None, step_name=None, step_ticks=None,
                   linestyle=None, linewidth=2, color=None):
    if param.ndim == 1:
        if param_name is None:
            param_name = r'\theta'
        if step_name is None:
            step_name = ''
        if step_ticks is None:
            step_ticks = np.arange(param.size)
        else:
            assert step_ticks.size == param.size
        plt.plot(step_ticks, param, linestyle=linestyle, linewidth=linewidth, c=color)
        plt.xlabel(step_name)
        plt.ylabel(param_name)
    else:
        for s, pa in enumerate(param.T):
            plt.subplot(param.shape[1], 1, s + 1)
            pa_name = param_name[s] if len(param_name) == param.shape[1] else param_name
            walk_parameter(pa, param_name=pa_name, step_name=step_name,
                           step_ticks=step_ticks, linestyle=linestyle,
                           linewidth=linewidth, color=None)


def corner_density(samples, labels=None, flavor='hist', bins_1d=60, bins_2d=60,
                   cmap=plt.cm.viridis, shading='gouraud', linestyle='-', linewidth=2,
                   normed=True, CI=[68, 95]):
    n_samples = samples.shape[1]
    if labels is None:
        labels = [r'$\theta_%d$' % i for i in range(n_samples)]
    else:
        assert len(labels) == n_samples
    fig, axes = plt.subplots(ncols=n_samples, nrows=n_samples, figsize=(10, 8))
    fig.subplots_adjust(left=0.15, bottom=0.12, right=0.90, top=0.96, wspace=0.1, hspace=0.1)
    if np.array(bins_1d).size == 1:
        bins_1d = [bins_1d for _ in range(n_samples)]
    if np.array(bins_2d).size == 1:
        bins_2d = [bins_2d for _ in range(n_samples)]

    for i in range(n_samples):
        for j in range(n_samples):
            if j > i:
                axes[i, j].set_visible(False)
            else:
                if i == j:
                    density_1D(samples[:, i], axes=axes[i, j], bins=bins_1d[i],
                               linestyle=linestyle, linewidth=linewidth, normed=normed)
                else:
                    im = density_2D(samples[:, j], samples[:, i], CI=CI, axes=axes[i, j],
                                    flavor=flavor, nbins=bins_2d[i], cmap=cmap, shading=shading)
            if j == 0:
                if i != 0:
                    axes[i, j].set_ylabel(labels[i])
                else:
                    axes[i, j].set_yticks([])
            else:
                axes[i, j].set_yticks([])
            if i == n_samples - 1:
                axes[i, j].set_xlabel(labels[j])
            else:
                axes[i, j].set_xticks([])
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    print_CI_samples(samples, bins_1d=bins_1d, CI=CI)


def print_CI_samples(samples, bins_1d=60, CI=[68, 95]):
    n_samples = samples.shape[1]
    if np.array(bins_1d).size == 1:
        bins_1d = [bins_1d for _ in range(n_samples)]
    print('Credible intervals for each parameter')
    for ii in range(n_samples):
        print('Parameter ' + str(ii + 1))
        print('------------------')
        for cc in CI:
            bla = get_CI_HDR_1D(samples[:, ii], percent=cc, bins=bins_1d[ii])
            print('%d percent:' % cc)
            for bb in zip(bla[0], bla[1]):
                print('[' + str(bb[0]) + ', ' + str(bb[1]) + ']')


def density_1D(x, axes=None, bins=60, linestyle='-', linewidth=2, color=None,
               show_mean=False, show_std=False, show_title=False, normed=True):
    ht = np.histogram(x, bins=bins)
    if axes is None:
        axes = plt
        axes_plt = True
    else:
        axes_plt = False
    xax = ht[1][1:] / 2 + ht[1][:-1] / 2.
    yax = gaussian_filter(1. * ht[0] / ht[0].max(), sigma=3.)
    axes.plot(xax, yax, linestyle=linestyle, linewidth=linewidth, c=color)
    if axes_plt:
        axes.ylim(0, 1)
    else:
        axes.set_ylim(0, 1)
    if show_mean:
        axes.plot(x.mean() * np.ones(10), np.linspace(0, ht[0].max(), 10), '--', c='k')
    if show_std:
        axes.plot(x.mean() * np.ones(10) - x.std(), np.linspace(0, ht[0].max(), 10), ':', c='k', linewidth=linewidth)
        axes.plot(x.mean() * np.ones(10) + x.std(), np.linspace(0, ht[0].max(), 10), ':', c='k', linewidth=linewidth)
    if show_title:
        axes.set_title(r'%.2f$_\mathrm{-%.2f}^\mathrm{+%.2f}$' % (x.mean(), x.std(), x.std()))


def density_2D(x, y, CI=[68, 95], axes=plt, flavor='hex', nbins='scott',
               cmap=plt.cm.BuGn_r, shading='gouraud'):
    if flavor.lower() == 'scatter':
        im = axes.plot(x, y, 'ko')
    elif flavor.lower() == 'hex':
        im = axes.hexbin(x, y, gridsize=nbins, cmap=cmap)
    elif flavor.lower() == 'hist':
        ht = np.histogram2d(x, y, bins=nbins, density=True)
        xx = ht[1][1:] / 2 + ht[1][:-1] / 2.
        yy = ht[2][1:] / 2 + ht[2][:-1] / 2.
        xi, yi = np.meshgrid(xx, yy)
        zi = ht[0].T
        zi = gaussian_filter(1. * zi, sigma=2.5)
        zi = (zi - zi.min()) / (zi.max() - zi.min())
        im = axes.pcolormesh(xi, yi, zi, cmap=cmap)
        axes.contour(xi, yi, zi, cmap='Reds',
                     levels=get_CI_HDR_2D(x, y, percent=CI[0], nbins=nbins), linestyles='-')
        axes.contour(xi, yi, zi, cmap='Reds',
                     levels=get_CI_HDR_2D(x, y, percent=CI[1], nbins=nbins), linestyles='--')
        if len(CI) > 2:
            axes.contour(xi, yi, zi, cmap='Reds',
                         levels=get_CI_HDR_2D(x, y, percent=CI[2], nbins=nbins), linestyles=':')
    elif flavor.lower() in ['kde', 'shading', 'contour']:
        from scipy.stats import kde
        data = np.vstack([x, y]).T
        k = kde.gaussian_kde(data.T, bw_method='scott')
        if np.array(nbins).size == 2:
            xbins, ybins = nbins
        else:
            xbins = ybins = nbins
        xi, yi = np.mgrid[x.min():x.max():xbins * 1j, y.min():y.max():ybins * 1j]
        xi = xi[int(xbins * 0.05):-int(xbins * 0.05), int(ybins * 0.05):-int(ybins * 0.05)]
        yi = yi[int(xbins * 0.05):-int(xbins * 0.05), int(ybins * 0.05):-int(ybins * 0.05)]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        im = axes.pcolormesh(xi, yi, zi.reshape(xi.shape) / zi.max(), shading=shading, cmap=cmap)
        if flavor.lower() == 'contour':
            axes.contour(xi, yi, zi.reshape(xi.shape) / zi.max())
    elif flavor == 'flat':
        if np.array(nbins).size == 2:
            xbins, ybins = nbins
        else:
            xbins = ybins = nbins
        xi, yi = np.mgrid[x.min():x.max():xbins * 1j, y.min():y.max():ybins * 1j]
        xx = np.linspace(x.min(), x.max(), xbins)
        yy = np.linspace(y.min(), y.max(), ybins)
        from scipy.interpolate import interp2d
        f = interp2d(x, y, np.ones(x.size), kind='linear')
        zi = f(xx, yy)
        zz = np.zeros(zi.shape)
        zz[zi > 0.5] = 1
        im = axes.pcolormesh(xi, yi, zz, shading=shading, cmap=cmap)
    return im


def get_CI_HDR_2D(x, y, percent=95., nbins=60):
    """Hyndman (1996) highest density region for 2D."""
    ht = np.histogram2d(x, y, bins=nbins, density=True)
    xx = ht[1][1:] / 2 + ht[1][:-1] / 2.
    yy = ht[2][1:] / 2 + ht[2][:-1] / 2.
    zi = ht[0].T
    zi = gaussian_filter(1. * zi, sigma=2.5)
    zi = (zi - zi.min()) / (zi.max() - zi.min())
    p_sorted = np.sort(zi.flatten())
    i, pp = 0, 1.
    while pp > percent / 100.:
        pp = p_sorted[p_sorted >= p_sorted[i]].sum() / p_sorted.sum()
        i += 1
    return p_sorted[i - 1]


def get_CI_HDR_1D(x, percent=95., bins=60):
    """Hyndman (1996) highest density region for 1D."""
    ht = np.histogram(x, bins=bins)
    xax = ht[1][1:] / 2 + ht[1][:-1] / 2.
    yax = gaussian_filter(1. * ht[0] / ht[0].max(), sigma=3.)
    p_sorted = np.sort(yax)
    i, pp = 0, 1.
    while pp > percent / 100.:
        pp = p_sorted[p_sorted >= p_sorted[i]].sum() / p_sorted.sum()
        i += 1
    pci = p_sorted[i - 1]
    intervals = find_intervals(xax, yax, pci)
    return intervals[0], intervals[1], pci


def find_intervals(xax, yax, y0):
    min_interval = []
    max_interval = []
    greater = yax[0] > y0
    if greater:
        min_interval.append('Prior minima')
    for xx, yy in zip(xax, yax):
        gg = yy > y0
        if int(gg) + int(greater) == 1:
            if gg:
                min_interval.append(xx)
            else:
                max_interval.append(xx)
            greater = gg
    if greater:
        max_interval.append('Prior maxima')
    return min_interval, max_interval
