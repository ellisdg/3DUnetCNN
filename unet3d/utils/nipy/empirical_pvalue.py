# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Routines to get corrected p-values estimates, based on the observations.

It implements 3 approaches:

- Benjamini-Hochberg FDR: http://en.wikipedia.org/wiki/False_discovery_rate

- a class that fits a Gaussian model to the central part of an
  histogram, following [1]

  [1] Schwartzman A, Dougherty RF, Lee J, Ghahremani D, Taylor
  JE. Empirical null and false discovery rate analysis in
  neuroimaging. Neuroimage. 2009 Jan 1;44(1):71-82.  PubMed PMID:
  18547821. DOI: 10.1016/j.neuroimage.2008.04.182

  This is typically necessary to estimate a FDR when one is not
  certain that the data behaves as a standard normal under H_0.

- a model based on Gaussian mixture modelling 'a la Oxford'

Author : Bertrand Thirion, Yaroslav Halchenko, 2008-2012
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from numpy.linalg import pinv
import scipy.stats as st


def check_p_values(p_values):
    """Basic checks on the p_values array: values should be within [0,1]

    Assures also that p_values are at least in 1d array.  None of the
    checks is performed if p_values is None.

    Parameters
    ----------
    p_values : array of shape (n)
        The sample p-values

    Returns
    -------
    p_values : array of shape (n)
        The sample p-values
    """
    if p_values is None:
        return None
    # Take all elements unfolded and assure having at least 1d
    p_values = np.atleast_1d(np.ravel(p_values))
    if np.any(np.isnan(p_values)):
        raise ValueError("%d values are NaN" % (sum(np.isnan(p_values))))
    if p_values.min() < 0:
        raise ValueError("Negative p-values. Min=%g" % (p_values.min(),))
    if p_values.max() > 1:
        raise ValueError("P-values greater than 1! Max=%g" % (
                p_values.max(),))
    return p_values


def gaussian_fdr(x):
    """Return the FDR associated with each value assuming a Gaussian distribution
    """
    return fdr(st.norm.sf(np.squeeze(x)))


def gaussian_fdr_threshold(x, alpha=0.05):
    """Return FDR threshold given normal variates

    Given an array x of normal variates, this function returns the
    critical p-value associated with alpha.
    x is explicitly assumed to be normal distributed under H_0

    Parameters
    -----------
    x: ndarray
        input data
    alpha: float, optional
        desired significance

    Returns
    -------
    threshold : float
        threshold, given as a Gaussian critical value
    """
    pvals = st.norm.sf(x)
    pth = fdr_threshold(pvals, alpha)
    return st.norm.isf(pth)


def fdr_threshold(p_values, alpha=0.05):
    """Return FDR threshold given p values

    Parameters
    -----------
    p_values : array of shape (n), optional
        The samples p-value
    alpha : float, optional
        The desired FDR significance

    Returns
    -------
    critical_p_value: float
        The p value corresponding to the FDR alpha
    """
    p_values = check_p_values(p_values)
    n_samples = np.size(p_values)
    p_corr = alpha / n_samples
    sp_values = np.sort(p_values)
    critical_set = sp_values[
        sp_values < p_corr * np.arange(1, n_samples + 1)]
    if len(critical_set) > 0:
        critical_p_value = critical_set.max()
    else:
        critical_p_value = p_corr
    return critical_p_value


def fdr(p_values=None, verbose=0):
    """Returns the FDR associated with each p value

    Parameters
    -----------
    p_values : ndarray of shape (n)
        The samples p-value

    Returns
    -------
    q : array of shape(n)
        The corresponding fdr values
    """
    p_values = check_p_values(p_values)
    n_samples = p_values.size
    order = p_values.argsort()
    sp_values = p_values[order]

    # compute q while in ascending order
    q = np.minimum(1, n_samples * sp_values / np.arange(1, n_samples + 1))
    for i in range(n_samples - 1, 0, - 1):
        q[i - 1] = min(q[i], q[i - 1])

    # reorder the results
    inverse_order = np.arange(n_samples)
    inverse_order[order] = np.arange(n_samples)
    q = q[inverse_order]

    if verbose:
        import matplotlib.pylab as mp
        mp.figure()
        mp.xlabel('Input p-value')
        mp.plot(p_values, q, '.')
        mp.ylabel('Associated fdr')
    return q


class NormalEmpiricalNull(object):
    """Class to compute the empirical null normal fit to the data.

    The data which is used to estimate the FDR, assuming a Gaussian null
    from Schwartzmann et al., NeuroImage 44 (2009) 71--82
    """

    def __init__(self, x):
        """Initialize an empirical null normal object.

        Parameters
        -----------
        x : 1D ndarray
            The data used to estimate the empirical null.
        """
        x = np.reshape(x, (- 1))
        self.x = np.sort(x)
        self.n = np.size(x)
        self.learned = 0

    def learn(self, left=0.2, right=0.8):
        """
        Estimate the proportion, mean and variance of a Gaussian distribution
        for a fraction of the data

        Parameters
        ----------
        left: float, optional
            Left cut parameter to prevent fitting non-gaussian data
        right: float, optional
            Right cut parameter to prevent fitting non-gaussian data

        Notes
        -----
        This method stores the following attributes:

        * mu = mu
        * p0 = min(1, np.exp(lp0))
        * sqsigma: variance of the estimated normal
          distribution
        * sigma: np.sqrt(sqsigma) : standard deviation of the estimated
          normal distribution
        """
        # take a central subsample of x
        x = self.x[int(self.n * left): int(self.n * right)]

        # generate the histogram
        step = 3.5 * np.std(self.x) / np.exp(np.log(self.n) / 3)
        bins = max(10, int((self.x.max() - self.x.min()) // step))
        hist, ledge = np.histogram(x, bins=bins)
        step = ledge[1] - ledge[0]
        medge = ledge + 0.5 * step

        # remove null bins
        hist = hist[hist > 0].astype(np.float)
        medge = medge[:-1][hist > 0]  # edges include rightmost outer

        # fit the histogram
        dmtx = np.ones((3, len(hist)))
        dmtx[1] = medge
        dmtx[2] = medge ** 2
        coef = np.dot(np.log(hist), pinv(dmtx))
        sqsigma = - 1.0 / (2 * coef[2])
        sqsigma = max(sqsigma, 1.e-6)
        mu = coef[1] * sqsigma
        lp0 = (coef[0] - np.log(step * self.n)
                + 0.5 * np.log(2 * np.pi * sqsigma) + mu ** 2 / (2 * sqsigma))
        self.mu = mu
        self.p0 = min(1, np.exp(lp0))
        self.sigma = np.sqrt(sqsigma)
        self.sqsigma = sqsigma

    def fdrcurve(self):
        """
        Returns the FDR associated with any point of self.x
        """
        import scipy.stats as st
        if self.learned == 0:
            self.learn()
        efp = (self.p0 * st.norm.sf(self.x, self.mu, self.sigma)
               * self.n / np.arange(self.n, 0, - 1))
        efp = np.minimum(efp, 1)
        ix = np.argsort(self.x)
        for i in range(np.size(efp) - 1, 0, - 1):
            efp[ix[i - 1]] = np.maximum(efp[ix[i]], efp[ix[i - 1]])
        self.sorted_x = self.x[ix]
        self.sorted_fdr = efp[ix]
        return efp

    def threshold(self, alpha=0.05, verbose=0):
        """
        Compute the threshold corresponding to an alpha-level FDR for x

        Parameters
        -----------
        alpha : float, optional
            the chosen false discovery rate threshold.
        verbose : boolean, optional
            the verbosity level, if True a plot is generated.

        Returns
        -------
        theta: float
            the critical value associated with the provided FDR
        """
        efp = self.fdrcurve()

        if verbose:
            self.plot(efp, alpha)

        if efp[-1] > alpha:
            print("the maximal value is %f , the corresponding FDR is %f "
                  % (self.x[ - 1], efp[ - 1]))
            return np.inf
        j = np.argmin(efp[:: - 1] < alpha) + 1
        return 0.5 * (self.x[ - j] + self.x[ - j + 1])

    def uncorrected_threshold(self, alpha=0.001, verbose=0):
        """Compute the threshold corresponding to a specificity alpha for x

        Parameters
        -----------
        alpha : float, optional
            the chosen false discovery rate (FDR) threshold.
        verbose : boolean, optional
            the verbosity level, if True a plot is generated.

        Returns
        -------
        theta: float
            the critical value associated with the provided p-value
        """
        if self.learned == 0:
            self.learn()
        threshold = st.norm.isf(alpha, self.mu, self.sigma)
        if not np.isfinite(threshold):
            threshold = np.inf
        if verbose:
            self.plot()
        return threshold

    def fdr(self, theta):
        """Given a threshold theta, find the estimated FDR

        Parameters
        ----------
        theta : float or array of shape (n_samples)
            values to test

        Returns
        -------
        afp : value of array of shape(n)
        """
        from scipy.stats import norm
        self.fdrcurve()
        if np.isscalar(theta):
            if theta > self.sorted_x[ - 1]:
                return 0
            maj = np.where(self.sorted_x >= theta)[0][0]
            efp = (self.p0 * norm.sf(theta, self.mu, self.sigma) * self.n\
                  / np.sum(self.x >= theta))
            efp = np.maximum(self.sorted_fdr[maj], efp)
        else:
            efp = []
            for th in theta:
                if th > self.sorted_x[ - 1]:
                    efp.append(0)
                    continue
                maj = self.sorted_fdr[np.where(self.sorted_x >= th)[0][0]]
                efp.append(np.maximum(maj, self.p0 * st.norm.sf(th, self.mu,
                           self.sigma) * self.n / np.sum(self.x >= th)))
            efp = np.array(efp)
            #
        efp = np.minimum(efp, 1)
        return efp

    def plot(self, efp=None, alpha=0.05, bar=1, mpaxes=None):
        """Plot the  histogram of x

        Parameters
        ------------
        efp : float, optional
            The empirical FDR (corresponding to x)
            if efp==None, the false positive rate threshold plot is not
            drawn.
        alpha : float, optional
            The chosen FDR threshold
        bar=1 : bool, optional
        mpaxes=None: if not None, handle to an axes where the fig
        will be drawn. Avoids creating unnecessarily new figures
        """
        if not self.learned:
            self.learn()
        n = np.size(self.x)
        bins = max(10, int(2 * np.exp(np.log(n) / 3.)))
        hist, ledge = np.histogram(self.x, bins=bins)
        hist = hist.astype('f') / hist.sum()
        step = ledge[1] - ledge[0]
        medge = ledge + 0.5 * step
        import scipy.stats as st
        g = self.p0 * st.norm.pdf(medge, self.mu, self.sigma)
        hist /= step

        import matplotlib.pylab as mp
        if mpaxes is None:
            mp.figure()
            ax = mp.subplot(1, 1, 1)
        else:
            ax = mpaxes
        if bar:
            # We need to cut ledge to len(hist) to accommodate for pre and
            # post numpy 1.3 hist semantic change.
            ax.bar(ledge[:len(hist)], hist, step)
        else:
            ax.plot(medge[:len(hist)], hist, linewidth=2)
        ax.plot(medge, g, 'r', linewidth=2)
        ax.set_title('Robust fit of the histogram', fontsize=12)
        l = ax.legend(('empirical null', 'data'), loc=0)
        for t in l.get_texts():
            t.set_fontsize(12)
        ax.set_xticklabels(ax.get_xticks(), fontsize=12)
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)

        if efp is not None:
            ax.plot(self.x, np.minimum(alpha, efp), 'k')


def three_classes_GMM_fit(x, test=None, alpha=0.01, prior_strength=100,
                          verbose=0, fixed_scale=False, mpaxes=None, bias=0,
                          theta=0, return_estimator=False):
    """Fit the data with a 3-classes Gaussian Mixture Model,
    i.e. compute some probability that the voxels of a certain map
    are in class disactivated, null or active

    Parameters
    ----------
    x: array of shape (nvox,1)
      The map to be analysed
    test: array of shape(nbitems,1), optional
      the test values for which the p-value needs to be computed
      by default (if None), test=x
    alpha: float, optional
      the prior weights of the positive and negative classes
    prior_strength: float, optional
      the confidence on the prior (should be compared to size(x))
    verbose: int
      verbosity mode
    fixed_scale: bool, optional
      boolean, variance parameterization. if True, the variance is locked to 1
      otherwise, it is estimated from the data
    mpaxes:
      axes handle used to plot the figure in verbose mode
      if None, new axes are created
    bias:  bool
      allows a rescaling of the posterior probability
      that takes into account the threshold theta. Not rigorous.
    theta: float
      the threshold used to correct the posterior p-values
      when bias=1; normally, it is such that test>theta
      note that if theta = -np.inf, the method has a standard behaviour
    return_estimator: boolean, optional
      If return_estimator is true, the estimator object is
      returned.

    Returns
    -------
    bfp : array of shape (nbitems,3):
        the posterior probability of each test item belonging to each component
        in the GMM (sum to 1 across the 3 classes)
        if np.size(test)==0, i.e. nbitem==0, None is returned
    estimator : nipy.labs.clustering.GMM object
        The estimator object, returned only if return_estimator is true.

    Notes
    -----
    Our convention is that:

    * class 1 represents the negative class
    * class 2 represents the null class
    * class 3 represents the positive class
    """
    from ..clustering.bgmm import VBGMM
    from ..clustering.gmm import GridDescriptor

    nvox = np.size(x)
    x = np.reshape(x, (nvox, 1))
    if test is None:
        test = x
    if np.size(test) == 0:
        return None

    sx = np.sort(x, 0)
    nclasses = 3

    # set the priors from a reasonable model of the data (!)
    # prior means
    mb0 = np.mean(sx[:int(alpha * nvox)])
    mb2 = np.mean(sx[int((1 - alpha) * nvox):])
    prior_means = np.reshape(np.array([mb0, 0, mb2]), (nclasses, 1))
    if fixed_scale:
        prior_scale = np.ones((nclasses, 1, 1)) * 1. / (prior_strength)
    else:
        prior_scale = np.ones((nclasses, 1, 1)) * 1. / \
            (prior_strength * np.var(x))
    prior_dof = np.ones(nclasses) * prior_strength
    prior_weights = np.array([alpha, 1 - 2 * alpha, alpha]) * prior_strength
    prior_shrinkage = np.ones(nclasses) * prior_strength

    # instantiate the class and set the priors
    BayesianGMM = VBGMM(nclasses, 1, prior_means, prior_scale,
                        prior_weights, prior_shrinkage, prior_dof)
    BayesianGMM.set_priors(prior_means, prior_weights, prior_scale,
                           prior_dof, prior_shrinkage)

    # estimate the model
    BayesianGMM.estimate(x, delta=1.e-8, verbose=max(0, verbose-1))

    # create a sampling grid
    if (verbose or bias):
        gd = GridDescriptor(1)
        gd.set([x.min(), x.max()], 100)
        gdm = gd.make_grid().squeeze()
        lj = BayesianGMM.likelihood(gd.make_grid())

    # estimate the prior weights
    bfp = BayesianGMM.likelihood(test)
    if bias:
        lw = np.sum(lj[gdm > theta], 0)
        weights = BayesianGMM.weights / (BayesianGMM.weights.sum())
        bfp = (lw / weights) * BayesianGMM.slikelihood(test)

    if verbose and (mpaxes is not False):
        BayesianGMM.show_components(x, gd, lj, mpaxes)

    bfp = (bfp.T / bfp.sum(1)).T
    if not return_estimator:
        return bfp
    else:
        return bfp, BayesianGMM


def gamma_gaussian_fit(x, test=None, verbose=0, mpaxes=False,
                       bias=1, gaussian_mix=0, return_estimator=False):
    """
    Computing some prior probabilities that the voxels of a certain map
    are in class disactivated, null or active using a gamma-Gaussian mixture

    Parameters
    ------------
    x: array of shape (nvox,)
        the map to be analysed
    test: array of shape (nbitems,), optional
        the test values for which the p-value needs to be computed
        by default, test = x
    verbose: 0, 1 or 2, optional
        verbosity mode, 0 is quiet, and 2 calls matplotlib to display
        graphs.
    mpaxes: matplotlib axes, optional
        axes handle used to plot the figure in verbose mode
        if None, new axes are created
        if false, nothing is done
    bias: float, optional
        lower bound on the Gaussian variance (to avoid shrinkage)
    gaussian_mix: float, optional
        if nonzero, lower bound on the Gaussian mixing weight
        (to avoid shrinkage)
    return_estimator: boolean, optional
        if return_estimator is true, the estimator object is
        returned.

    Returns
    -------
    bfp: array of shape (nbitems,3)
        The probability of each component in the mixture model for each
        test value
    estimator: nipy.labs.clustering.ggmixture.GGGM object
        The estimator object, returned only if return_estimator is true.
    """
    from . import ggmixture
    Ggg = ggmixture.GGGM()
    Ggg.init_fdr(x)
    Ggg.estimate(x, niter=100, delta=1.e-8, bias=bias, verbose=0,
                 gaussian_mix=gaussian_mix)
    if mpaxes is not False:
        # hyper-verbose mode
        Ggg.show(x, mpaxes=mpaxes)
        Ggg.parameters()
    if test is None:
        test = x

    test = np.reshape(test, np.size(test))

    bfp = np.array(Ggg.posterior(test)).T
    if return_estimator:
        return bfp, Ggg
    return bfp


def smoothed_histogram_from_samples(x, bins=None, nbins=256, normalized=False):
    """ Smooth histogram corresponding to density underlying the samples in `x`

    Parameters
    ----------
    x: array of shape(n_samples)
       input data
    bins: array of shape(nbins+1), optional
       the bins location
    nbins: int, optional
       the number of bins of the resulting histogram
    normalized: bool, optional
       if True, the result is returned as a density value

    Returns
    -------
    h: array of shape (nbins)
       the histogram
    bins: array of shape(nbins+1),
       the bins location
    """
    from scipy.ndimage import gaussian_filter1d

    # first define the bins
    if bins is None:
        h, bins = np.histogram(x, nbins)
        bins = bins.mean() + 1.2 * (bins - bins.mean())
        h, bins = np.histogram(x, bins)

    # possibly normalize to density
    h = 1.0 * h
    dc = bins[1] - bins[0]
    if normalized:
        h /= (dc * h.sum())

    # define the optimal width
    sigma = x.std() / (dc * np.exp(.2 * np.log(x.size)))

    # smooth the histogram
    h = gaussian_filter1d(h, sigma, mode='constant')

    return h, bins
