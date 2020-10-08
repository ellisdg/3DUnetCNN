# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
One-dimensional Gamma-Gaussian mixture density classes : Given a set
of points the algo provides approcumate maximum likelihood estimates
of the mixture distribution using an EM algorithm.

Author: Bertrand Thirion and Merlin Keller 2005-2008
"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import scipy.stats as st
import scipy.special as sp


#############################################################################
# Auxiliary functions #######################################################
#############################################################################


def _dichopsi_log(u, v, y, eps=0.00001):
    """ Implements the dichotomic part of the solution of psi(c)-log(c)=y
    """
    if u > v:
        u, v = v, u
    t = (u + v) / 2
    if np.absolute(u - v) < eps:
        return t
    else:
        if sp.psi(t) - np.log(t) > y:
            return _dichopsi_log(u, t, y, eps)
        else:
            return _dichopsi_log(t, v, y, eps)


def _psi_solve(y, eps=0.00001):
    """ Solve psi(c)-log(c)=y by dichotomy
    """
    if y > 0:
        print("y", y)
        raise ValueError("y>0, the problem cannot be solved")
    u = 1.
    if y > sp.psi(u) - np.log(u):
        while sp.psi(u) - np.log(u) < y:
            u *= 2
        u /= 2
    else:
        while sp.psi(u) - np.log(u) > y:
            u /= 2
    return _dichopsi_log(u, 2 * u, y, eps)


def _compute_c(x, z, eps=0.00001):
    """
    this function returns the mle of the shape parameter if a 1D gamma
    density
    """
    eps = 1.e-7
    y = np.dot(z, np.log(x)) / np.sum(z) - np.log(np.dot(z, x) / np.sum(z))
    if y > - eps:
        c = 10
    else:
        c = _psi_solve(y, eps=0.00001)
    return c


def _gaus_dens(mean, var, x):
    """ evaluate the gaussian density (mean,var) at points x
    """
    Q = - (x - mean) ** 2 / (2 * var)
    return 1. / np.sqrt(2 * np.pi * var) * np.exp(Q)


def _gam_dens(shape, scale, x):
    """evaluate the gamma density (shape,scale) at points x

    Notes
    -----
    Returns 0 on negative subspace
    """
    ng = np.zeros(np.size(x))
    cst = - shape * np.log(scale) - sp.gammaln(shape)
    i = np.ravel(np.nonzero(x > 0))
    if np.size(i) > 0:
        lz = cst + (shape - 1) * np.log(x[i]) - x[i] / scale
        ng[i] = np.exp(lz)
    return ng


def _gam_param(x, z):
    """ Compute the parameters of a gamma density from data weighted points

    Parameters
    ----------
    x: array of shape(nbitem) the learning points
    z: array of shape(nbitem), their membership within the class

    Notes
    -----
    if no point is positive then the couple (1, 1) is returned
    """
    eps = 1.e-5
    i = np.ravel(np.nonzero(x > 0))
    szi = np.sum(z[i])
    if szi > 0:
        shape = _compute_c(x[i], z[i], eps)
        scale = np.dot(x[i], z[i]) / (szi * shape)
    else:
        shape = 1
        scale = 1
    return shape, scale


##############################################################################
# class `Gamma`
##############################################################################


class Gamma(object):
    """ Basic one dimensional Gaussian-Gamma Mixture estimation class

    Note that it can work with positive or negative values,
    as long as there is at least one positive value.
    NB : The gamma distribution is defined only on positive values.
    5 parameters are used:
    - mean: gaussian mean
    - var: gaussian variance
    - shape: gamma shape
    - scale: gamma scale
    - mixt: mixture parameter (weight of the gamma)
    """

    def __init__(self, shape=1, scale=1):
        self.shape = shape
        self.scale = scale

    def parameters(self):
        print("shape: ", self.shape, "scale: ", self.scale)

    def check(self, x):
        if (x.min() < 0):
            raise ValueError("negative values in input")

    def estimate(self, x, eps=1.e-7):
        """
        ML estimation of the Gamma parameters
        """
        self.check(x)
        n = np.size(x)
        y = np.sum(np.log(x)) / n - np.log(np.sum(x) / n)
        if y > - eps:
            self.shape = 1
        else:
            self.shape = _psi_solve(y)
        self.scale = np.sum(x) / (n * self.shape)


##############################################################################
# Gamma-Gaussian Mixture class
##############################################################################


class GGM(object):
    """
    This is the basic one dimensional Gaussian-Gamma Mixture estimation class
    Note that it can work with positive or negative values,
    as long as there is at least one positive value.
    NB : The gamma distribution is defined only on positive values.

    5 scalar members
    - mean: gaussian mean
    - var: gaussian variance (non-negative)
    - shape: gamma shape (non-negative)
    - scale: gamma scale (non-negative)
    - mixt: mixture parameter (non-negative, weight of the gamma)
    """

    def __init__(self, shape=1, scale=1, mean=0, var=1, mixt=0.5):
        self.shape = shape
        self.scale = scale
        self.mean = mean
        self.var = var
        self.mixt = mixt

    def parameters(self):
        """ print the paramteres of self
        """
        print("Gaussian: mean: ", self.mean, "variance: ", self.var)
        print("Gamma: shape: ", self.shape, "scale: ", self.scale)
        print("Mixture gamma: ", self.mixt, "Gaussian: ", 1 - self.mixt)

    def Mstep(self, x, z):
        """
        Mstep of the model: maximum likelihood
        estimation of the parameters of the model

        Parameters
        ----------
        x  : array of shape (nbitems,)
            input data
        z array of shape(nbitrems, 2)
            the membership matrix
        """
        # z[0,:] is the likelihood to be generated by the gamma
        # z[1,:] is the likelihood to be generated by the gaussian

        tiny = 1.e-15
        sz = np.maximum(tiny, np.sum(z, 0))

        self.shape, self.scale = _gam_param(x, z[:, 0])
        self.mean = np.dot(x, z[:, 1]) / sz[1]
        self.var = np.dot((x - self.mean) ** 2, z[:, 1]) / sz[1]
        self.mixt = sz[0] / np.size(x)

    def Estep(self, x):
        """
        E step of the estimation:
        Estimation of ata membsership

        Parameters
        ----------
        x: array of shape (nbitems,)
            input data

        Returns
        -------
        z: array of shape (nbitems, 2)
            the membership matrix
        """
        eps = 1.e-15
        z = np.zeros((np.size(x), 2), 'd')
        z[:, 0] = _gam_dens(self.shape, self.scale, x)
        z[:, 1] = _gaus_dens(self.mean, self.var, x)
        z = z * np.array([self.mixt, 1. - self.mixt])
        sz = np.maximum(np.sum(z, 1), eps)
        L = np.sum(np.log(sz)) / np.size(x)
        z = (z.T / sz).T
        return z, L

    def estimate(self, x, niter=10, delta=0.0001, verbose=False):
        """ Complete EM estimation procedure

        Parameters
        ----------
        x : array of shape (nbitems,)
            the data to be processed
        niter : int, optional
            max nb of iterations
        delta : float, optional
            criterion for convergence
        verbose : bool, optional
            If True, print values during iterations

        Returns
        -------
        LL, float
            average final log-likelihood
        """
        if x.max() < 0:
            # all the values are generated by the Gaussian
            self.mean = np.mean(x)
            self.var = np.var(x)
            self.mixt = 0.
            L = 0.5 * (1 + np.log(2 * np.pi * self.var))
            return L

        # proceed with standard estimate
        z, L = self.Estep(x)
        L0 = L - 2 * delta
        for i in range(niter):
            self.Mstep(x, z)
            z, L = self.Estep(x)
            if verbose:
                print(i, L)
            if (L < L0 + delta):
                break
            L0 = L
        return L

    def show(self, x):
        """ Visualization of the mm based on the empirical histogram of x

        Parameters
        ----------
        x : array of shape (nbitems,)
            the data to be processed
        """
        step = 3.5 * np.std(x) / np.exp(np.log(np.size(x)) / 3)
        bins = max(10, int((x.max() - x.min()) / step))
        h, c = np.histogram(x, bins)
        h = h.astype(np.float) / np.size(x)
        p = self.mixt

        dc = c[1] - c[0]
        y = (1 - p) * _gaus_dens(self.mean, self.var, c) * dc
        z = np.zeros(np.size(c))
        z = _gam_dens(self.shape, self.scale, c) * p * dc

        import matplotlib.pylab as mp
        mp.figure()
        mp.plot(0.5 * (c[1:] + c[:-1]), h)
        mp.plot(c, y, 'r')
        mp.plot(c, z, 'g')
        mp.plot(c, z + y, 'k')
        mp.title('Fit of the density with a Gamma-Gaussians mixture')
        mp.legend(('data', 'gaussian acomponent', 'gamma component',
                   'mixture distribution'))

    def posterior(self, x):
        """Posterior probability of observing the data x for each component

        Parameters
        ----------
        x: array of shape (nbitems,)
            the data to be processed

        Returns
        -------
        y, pg : arrays of shape (nbitem)
            the posterior probability
        """
        p = self.mixt
        pg = p * _gam_dens(self.shape, self.scale, x)
        y = (1 - p) * _gaus_dens(self.mean, self.var, x)
        return y / (y + pg), pg / (y + pg)


##############################################################################
# double-Gamma-Gaussian Mixture class
##############################################################################


class GGGM(object):
    """
    The basic one dimensional Gamma-Gaussian-Gamma Mixture estimation
    class, where the first gamma has a negative sign, while the second
    one has a positive sign.

    7 parameters are used:
    - shape_n: negative gamma shape
    - scale_n: negative gamma scale
    - mean: gaussian mean
    - var: gaussian variance
    - shape_p: positive gamma shape
    - scale_p: positive gamma scale
    - mixt: array of mixture parameter
    (weights of the n-gamma,gaussian and p-gamma)
    """

    def __init__(self, shape_n=1, scale_n=1, mean=0, var=1,
                 shape_p=1, scale_p=1, mixt=np.array([1.0, 1.0, 1.0]) / 3):
        """ Constructor

        Parameters
        -----------
        shape_n : float, optional
        scale_n: float, optional
            parameters of the nehative gamma; must be positive
        mean : float, optional
        var : float, optional
            parameters of the gaussian ; var must be positive
        shape_p : float, optional
        scale_p : float, optional
            parameters of the positive gamma; must be positive
        mixt : array of shape (3,), optional
            the mixing proportions; they should be positive and sum to 1
        """
        self.shape_n = shape_n
        self.scale_n = scale_n
        self.mean = mean
        self.var = var
        self.shape_p = shape_p
        self.scale_p = scale_p
        self.mixt = mixt

    def parameters(self):
        """ Print the parameters
        """
        print("Negative Gamma: shape: ", self.shape_n,
              "scale: ", self.scale_n)
        print("Gaussian: mean: ", self.mean, "variance: ", self.var)
        print("Positive Gamma: shape: ", self.shape_p, "scale: ", self.scale_p)
        mixt = self.mixt
        print("Mixture neg. gamma: ", mixt[0], "Gaussian: ", mixt[1],
              "pos. gamma: ", mixt[2])

    def init(self, x, mixt=None):
        """
        initialization of the different parameters

        Parameters
        ----------
        x: array of shape(nbitems)
           the data to be processed
        mixt : None or array of shape(3), optional
           prior mixing proportions. If None, the classes have equal weight
        """
        if mixt is not None:
            if np.size(mixt) == 3:
                self.mixt = np.ravel(mixt)
            else:
                raise ValueError('bad size for mixt')

        # gaussian
        self.mean = np.mean(x)
        self.var = np.var(x)

        # negative gamma
        i = np.ravel(np.nonzero(x < 0))
        if np.size(i) > 0:
            mn = - np.mean(x[i])
            vn = np.var(x[i])
            self.scale_n = vn / mn
            self.shape_n = mn ** 2 / vn
        else:
            self.mixt[0] = 0

        # positive gamma
        i = np.ravel(np.nonzero(x > 0))
        if np.size(i) > 0:
            mp = np.mean(x[i])
            vp = np.var(x[i])
            self.scale_p = vp / mp
            self.shape_p = mp ** 2 / vp
        else:
            self.mixt[2] = 0

        # mixing proportions
        self.mixt = self.mixt / np.sum(self.mixt)

    def init_fdr(self, x, dof=-1, copy=True):
        """
        Initialization of the class based on a fdr heuristic: the
        probability to be in the positive component is proportional to
        the 'positive fdr' of the data.  The same holds for the
        negative part.  The point is that the gamma parts should model
        nothing more that the tails of the distribution.

        Parameters
        ----------
        x: array of shape(nbitem)
            the data under consideration
        dof: integer, optional
            number of degrees of freedom if x is thought to be a student
            variate. By default, it is handeled as a normal
        copy: boolean, optional
            If True, copy the data.
        """
        # Safeguard ourselves against modifications of x, both by our
        # code, and by external code.
        if copy:
            x = x.copy()
        # positive gamma
        i = np.ravel(np.nonzero(x > 0))
        from .empirical_pvalue import fdr

        if np.size(i) > 0:
            if dof < 0:
                pvals = st.norm.sf(x)
            else:
                pvals = st.t.sf(x, dof)
            q = fdr(pvals)
            z = 1 - q[i]
            self.mixt[2] = np.maximum(0.5, z.sum()) / np.size(x)
            self.shape_p, self.scale_p = _gam_param(x[i], z)
        else:
            self.mixt[2] = 0

        # negative gamma
        i = np.ravel(np.nonzero(x < 0))
        if np.size(i) > 0:
            if dof < 0:
                pvals = st.norm.cdf(x)
            else:
                pvals = st.t.cdf(x, dof)
            q = fdr(pvals)
            z = 1 - q[i]
            self.shape_n, self.scale_n = _gam_param( - x[i], z)
            self.mixt[0] = np.maximum(0.5, z.sum()) / np.size(x)
        else:
            self.mixt[0] = 0
        self.mixt[1] = 1 - self.mixt[0] - self.mixt[2]

    def Mstep(self, x, z):
        """
        Mstep of the estimation:
        Maximum likelihood update the parameters of the three components

        Parameters
        ------------
        x: array of shape (nbitem,)
            input data
        z: array of shape (nbitems,3)
            probabilistic membership
        """
        tiny = 1.e-15
        sz = np.maximum(np.sum(z, 0), tiny)
        self.mixt = sz / np.sum(sz)

        # negative gamma
        self.shape_n, self.scale_n = _gam_param( - x, z[:, 0])

        # gaussian
        self.mean = np.dot(x, z[:, 1]) / sz[1]
        self.var = np.dot((x - self.mean) ** 2, z[:, 1]) / sz[1]

        # positive gamma
        self.shape_p, self.scale_p = _gam_param(x, z[:, 2])

    def Estep(self, x):
        """ Update probabilistic memberships of the three components

        Parameters
        ----------
        x: array of shape (nbitems,)
            the input data

        Returns
        -------
        z: ndarray of shape (nbitems, 3)
            probabilistic membership

        Notes
        -----
        z[0,:] is the membership the negative gamma
        z[1,:] is the membership of  the gaussian
        z[2,:] is the membership of the positive gamma
        """
        tiny = 1.e-15
        z = np.array(self.component_likelihood(x)).T * self.mixt
        sz = np.maximum(tiny, np.sum(z, 1))
        L = np.mean(np.log(sz))
        z = (z.T / sz).T
        return z, L

    def estimate(self, x, niter=100, delta=1.e-4, bias=0, verbose=0,
                 gaussian_mix=0):
        """ Whole EM estimation procedure:

        Parameters
        ----------
        x: array of shape (nbitem)
            input data
        niter: integer, optional
            max number of iterations
        delta: float, optional
            increment in LL at which convergence is declared
        bias: float, optional
            lower bound on the gaussian variance (to avoid shrinkage)
        gaussian_mix: float, optional
            if nonzero, lower bound on the gaussian mixing weight
            (to avoid shrinkage)
        verbose: 0, 1 or 2
            verbosity level

        Returns
        -------
        z: array of shape (nbitem, 3)
            the membership matrix
        """
        z, L = self.Estep(x)

        L0 = L - 2 * delta
        for i in range(niter):
            self.Mstep(x, z)
            # Constraint the Gaussian variance
            if bias > 0:
                self.var = np.maximum(bias, self.var)
            # Constraint the Gaussian mixing ratio
            if gaussian_mix > 0 and self.mixt[1] < gaussian_mix:
                upper, gaussian, lower = self.mixt
                upper_to_lower = upper / (lower + upper)
                gaussian = gaussian_mix
                upper = (1 - gaussian_mix) * upper_to_lower
                lower = 1 - gaussian_mix - upper
                self.mixt = lower, gaussian, upper

            z, L = self.Estep(x)
            if verbose:
                print(i, L)
            if (L < L0 + delta):
                break
            L0 = L

        return z

    def posterior(self, x):
        """
        Compute the posterior probability of the three components
        given the data

        Parameters
        -----------
        x: array of shape (nbitem,)
            the data under evaluation

        Returns
        --------
        ng,y,pg: three arrays of shape(nbitem)
            the posteriori of the 3 components given the data

        Notes
        -----
        ng + y + pg = np.ones(nbitem)
        """
        p = self.mixt
        ng, y, pg = self.component_likelihood(x)
        total = ng * p[0] + y * p[1] + pg * p[2]
        return ng * p[0] / total, y * p[1] / total, pg * p[2] / total

    def component_likelihood(self, x):
        """
        Compute the likelihood of the data x under
        the three components negative gamma, gaussina, positive gaussian

        Parameters
        -----------
        x: array of shape (nbitem,)
            the data under evaluation

        Returns
        --------
        ng,y,pg: three arrays of shape(nbitem)
            The likelihood of the data under the 3 components
        """
        ng = _gam_dens(self.shape_n, self.scale_n, - x)
        y = _gaus_dens(self.mean, self.var, x)
        pg = _gam_dens(self.shape_p, self.scale_p, x)

        return ng, y, pg

    def show(self, x, mpaxes=None):
        """ Visualization of mixture shown on the empirical histogram of x

        Parameters
        ----------
        x: ndarray of shape (nditem,)
            data
        mpaxes: matplotlib axes, optional
            axes handle used for the plot if None, new axes are created.
        """
        import matplotlib.pylab as mp

        step = 3.5 * np.std(x) / np.exp(np.log(np.size(x)) / 3)
        bins = max(10, int((x.max() - x.min()) / step))
        h, c = np.histogram(x, bins)
        h = h.astype(np.float) / np.size(x)
        dc = c[1] - c[0]

        ng = self.mixt[0] * _gam_dens(self.shape_n, self.scale_n, - c)
        y = self.mixt[1] * _gaus_dens(self.mean, self.var, c)
        pg = self.mixt[2] * _gam_dens(self.shape_p, self.scale_p, c)
        z = y + pg + ng

        if mpaxes is None:
            mp.figure()
            ax = mp.subplot(1, 1, 1)
        else:
            ax = mpaxes

        ax.plot(0.5 * (c[1:] + c[:-1]), h / dc, linewidth=2, label='data')
        ax.plot(c, ng, 'c', linewidth=2, label='negative gamma component')
        ax.plot(c, y, 'r', linewidth=2, label='Gaussian component')
        ax.plot(c, pg, 'g', linewidth=2, label='positive gamma component')
        ax.plot(c, z, 'k', linewidth=2, label='mixture distribution')
        ax.set_title('Fit of the density with a Gamma-Gaussian mixture',
                     fontsize=12)
        l = ax.legend()
        for t in l.get_texts():
            t.set_fontsize(12)
        ax.set_xticklabels(ax.get_xticks(), fontsize=12)
        ax.set_yticklabels(ax.get_yticks(), fontsize=12)
