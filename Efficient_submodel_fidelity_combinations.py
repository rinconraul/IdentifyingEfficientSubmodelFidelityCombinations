# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 10:02:33 2025

@author: jrrin
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif']


class Efficient_model_selection:
    """
    Code for identifying ‘efficient submodel fidelity combinations’ in multiscale models.

    Examples
    --------
    >>> import Efficient_submodel_fidelity_combinations as algorithm1
    >>> from Funct import funct

    >>> m1 = funct(mode="MABE", cost_per_sim=10)
    >>> mk = funct(mode="k", cost_per_sim=0.1)

    >>> model_selection = algorithm1.Efficient_model_selection(m1, mk, alpha=0.05,
                                                    ci_lim=0.05, m=1000,
                                                    B_resamples=1000, W=100,
                                                    increm=100, seed=456)
    >>> results = model_selection.run_model_selection_criterion(max_iter=100)
    The model reached convergence. The estimated bias is  -4.479  and the limiting bias is  40.687.
    Gamma_m = 100.0
    >>> results
        (-4.479329137930949, 40.68711019287388)
    >>> model_selection.plot_bias_evolution()
    """

    def __init__(self, m1, mk, alpha: float,
                 ci_lim: float, m: int, B_resamples: int,
                 W: float, increm: int, abs_ci_length_lim: float = None,
                 min_iter_under_ci: int = 2,
                 seed: int = 2025):
        """
        Parameters for class Efficient_model_selection.

        Parameters
        ----------
        m1 : object
            Model instance with a ``estimate_responses(seed:int, samples:int, previous_evals:int)``
            method returning a tuple of model responses (ndarray) and model costs associated with
            the evaluated samples (dict). This model corresponds to the Most Accurate But Expensive
            (MABE) model.
        mk : object
            Model instance with a ``estimate_responses(seed:int, samples:int, previous_evals:int)``
            method returning a tuple of model responses (ndarray) and model costs associated with
            the evaluated samples (dict). This model corresponds to the a lower-fidelity model.
        alpha : float
            Significance level for the confidence interval.
        ci_lim : float
            Minimum acceptable normalized confidence interval; used as stopping criteria.
        m : int
            Initial number of simulations.
        B_resamples : int
            Number of bootstrap resamples.
        W : float
            Computational budget.
        increm : int
            Number of samples to add at every iteration.
        abs_ci_length_lim : float, optional
            Minimum acceptable confidence interval length; used as stopping criteria.
            The default is None.
        min_iter_under_ci : int, optional
            Minimum number of iterations of the algorithm. The default is 2 to observe at least two
            points when plotting the convergence plots.
        seed : int, optional
            Seed of random numbers. This seed is passed to the `m1` and `mk` objects to induce
            correlated sampling. The default is 2025.
        """
        self.model1 = m1
        self.modelk = mk
        self.alpha = alpha
        self.ci_lim = ci_lim,
        self.m = m
        self.budget = W
        self.resamples = B_resamples
        self.increm = increm
        self.seed = seed

        self.model1_cost = None
        self.modelk_cost = None
        self.stable_mu_bias = None
        self.non_converg_mu_bias = None
        self.y1 = None
        self.yk = None
        self.mu_bias = None
        self.ci_high = []
        self.ci_low = []
        self.ci_alpha = []
        self.samples_eval = []
        self.mu_bias_eval = []
        self.abs_ci_length = []
        self.abs_ci_length_lim = abs_ci_length_lim
        self.min_iter_under_ci = min_iter_under_ci

    def run_model_selection_criterion(self, max_iter: int = 100):
        """
        Start the model comparison based on estimated bias in `mk` with respect to `m1`.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations. This value limits the expense of the pilot analysis;
            i.e., works as a third stopping criteria. The default is 100.

        Returns
        -------
        mu_bias: float
            Returns the mean estimate for the model bias. The model prints a message to describe if
            this value was obtained after or prior convergence. Depending on the state of
            convergence, this value can be accessed by the parameter ``stable_mu_bias`` or by
            ``non_converg_mu_bias``
        bias_max : float
            Limiting Bias obtained using Equation 13 in Rincon, R. and Padgett, J.E.,(2026).
            Identifies the maximum bias acceptable to determine if the lower-fidelity multiscale
            model consists of a ‘efficient submodel fidelity combination’, given budget W .

        """
        cont = 0
        cont_min_ci_alp = 0
        cont_min_ci_len = 0
        if self.min_iter_under_ci > max_iter:
            print('The minimum number of iterations will be updated to match the maximum iterations\
                  defined by the user.')
            self.min_iter_under_ci = max_iter

        for i in range(max_iter):
            if self.y1 is None:
                self.compute_bias_sample(self.m, cont)
                if self.ci_lim is not None:
                    if self.ci_alpha[-1] < self.ci_lim:
                        cont_min_ci_alp = cont_min_ci_alp + 1
                        if (cont_min_ci_alp >= self.min_iter_under_ci):
                            self.stable_mu_bias = self.mu_bias
                if (self.abs_ci_length_lim is not None):
                    if (self.abs_ci_length[-1] < self.abs_ci_length_lim):
                        cont_min_ci_len = cont_min_ci_len + 1
                        if (cont_min_ci_len >= self.min_iter_under_ci):
                            self.stable_mu_bias = self.mu_bias
            else:
                if self.ci_alpha[-1] < self.ci_lim:
                    if (cont_min_ci_alp >= self.min_iter_under_ci):
                        self.stable_mu_bias = self.mu_bias
                    else:
                        cont_min_ci_alp = cont_min_ci_alp + 1
                        cont = self._evaluate_next_run(cont, max_iter)

                elif (self.abs_ci_length_lim is not None):
                    if self.abs_ci_length[-1] < self.abs_ci_length_lim:
                        if (cont_min_ci_len >= self.min_iter_under_ci):
                            self.stable_mu_bias = self.mu_bias
                        else:
                            cont_min_ci_len = cont_min_ci_len + 1
                            cont = self._evaluate_next_run(cont, max_iter)
                    else:
                        cont = self._evaluate_next_run(cont, max_iter)
                else:
                    cont = self._evaluate_next_run(cont, max_iter)

        # After simulations, if non convergency is observed, the model bias is saved in
        # non_converg_mu_bias
        if self.stable_mu_bias is None:
            self.non_converg_mu_bias = self.mu_bias
        try:
            gamma_m = (self.model1_cost) / (self.modelk_cost)
        except:
            gamma_m = 1

        sample_var_model1 = np.var(self.y1, ddof=1)
        sample_var_modelk = np.var(self.yk, ddof=1)

        squared_bias_max = (self.modelk_cost / (self.budget * self.m)) * \
            (gamma_m * sample_var_model1 - sample_var_modelk)
        self.bias_max = squared_bias_max**0.5
        self.sample_var_model1 = sample_var_model1
        self.sample_var_modelk = sample_var_modelk

        if self.stable_mu_bias:
            print('The model reached convergence. The estimated bias is ',
                  np.round(self.stable_mu_bias, 3), ' and the limiting bias is ',
                  np.round(self.bias_max, 3), '. Gamma_m =', np.round(gamma_m, 2))
            return self.stable_mu_bias, self.bias_max
        else:
            print('The model did not reach convergence. The estimated bias is ',
                  np.round(self.non_converg_mu_bias, 3), ' and the limiting bias is ',
                  np.round(self.bias_max, 3), '. Gamma_m =', np.round(gamma_m, 2), '.\n',
                  'Check if increasing max_iter or accepting the nonconvergence of the estimator \
                      is necessary.')
            return self.non_converg_mu_bias, self.bias_max

    def _evaluate_next_run(self, cont: int, max_iter: int):
        # This function evaluates the bias in a new sample
        if cont < max_iter - 1:
            cont = cont + 1
            self.compute_bias_sample(self.increm, cont, previous_evals=self.m)
        return cont

    def compute_bias_sample(self, samples: int, cont: int, previous_evals: int = None):
        """
        Compute the bias for a total of `samples`.

        Parameters
        ----------
        samples : int
            Number of samples to evaluate on `m1` and `mk`.
        cont : int
            Number of iteration to check if iterations<iter_max.
        previous_evals : int, optional
            If the model objects have implemented saving the responses, then this function checks
            the responses using `previous_evals` to avoid running again samples already analyzed.
            The default is None.

        Returns
        -------
        All the results are saved in the instance attributes.

        """
        y1, costs1 = self.model1.estimate_responses(self.seed + cont, samples, previous_evals)
        yk, costsk = self.modelk.estimate_responses(self.seed + cont, samples, previous_evals)

        if self.y1 is None:
            self.y1 = y1
            self.yk = yk

        else:
            self.y1 = np.hstack((self.y1, y1))
            self.yk = np.hstack((self.yk, yk))
        self.m = len(self.y1)

        deltas = self.yk - self.y1
        self.mu_bias = np.average(deltas)
        self.mu_bias_eval.append(self.mu_bias)
        self.model1_cost = costs1
        self.modelk_cost = costsk
        self.mu_y1 = np.mean(self.y1)
        self.mu_yk = np.mean(self.yk)

        ci_low, ci_high = self.bootstrap_mean_ci(deltas, B=self.resamples, alpha=self.alpha, seed=0)
        ci_alpha = (ci_high - ci_low) / abs(self.mu_bias)
        abs_ci_length = np.abs(ci_high - ci_low)

        self.ci_high.append(ci_high)
        self.ci_low.append(ci_low)
        self.ci_alpha.append(ci_alpha)
        self.abs_ci_length.append(abs_ci_length)
        self.samples_eval.append(self.m)

    def plot_bias_evolution(self, miny: float = None, minx: float = None):
        """
        Plot the evolution of the mu_bias and the normalized confidence interval.

        Parameters
        ----------
        miny : float, optional
            Minimum value of the y-axis. The default is None.
        minx : float, optional
            Minimum value of the x-axis. The default is None.

        Returns
        -------
        A matplotlib figure.

        """
        fig, ax = plt.subplots(1, 1, dpi=300, figsize=(3.5, 2.5), layout='constrained')
        ax.plot(self.samples_eval, self.mu_bias_eval, 'k', linewidth=2)
        ax.plot(self.samples_eval, self.ci_high, '--k', linewidth=1)
        ax.plot(self.samples_eval, self.ci_low, '--k', linewidth=1)
        ax.set_ylabel(r'$\hat{\mu}_{\mathrm{Bias}(\cdot)}$', color='k')

        ax2 = ax.twinx()
        ax2.plot(self.samples_eval, self.ci_alpha, 'b', alpha=0.5, zorder=0)
        ax2.set_ylabel(r'$\widetilde{\mathrm{CI}}_{\alpha}$', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        ax.set_xlabel('Simulations in pilot study')

        try:
            if miny is not None:
                ax.set_ylim([miny, 1.15 * max(self.ci_high)])
            else:
                if self.mu_bias_eval[-1] > 0:
                    maxy = max(self.ci_high)
                    ax.set_ylim([0, 1.15 * maxy])
                else:
                    miny = min(self.ci_high)
                    ax.set_ylim([1.15 * miny, 0])
        except:
            pass

        if minx is not None:
            ax.set_xlim([minx, max(self.samples_eval)])
        plt.show()

    def bootstrap_mean_ci(self, deltas, B: int = 1000,
                          alpha: float = 0.05, seed: int = None) -> tuple[float, float]:
        """
        Compute the bootstrap confidence interval for the mean of an array.

        Parameters
        ----------
        deltas : ndarray
            Array with the differences of the responses of `mk` and `m1`.
            These must be previously obtained by preserving the correlation among input parameters.
        B : int, optional
            Bootstrap samples. The default is 1000.
        alpha : float, optional
                Significance level for the confidence interval. The default is 0.05.
        seed : int, optional
            Seed for the resampling process. The default is None.

        Returns
        -------
        tuple[float, float]
            Lower and upper values of the bootstrap confidence interval of the mean.
        """
        rng = np.random.default_rng(seed)
        n = len(deltas)

        boot = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            boot[b] = deltas[idx].mean()

        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        return lo, hi
