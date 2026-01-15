# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:26:21 2026

@author: jr93
"""
import numpy as np
import time
# This is an example on how to use the Algorithm 1


class funct:
    """Code to create example functions of varying fidelity."""

    def __init__(self, mode="MABE",
                 cost_per_sim: float = None,
                 seed: int = None):
        """
        Initiate the model.

        Parameters
        ----------
        mode : str, optional
            MABE model is created if mode="MABE" or a reduced fidelity model is created if mode="k".
            Creates a second reduced fidelity otherwise. The default is "MABE".
        cost_per_sim : float, optional
            IF the cost is not dominated by computational time, the user can pass the cost of the
            model per simulation. The default is None.
        seed : int, optional
            Seed for creating the correlated random imput. The default is None.
        """
        self.mode = mode
        self.seed = None
        self.responses = None
        self.saved_responses = 0
        self.model_cost = dict()
        self.cost_per_sim = cost_per_sim

    def _get_input_samples(self, samples):
        rng = np.random.default_rng(self.seed)
        x1 = rng.normal(10, 1, samples)
        x2 = rng.normal(-5, 1, samples)
        return [x1, x2]

    def estimate_responses(self, seed: int, samples: int, previous_evals: int = None):
        """
        Estimate the outcomes of the model; it checks if they were already computed.

        Parameters
        ----------
        seed : int
            Seed used for sampling.
        samples : int
            Number of samples to evaluate responses.
        previous_evals : int, optional
            This model saves the responses, then this function checks the responses using
            `previous_evals` to avoid running again samples already analyzed. The default is None.

        Returns
        -------
        y : ndarray
            Array of outcomes (responses) of the model.
        costs_y : dict
            Dictionary with model costs associated with given numbers of evaluated samples.

        """
        if (previous_evals is not None) and (previous_evals + samples <= self.saved_responses):
            y = self.responses[previous_evals:previous_evals + samples]
            costs_y = self.model_cost[str(previous_evals + samples)]
        elif (previous_evals is None) and (len(self.model_cost) > 0) and (samples <= self.saved_responses):
            y = self.responses[: samples]
            costs_y = self.model_cost[str(samples)]
        else:
            time_start = time.time()
            self.seed = seed
            rng = np.random.default_rng(123)
            x1, x2 = self._get_input_samples(samples)
            noise = rng.normal(0, 0.5, samples)

            if self.mode == "MABE":
                y = x1**2 + 3 * x2 + np.log(x1) + noise
                time.sleep(0.0002 * samples)
            elif self.mode == "k":
                y = x1**2 + 3.5 * x2 + 10 * noise
                time.sleep(0.0001 * samples)
            else:
                y = x1**2 + x1 * x2 * noise + 3.5 * x2 + 2 * noise
                time.sleep(0.0001 * samples)
            time_finish = time.time() - time_start

            self._save_responses(y)

            if self.cost_per_sim is not None:
                self.model_cost[str(self.saved_responses)
                                ] = self.saved_responses * self.cost_per_sim

            else:
                if len(self.model_cost) > 0:
                    self.model_cost[str(self.saved_responses)] = next(
                        reversed(self.model_cost.values())) + time_finish
                else:
                    self.model_cost[str(self.saved_responses)] = time_finish
            costs_y = self.model_cost[str(self.saved_responses)]

        return y, costs_y

    def _save_responses(self, y):
        if self.responses is None:
            self.responses = y
        else:
            self.responses = np.hstack((self.responses, y))
        self.saved_responses = len(self.responses)
