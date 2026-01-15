# IdentifyingEfficientSubmodelFidelityCombinations

This repo corresponds to the code needed to run Algorithm 1 in 
Rincon, R., &amp; Padgett, J.E. (2026) "Identifying submodel fidelity combinations that yield efficient multiscale models for infrastructure performance estimation".

It is a Python code for identifying ‘efficient submodel fidelity combinations’ in multiscale models. 
The main class is noted as `Efficient_model_selection` within the script 'Efficient_submodel_fidelity_combinations.py'. It uses `run_model_selection_criterion` to determine the model bias and limiting bias value.

To run an example, we have added a 'model generator function' called `funct` within 'Funct.py'.

## Example
```python
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
```

We have also added the 'Example_code' Jupyter Notebook with the explanation of the function and some plots obtained.
