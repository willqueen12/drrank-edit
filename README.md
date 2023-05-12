# DRrank

DRrank is a Python library to implement the Empirical Bayes ranking scheme developed in Kline, Rose, and Walters (2023).

## Installation:

The package uses the Gurobi optimizer. To use **DRrank** you must first install Gurobia and acquire a license. More guidance is available from Guorbi [here](https://www.gurobi.com/documentation/9.5/quickstart_windows/cs_python_installation_opt.html)). Gurobi offers a variety of free licenses for academic use. For more information, see the following [page](https://www.gurobi.com/academia/academic-program-and-licenses/).


After having successfully set up Gurobipy, install  **DRrank** via pip:

```bash
pip install drrank
```

## Usage

To compute rankings, provide the **fit** function with a matrix $P$ of posterior estimates of the probability observation i's latent measure (e.g., bias, quality, etc.) exceeds unit j's. That is, each element of this matrix takes the form:

$P_{ij} = Pr(\theta_i > \theta_j | Y_i = y_i, Y_j = y_j)$

**DRrank** expects these probabilities to satisfy $P_ij = 1-P_ji$. 


There are two ways to use **DRrank**.

First, One can supply a parameter $\lambda \in [0,1]$, which corresponds to the user's value of correctly ranking pairs of units relative to the costs of misclassifying them. $\lambda=1$ implies correct and incorrect rankings are valued equally, while $\lambda=0$ implies correct rankings are not valued at all. In pairwise comparisons between units, it is optimal to assign unit $i$ a higher grade than unit $j$ when $P_{ij} > 1/(1+\lambda)$, which implies $\lambda$ also corresponds to the minimum level of posterior certainty required to rank units pairwise.

```python
from drrank.drrank import fit
from drrank.simul import simul_data

# Simulate data
p_ij = simul_data(size = 25)

# Fit the report card function
results = fit(p_ij, lamb = 0.25, DP = None, save_controls=True)
```

We also provide a function to test a variety of different values of $\lambda$:

```python
from drrank.drrank import fit_tuning

# looping over lambda
lamdas = np.append(np.arange(0, 0.9, 0.01), [1.0])

# Try different values of Lambda
results_l = fit_multiple(p_ij, list(lamdas), DP = None)
```

Second, one can ask **DRrank** to compute grades that maximize Kendall (1938)'s $\tau$, a measure of the rank correlation between units' latent rankings and assigned grades, subject to a constraint on the expected share of pairwise units incorrectly misclassified, which we refer to as the discordance proportion.

```python

# Fit the report card function
results = fit(p_ij, lamb = None, DP = 0.05, save_controls=True)
```

