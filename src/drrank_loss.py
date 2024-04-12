##########################################################
#### Loss functions and estimation functions          ####
##########################################################
import numpy as np

def tau(i_j, Pij, Dij):
    """
    Computed expected tau using NumPy for faster computation
    Parameters:
    i_j: Coordinates of the Pij
    Pij: Posterior estimates of the probability of observation i being more biased than observation j
    Dij: Pairwise indicators
    """
    a_array = np.array(i_j)
    tau_sum = np.sum(
        Pij[a_array[:, 0]] * Dij[a_array[:, 0]] + Pij[a_array[:, 1]] * Dij[a_array[:, 1]]
        - Pij[a_array[:, 0]] * Dij[a_array[:, 1]] - Pij[a_array[:, 1]] * Dij[a_array[:, 0]]
    )
    return tau_sum


def dp(i_j, Pij, Dij):
    """
    Computed expected discordance proportion using NumPy
    Parameters:
    i_j: Coordinates of the Pij
    Pij: Posterior estimates of the probability of observation i being more biased than observation j
    Dij: Pairwise indicators
    Eij: Pairwise indicators
    """
    a_array = np.array(i_j)
    dp_sum = np.sum(
        Pij[a_array[:, 0]] * Dij[(a_array[:, 1], a_array[:, 0])]
        + Pij[(a_array[:, 1], a_array[:, 0])] * Dij[a_array[:, 0]]
    )
    return dp_sum


