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
    a0_indices = a_array[:, 0].astype(int)  # Convert to integer array
    a1_indices = a_array[:, 1].astype(int)  # Convert to integer array
    
    tau_sum = np.sum(
        Pij[a0_indices] * Dij[a0_indices] + Pij[a1_indices] * Dij[a1_indices]
        - Pij[a0_indices] * Dij[a1_indices] - Pij[a1_indices] * Dij[a0_indices]
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
    a0_indices = a_array[:, 0].astype(int)  # Convert to integer array
    a1_indices = a_array[:, 1].astype(int)  # Convert to integer array
    
    dp_sum = np.sum(
        Pij[a0_indices] * Dij[(a_array[:, 1], a_array[:, 0])] +
        Pij[(a_array[:, 1], a_array[:, 0])] * Dij[a0_indices]
    )
    return dp_sum


