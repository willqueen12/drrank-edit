##########################################################
#### Report card estiamtes functions                  ####
##########################################################
import pandas as pd
import numpy as np
import os
from scipy.sparse.csgraph import connected_components
import multiprocessing
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt
from drrank_loss import dp, tau
import tqdm
import time

## function to clean the Pij matrix ##
def clean_data(pi):
    """
    Function to prepare the P_ij data to be fitted with the report card model
    p_ij: a numpy matrix of (n_obs,n_obs) dimension containing the posterior estimates of the probabilities
          P_ij of observation i being more biased than observation j - P_ij = Pr(Theta_i > Theta_j | Y_i, Y_j)
    """
    # Create a copy to not modify data in place
    p_ij = pi.copy()

    # Requires np.array as inputs
    if type(p_ij) is not np.ndarray:
        raise ValueError("Pi matrix must be numpy array.")

    # Fill the diagonal with 0, so no gain / cost to ranking i tied with i
    np.fill_diagonal(p_ij, 0)

    # Check if p_ij = 1-p_ij.T
    if not np.allclose(p_ij + np.eye(p_ij.shape[0]), (1-p_ij).T):
        print("Warrning: Pi matrix does not appear equal to (1 - Pi.T)")
        print("Max deviation {:7.6f}".format(np.max(np.abs(p_ij +  np.eye(p_ij.shape[0]) -  (1-p_ij).T))))

    # Get coordinate representation
    N = p_ij.shape[0]
    i = np.repeat(np.arange(N),N) + 1
    j = np.tile(np.arange(N),N) + 1
    v = np.ravel(p_ij)

    # Make a dataframe out of the coordinate representation of the matrix
    df_p_ij = pd.DataFrame({'i':i,'j':j,'P_ij':v})

    # Get the matrix coordinates
    df_p_ij['idx'] = list(zip(df_p_ij.i, df_p_ij.j))

    # Make a dictionary with keys the matrix coordinates, values the P_ij
    p_ij_dict = {x:v for x, v in zip(df_p_ij['idx'], df_p_ij['P_ij'])}

    # Split a single dictionary into multiple dictionaries
    i_j = list(p_ij_dict.keys())
    Pij = list(p_ij_dict.values())

    return i_j, Pij

## main function ##
import time
import numpy as np
import pandas as pd

def report_cards(i_j, Pij, lamb=None, DR=None, loss='binary', save_controls=False, save_dir="dump", save_name="_debug", warmstart=True, add_cond=True, FeasibilityTol=1e-9, IntFeasTol=1e-9, OptimalityTol=1e-9):
    """
    Compute the report cards without Gurobi optimization
    Parameters:
    i_j: Coordinates of the Pij
    Pij: Posterior estimates of the probability of observation i being more biased than observation j
    lamb: Tuning parameter trading off the gains of correctly ranking pairs of observations against the cost of misclassifying them - either lamb or DR must be specified
    DR: Discordance rate (i.e. shares of observation pairs misranked according to their grades) - either lamb or DR must be specified
    save_controls: if True, saves the estimates for debugging purposes (default = False)
    save_dir: if save_controls == True, name for the directory in which the estimates will be saved, if the default directory is used, it creates a folder named "dump" (default = "dump")
    save_name: if save_controls == True, name for the file in which the estimates will be saved (default = "_debug")
    FeasibilityTol: Feasibility Tolerance, Gurobi parameter (default = 1e-9)
    IntFeasTol: Integer feasibility Tolerance, Gurobi parameter (default = 1e-9)
    OptimalityTol: Optimality Tolerance, Gurobi parameter (default = 1e-9)
    """
    # Time execution
    start = time.time()

    # Check lambda and DR
    if (lamb is not None) and (DR is not None):
        raise AssertionError("Must supply either lambda or DR, but not both.")
    if lamb is not None and (lamb > 1 or lamb < 0):
        raise AssertionError("Lambda must be within [0, 1].")
    if DR is not None and (DR > 1 or DR < 0):
        raise AssertionError("DR must be within [0, 1].")

    n_firms = max(i_j)[0] + 1

    # Tolerances
    FeasibilityTol = 1e-9
    IntFeasTol = 1e-9
    OptimalityTol = 1e-9

    # control variables
    Dij = {}
    grades = {}

    # Add constraints
    print("Building constraints...")
    for i in tqdm.tqdm(range(1, n_firms)):
        for j in range(1, n_firms):
            Dij[(i, j)] = None  # Initialize Dij
            # Initialize grades if not initialized
            if i not in grades:
                grades[i] = None
            if j not in grades:
                grades[j] = None

    # Objective
    loss_condorcet = -tau(i_j, Pij, Dij)
    if lamb is not None:
        loss = (1 - lamb) * dp(i_j, Pij, Dij) - lamb * tau(i_j, Pij, Dij)
    elif DR is not None:
        loss = loss_condorcet
    else:
        raise AssertionError("Must supply either lambda or DR.")

    # Warmstart at lambda=1
    if warmstart:
        print("Warm starting at lambda = 1")
        loss_condorcet = None  # Define loss_condorcet based on your calculation
        D_ij_hat_cond = None  # Define D_ij_hat_cond based on your calculation

    loss = None  # Define loss based on your calculation

    # DR constraint, if necessary
    if DR is not None:
        npairs = np.sum(list(Pij.values()))
        dr_constr = None  # Define dr_constr based on your constraints

    # First optimize() if call fails - need to set NonConvex to 2
    try:
        print("Optimizing model...")
        # Optimize the model
        model_opt = None  # Define your optimization model
        model_opt.optimize()
    except:
        print("Optimize failed due to non-convexity")
        print("Optimizing with non-convexity")
        # Solve bilinear model
        model_opt = None  # Define your optimization model with non-convexity
        model_opt.optimize()
    print("Optimized model successfully")

    # Print and return results
    print("Time elapsed solving LP problem(s): {:4.3f} minutes".format((time.time() - start) / 60))
    return df_groups.drop('D_ij', axis=1)
    

## function to fit the ranking model ##
def fit(Pij, lamb, DR = None, save_controls = False, save_dir = "dump", save_name = '_debug', FeasibilityTol = 1e-9, IntFeasTol = 1e-9, OptimalityTol = 1e-9):
    """
    Function to fit the report card model on a Pij matrix of posterior estimates of bias
    Parameters:
    i_j: Coordinates of the Pij
    Pij: Posterior estimates of the probability of observation i being more biased than observation j
    lamb:  Tuning parameter trading off the gains of correctly ranking pairs of observations against the cost of misclassifying them
    DR: Discordance proportion (i.e. shares of observation pairs misranked according to their grades) - either lamb or DR must be specified
    save_controls: if True, saves the estimates for debugging purposes (default = False)
    save_dir: if save_controls == True, name for the directory in which the estimates will be saved, if the default directory is used, it creates a folder named "dump" (default = "dump")
    save_name: if save_controls == True, name for the file in which the estimates will be saved (default = "_debug")
    FeasibilityTol: Feasibility Tollerance, Gurobi parameter (default = 1e-9)
    IntFeasTol: Integer feasibility Tollerance, Gurobi parameter (default = 1e-9)
    OptimalityTol: Optimality Tollerance, Gurobi parameter (default = 1e-9)
    """    
    # Clean the Pij and retrieve the sparse representation
    i_j, Pij = clean_data(Pij)

    # Produce the report cards
    res_df = report_cards(i_j, Pij, lamb, DR, save_controls = save_controls, save_dir = save_dir, save_name = save_name, FeasibilityTol = FeasibilityTol, IntFeasTol = IntFeasTol, OptimalityTol = OptimalityTol)

    return res_df

## function to get different ranking estimations based on a list of lambdas ##
def fit_multiple(Pij, lamb_list, ncores = 1, save_controls = False, save_dir = "dump", save_name = '_debug', FeasibilityTol = 1e-9, IntFeasTol = 1e-9, OptimalityTol = 1e-9):
    """
    Function to fit the report card model on a Pij matrix of posterior estimates of bias
    -> gives back a dataframe with results for various lambda parameters
    Parameters:
    i_j: Coordinates of the Pij
    Pij: Posterior estimates of the probability of observation i being more biased than observation j
    lamb_list:  List of lambdas to be used (lambda: Tuning parameter trading off the gains of correctly ranking pairs of observations against the cost of misclassifying them)
    ncores: Number of cores, if set to -1 uses all the cores available (default = 1)
    save_controls: if True, saves the estimates for debugging purposes (default = False)
    save_dir: if save_controls == True, name for the directory in which the estimates will be saved, if the default directory is used, it creates a folder named "dump" (default = "dump")
    save_name: if save_controls == True, name for the file in which the estimates will be saved (default = "_debug")
    FeasibilityTol: Feasibility Tollerance, Gurobi parameter (default = 1e-9)
    IntFeasTol: Integer feasibility Tollerance, Gurobi parameter (default = 1e-9)
    OptimalityTol: Optimality Tollerance, Gurobi parameter (default = 1e-9)
    """
    # If ncores is -1, use all CPUs
    if ncores == -1:
        ncores = multiprocessing.cpu_count() 

    # Clean the Pij and retrieve the sparse representation
    i_j, Pij = clean_data(Pij)

    # final dataframe to store all results
    n_obs = max(i_j)[0] + 1
    final_df = pd.DataFrame({'obs_idx': range(1, n_obs)})

    def report_cards_helper(x):
        """
        Helper function to run report_cards within a multiprocessing tool
        x: lambda value over which we iterate
        """
        return report_cards(i_j, Pij, x, DR = None, add_cond=False, save_controls = save_controls, save_dir = save_dir, save_name = save_name, FeasibilityTol = FeasibilityTol, IntFeasTol = IntFeasTol, OptimalityTol = OptimalityTol)

    # Compute using multiprocessing to speed up everything
    print("Computing grades with {} cores".format(ncores), flush=True)
    with ThreadPool(ncores) as p:
        dfs = p.map(report_cards_helper, lamb_list)

    # Get the final result dataset
    for d in dfs:
        final_df = final_df.merge(d, on='obs_idx', validate="1:1")

    return final_df


def fig_ranks(ranking, posterior_features, gradecol=None, ylabels=None, show_plot=True, save_path=None, trans=False, ylabel_fontsize=8):
    """
    Function to plot an histogram of the estimates
    Arguments:
        ranking: dataframe with ranking results from drrank.fit()
        posterior_features: posterior features computed from the drrank.prior_estimate() class
        gradecol: column name with the ranking grades, useful when fitting multiple lambdas. If None, looks for the right columns and pick the first one
        ylabels: column in "ranking" providing the names of the observations, if None simply shows the observation number
        show_plot: set to True to display the plot
        save_path: specify the path to save the plot, set to None to avoid saving it
        trans: set to True if we want to plot the transformed posterior means
    """
    # Find smallest lambda that delivers num_groups
    if gradecol is not None:
        group = [x for x in ranking.keys() if gradecol in x][0]
    else:
        group = [x for x in ranking.keys() if 'grades_' in x][0]
    print("Using group {}".format(group))

    # First step - get posterior features:
    # Merge posterior features with the group rankings
    df_plot = pd.concat([ranking, posterior_features], axis = 1)

    # Sort everything
    df_plot = df_plot.sort_values("condorcet_rank", ascending=False)

    # Get upper and lower bounds
    if trans:
        df_plot["lb"] = df_plot['pmean_trans'] - df_plot['lci_trans']
        df_plot["ub"] = df_plot['uci_trans'] - df_plot['pmean_trans']
        mean_col = 'pmean_trans'
    else:
        df_plot["lb"] = df_plot['pmean'] - df_plot['lci']
        df_plot["ub"] = df_plot['uci'] - df_plot['pmean']
        mean_col = 'pmean'

    # Construct figure
    fig, ax1 = plt.subplots(dpi=400, figsize=(6,7))
    y_range = np.arange(len(df_plot))
    groups = df_plot[group].unique()
    # Prepare labels for each group
    if len(groups) > 6:
        labels = [r'${} \bigstar$'.format(c) for c in range(len(groups),0,-1)]
    else:
        labels = [r'$\bigstar$'*c for c in range(len(groups),0,-1)]

    # Iterate and produce a plot for each group
    mean_lines = []
    for i, gr in enumerate(np.flip(groups)):
        idx = df_plot[group] == gr
        mean_lines += [ax1.errorbar(df_plot[mean_col][idx], y_range[idx],
                        xerr=[df_plot['lb'][idx], df_plot['ub'][idx]],
                        alpha=0.6, ms=1.5, elinewidth=0.6,
                    fmt='o', capsize=1.5, label=labels[i]
                        )]

    plt.grid(axis='y', alpha=0.35, linewidth=0.3)
    plt.legend(mean_lines, labels, loc='lower right', markerscale=1)
    if ylabels != None:
        plt.yticks(y_range, df_plot[ylabels], fontsize=ylabel_fontsize)    
    else:
        plt.yticks(y_range, df_plot['obs_idx'], fontsize=ylabel_fontsize)    

    ax1.set_xlabel('Posterior means')
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    if save_path != None:
        plt.tight_layout()
        plt.savefig(save_path, format='jpg', dpi=300)
        print("Figure saved!")

    if show_plot:
        plt.show()

    plt.clf()
    plt.close('all')
