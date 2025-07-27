import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from environment.heterogeneous_cluster import HeterogeneousClusterEnvironment
from experiments.utils import stats
import torch
import time
from datetime import datetime

from learners import UCBLearner, OracleUCBLearner, ClusteredKernelUCB


def run_ucb(env, T):
    total_arms = sum(env.num_arms)
    learner    = UCBLearner(total_arms)

    regret_list = []
    delta       = env.regret_per_action()

    arm_to_cluster = {}
    idx = 0
    for c in range(env.num_cluster):
        for a in range(env.num_arms[c]):
            arm_to_cluster[idx] = (c, a)
            idx += 1

    for i in range(total_arms):
        c, a = arm_to_cluster[i]
        r    = env.feedback(c, a)
        learner.receive_feedback(i, r)
        regret_list.append(delta[c][a])

    for t in range(total_arms, T):
        ucb_vals   = learner.get_ucb(t)
        chosen_arm = np.argmax(ucb_vals)
        c, a       = arm_to_cluster[chosen_arm]
        r          = env.feedback(c, a)
        learner.receive_feedback(chosen_arm, r)
        regret_list.append(delta[c][a])

    return np.array(regret_list)


def run_oracle_ucb(env, T):
    learner = OracleUCBLearner(env.num_cluster, env.num_arms, env.arm_dist_types)
    regret_list = []
    delta = env.regret_per_action()

    # Global arm mapping
    arm_to_cluster = {}
    idx = 0
    for c in range(env.num_cluster):
        for a in range(env.num_arms[c]):
            arm_to_cluster[idx] = (c, a)
            idx += 1

    total_arms = sum(env.num_arms)

    # Initialization: pull every arm once
    for i in range(total_arms):
        c, a = arm_to_cluster[i]
        r    = env.feedback(c, a)
        learner.receive_feedback(c, a, r)
        regret_list.append(delta[c][a])

    for t in range(total_arms, T):
        ucb_vals   = learner.get_ucb(t)
        chosen_arm = np.argmax(ucb_vals)
        c, a       = arm_to_cluster[chosen_arm]
        r          = env.feedback(c, a)
        learner.receive_feedback(c, a, r)
        regret_list.append(delta[c][a])

    return np.array(regret_list)



def run_ck_ucb(env, T, rep=None):

    """
    Run the CK-UCB algorithm for T rounds on the given environment.
    """
    if rep is not None:
        logging.info(f"Replication {rep+1}: Starting CK-UCB run")

    learner = ClusteredKernelUCB(
        num_arms       = env.num_arms,  
        contexts       = env.contexts,  
        lambda_        = 0.1,  # Regularization parameter
        gamma          = 1.0,  # Bandwidth for Gaussian kernel
        B              = 1.0,  # RKHS norm bound
        sigma          = 1.0,  # Sub-Gaussian noise parameter
        delta          = 0.1,  # Confidence level parameter
    )
    # Log CK-UCB hyperparameters
    logging.info(
        f"CK-UCB hyperparameters: "
        f"lambda={learner.lambda_}, gamma={learner.gamma}, "
        f"B={learner.B}, sigma={learner.sigma}, delta={learner.delta}"
    )

    regret_list = []
    delta       = env.regret_per_action()

    # Global arm mapping
    arm_to_cluster = {}
    idx = 0
    for c in range(env.num_cluster):
        for a in range(env.num_arms[c]):
            arm_to_cluster[idx] = (c, a)
            idx += 1

    total_arms = sum(env.num_arms)

    # Initialization: pull every arm once
    for i in range(total_arms):
        c_, a_ = arm_to_cluster[i]
        r_     = env.feedback(c_, a_)
        regret = delta[c_][a_]
        logging.info(f"[Rep {rep+1}] Init t={i}, arm={i}, reward={r_:.4f}, regret={regret:.4f}")
        learner.update(i, r_)
        regret_list.append(delta[c_][a_])

    # Main loop
    for t in tqdm(range(total_arms, T), desc=f"Timestep", position=1, leave=False):
        ucb_vals   = learner.select_arm(t)
        chosen_arm = np.argmax(ucb_vals)
        c_, a_     = arm_to_cluster[chosen_arm]
        r_         = env.feedback(c_, a_)
        regret = delta[c_][a_]
        if t % 100 == 0 or t == T - 1:
            logging.info(f"[Rep {rep+1}] Main t={t}, chosen arm={chosen_arm}, "
                         f"reward={r_:.4f}, regret={regret:.4f}, ucb={ucb_vals[chosen_arm]:.4f}")            
        learner.update(chosen_arm, r_)
        regret_list.append(delta[c_][a_])

    return np.array(regret_list)


def run_experiments(env, T, repeat):

    ucb_results    = []
    oracle_results = []
    ck_ucb_results    = []

    for rep in tqdm(range(repeat), desc="Replications", position=0, leave=True):
        logging.info(f"================================================")
        logging.info(f"================================================")
        logging.info(f"================================================")
        logging.info(f"Starting replication {rep+1}/{repeat}")
        results_ucb = run_ucb(env, T).cumsum()
        logging.info(f"Finished standard UCB replication {rep+1}")
        ucb_results.append(results_ucb)
        results_oracle = run_oracle_ucb(env, T).cumsum()
        logging.info(f"Finished Oracle UCB replication {rep+1}")
        oracle_results.append(results_oracle)
        results_ck = run_ck_ucb(env, T, rep).cumsum()
        logging.info(f"Finished CK-UCB replication {rep+1}")
        ck_ucb_results.append(results_ck)
        # Summary of final cumulative regret for this replication
        final_ucb     = results_ucb[-1]
        final_oracle = results_oracle[-1]
        final_ck      = results_ck[-1]
        logging.info(
            f"Replication {rep+1} summary: "
            f"final cumulative regret - UCB: {final_ucb:.4f}, "
            f"Oracle: {final_oracle:.4f}, CK-UCB: {final_ck:.4f}"
        )

    ucb_results    = np.array(ucb_results)
    oracle_results = np.array(oracle_results)
    ck_ucb_results = np.array(ck_ucb_results)

    return {
        'ucb'   : (ucb_results),
        'oracle': (oracle_results),
        'ck-ucb': (ck_ucb_results),
    }



experiment_name = 'ckucb_3.3b'
# 2) nonmonotonic, arm thetas scattered, std=0.5; sigma=None

# Set up experiment logging directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join('runs', experiment_name, timestamp)
os.makedirs(log_dir, exist_ok=True)
# Configure logging to file
logging.basicConfig(
    filename=os.path.join(log_dir, 'log.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info(f"Starting experiment {experiment_name}")



# Environment configuration
num_arms     = [3, 2, 3]
contexts     = np.array([
                [1, 0.2, 0],
                [1, 0.4, 0],
                [1, 0.6, 0],
                
                [0, 0.7, 0.1],
                [0, 0.8, 0.2],
                
                [0.2, 0.8, 1],
                [0.3, 0.7, 1],
                [0.4, 0.6, 1],
                ])
#theta_array  = [0.6, 0.5, 0.4]              # shared θ per cluster
# TODO: shared variance; 1) cluster group mean, no overlap arm theta 2) cluster thetas sampled from cluster level theta dist.
# TODO: 1) monotonic no shared mean, 2) nonmonotoic, scattered thetas, from dist 3) fixed shared variance, then get theta. [all with shared context feature within cluster]
cluster_theta_dist_mean = [0.6, 0.5, 0.4]  # shared cluster mean
cluster_theta_dist_std  = 0.2  # std for cluster level theta distribution

#theta_array  = [[0.6,0.5,0.4], [0.5,0.4], [0.4,0.3,0.2]]     # diff theta within clusters. 
theta_array  = [
    np.random.normal(loc=cluster_theta_dist_mean[0], scale=cluster_theta_dist_std, size=num_arms[0]).tolist(),
    np.random.normal(loc=cluster_theta_dist_mean[1], scale=cluster_theta_dist_std, size=num_arms[1]).tolist(),
    np.random.normal(loc=cluster_theta_dist_mean[2], scale=cluster_theta_dist_std, size=num_arms[2]).tolist()
]  # diff theta within clusters.
sigma_array  = [None, None, None]                # σ for Gaussian rewards
arm_dist_types = [
    ['gaussian', 'bernoulli', 'exponential'],
    ['bernoulli', 'gaussian'],
    ['exponential', 'gaussian', 'bernoulli'],
]


env = HeterogeneousClusterEnvironment(
        num_arms, theta_array, 
        sigma_array, arm_dist_types,
        contexts
)

T       = 10000
repeat  = 20

# Log environment and experiment settings
logging.info("Environment settings:")
logging.info(f"  num_arms      = {num_arms}")
logging.info(f"  contexts.shape= {contexts.shape}")
logging.info(f"  contexts      = {contexts.tolist()}")
logging.info(f"  theta_array   = {(theta_array)}")
logging.info(f"  sigma_array   = {sigma_array}")
logging.info(f"  arm_dist_types= {arm_dist_types}")
logging.info(f"  T             = {T}")
logging.info(f"  repeat        = {repeat}")



logging.info(f"Actual settings:")
logging.info(f"   Thetas: {[c.thetas for c in env.clusters]}")
logging.info(f"   Vars: {[c.vars for c in env.clusters]}")
logging.info(f"   Arm types: {[c.dist_types for c in env.clusters]}")



# Use automatic mixed precision for the heavy GPU linear algebra
with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                        dtype=torch.bfloat16):
    results = run_experiments(env, T, repeat)

# Save experiment results dict
results_path = os.path.join(log_dir, 'results.pkl')
with open(results_path, 'wb') as f:
    pickle.dump(results, f)
logging.info(f"Saved results to {results_path}")


plt.figure(figsize=(9, 6))
x_axis = np.arange(1, T + 1)


avg_ucb, ub_ucb, lb_ucb = stats(results['ucb'])
plt.plot(x_axis, avg_ucb, 'b-', label="Standard UCB")
plt.fill_between(x_axis, lb_ucb, ub_ucb, color='blue', alpha=0.15)

avg_oracle, ub_oracle, lb_oracle = stats(results['oracle'])
plt.plot(x_axis, avg_oracle, 'r-', label="UCB-C")
plt.fill_between(x_axis, lb_oracle, ub_oracle, color='red', alpha=0.15)

avg_bic, ub_bic, lb_bic = stats(results['ck-ucb'])
plt.plot(x_axis, avg_bic, 'g-', label="CK-UCB")
plt.fill_between(x_axis, lb_bic, ub_bic, color='green', alpha=0.15)

plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.title("3.3: Cumulative Regret of UCB, UCB-C, and CK-UCB")
plt.legend()
plt.tight_layout()
# Save the plot as PNG
plot_path = os.path.join(log_dir, 'cumulative_regret.png')
plt.savefig(plot_path)
logging.info(f"Saved plot to {plot_path}")
plt.show()
