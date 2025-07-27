import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from datetime import datetime

from learners import UCBLearner, OracleUCBLearner, ClusteredKernelUCB
from environment.heterogeneous_cluster import HeterogeneousClusterEnvironment
from experiments.utils import stats


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

def stats(arr):
    # arr: shape (repeat, T)
    avg = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])  # standard error of the mean
    z = 1.96
    lb  = avg - z * sem
    ub  = avg + z * sem
    return avg, ub, lb


# Set up experiment logging directory with timestamp
experiment_name = 'ckucb_experiment_t10k_r20_0.1gap_diff_theta'
# Generate timestamp for this run
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



# ================= Experiment parameters =================
# Cluster setup (unchanged)
num_arms      = [3, 2, 3]
contexts      = np.array([
    [1, 0.2, 0], [1, 0.4, 0], [1, 0.6, 0],
    [0, 0.5, 0.1], [0, 0.8, 0.2],
    [0.2, 0.8, 1], [0.3, 0.7, 1], [0.4, 0.6, 1],
])
sigma_array   = [1, 1, 1]
arm_dist_types = [
    ['gaussian', 'bernoulli', 'exponential'],
    ['bernoulli', 'gaussian'],
    ['exponential', 'gaussian', 'bernoulli'],
]

T = 10000            # Time horizon
repeats = 20        # Number of replications per gap

gaps = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]

# ================= Prepare logging =================
exp_name = 'ckucb_gap_vs_final_regret'
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = os.path.join('runs', exp_name, ts)
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'experiment.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)
logging.info(f"Experiment: {exp_name}, T={T}, repeats={repeats}")

# Storage for final regrets
mean_finals = []
ci_finals = []
all_time_series = []

# ================= Loop over gaps =================
with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", 
                        dtype=torch.bfloat16):
    for gap in tqdm(gaps, desc='Gaps'):
        logging.info(f"Starting gap={gap:.3f}")

        # For each cluster, use same theta within cluster, only cluster 0 gap varies
        theta_array = [
            0.5 + gap/2,  # cluster 0 mean
            0.5 - gap/2,  # cluster 1 mean
            0.5 - gap/2   # cluster 2 mean
        ]

        # Initialize environment
        env = HeterogeneousClusterEnvironment(
                num_arms, theta_array, 
                sigma_array, arm_dist_types,
                contexts
        )

        # Run CK-UCB across replicates
        finals = []
        time_series_runs = []
        for rep in range(repeats):
            logging.info(f"  Replication {rep+1}/{repeats}")
            regret = run_ck_ucb(env, T, rep)  # returns length-T regret array
            cumreg = np.cumsum(regret)
            finals.append(cumreg[-1])
            time_series_runs.append(cumreg)

        finals = np.array(finals)
        mean_final = finals.mean()
        sem_final = finals.std(ddof=1) / np.sqrt(repeats)
        ci = 1.96 * sem_final

        mean_finals.append(mean_final)
        ci_finals.append(ci)
        all_time_series.append(np.vstack(time_series_runs))

        logging.info(
            f"Gap {gap:.3f}: mean final regret={mean_final:.4f}, CI=±{ci:.4f}"
        )

# ================= Plot 1: final regret vs gap =================
plt.figure(figsize=(10,6))
mean_finals = np.array(mean_finals)
ci_finals = np.array(ci_finals)
plt.errorbar(gaps, mean_finals, yerr=ci_finals, fmt='-o')
plt.xlabel('Suboptimality gap')
plt.ylabel(f'Final cumulative regret (T={T})')
plt.title('Final regret vs suboptimality gap')
plt.tight_layout()
out1 = os.path.join(log_dir, 'final_vs_gap.png')
plt.savefig(out1)
logging.info(f"Saved final-vs-gap plot to {out1}")

# ================= Plot 2: normalized final regret vs gap =================
plt.figure(figsize=(10,6))
normalized_mean = mean_finals / np.array(gaps)
normalized_ci = ci_finals / np.array(gaps)
plt.errorbar(gaps, normalized_mean, yerr=normalized_ci, fmt='-o')
plt.xlabel('Suboptimality gap')
plt.ylabel(f'Final cumulative regret normalized by gap (T={T})')
plt.title('Normalized final regret vs suboptimality gap')
plt.tight_layout()
out_norm = os.path.join(log_dir, 'final_vs_gap_normalized.png')
plt.savefig(out_norm)
logging.info(f"Saved normalized final-vs-gap plot to {out_norm}")

# ================= Plot 3: time-series for each gap =================
colors = plt.cm.viridis(np.linspace(0, 1, len(gaps)))

plt.figure(figsize=(10,6))
for i, (gap, ts_runs) in enumerate(zip(gaps, all_time_series)):
    avg, ub, lb = stats(ts_runs)
    plt.plot(np.arange(1, T+1), avg, label=f'gap={gap:.2f}', color=colors[i])
    plt.fill_between(np.arange(1, T+1), lb, ub, alpha=0.1, color=colors[i])

plt.xlabel('Time')
plt.ylabel('Cumulative regret')
plt.title('CK-UCB learning curves for different gaps')
plt.legend(loc='upper left', fontsize='small')
plt.tight_layout()
out2 = os.path.join(log_dir, 'curves_vs_gap.png')

plt.savefig(out2)
logging.info(f"Saved time-series vs-gap plot to {out2}")

# ================= Plot 4: normalized time-series regret vs gap =================
plt.figure(figsize=(10,6))
for i, (gap, ts_runs) in enumerate(zip(gaps, all_time_series)):
    # normalize each run by the gap
    normalized_runs = ts_runs / gap
    avg_norm, ub_norm, lb_norm = stats(normalized_runs)
    x = np.arange(1, T+1)
    plt.plot(x, avg_norm, label=f'gap={gap:.2f}', color=colors[i])
    plt.fill_between(x, lb_norm, ub_norm, alpha=0.1, color=colors[i])

plt.xlabel('Time')
plt.ylabel('Cumulative Regret / Gap')
plt.title('Normalized CK-UCB learning curves for different gaps')
#plt.title('Cumulative Regret Normalized by Gap')
plt.legend(loc='upper left', fontsize='small')
plt.tight_layout()
out3 = os.path.join(log_dir, 'curves_vs_gap_normalized_time.png')
plt.savefig(out3)
logging.info(f"Saved normalized time-series vs-gap plot to {out3}")
# ================= Save results dict =================
results_dict = {'gaps': gaps,
                'mean_final': np.array(mean_finals),
                'ci_final': np.array(ci_finals),
                'time_series': all_time_series}
with open(os.path.join(log_dir, 'gap_results.pkl'), 'wb') as f:
    pickle.dump(results_dict, f)
logging.info(f"Saved gap results to gap_results.pkl in {log_dir}")
