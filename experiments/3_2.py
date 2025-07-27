import numpy as np
import matplotlib.pyplot as plt

from learners import UCBLearner, OracleUCBLearner, BICUCBLearner
from environment.candidates import GaussianFixedVarCandidate, ExponentialCandidate, BernoulliCandidate
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



CANDIDATE_REGISTRY = {
    'gaussian':   GaussianFixedVarCandidate,
    'exponential': ExponentialCandidate,
    'bernoulli':  BernoulliCandidate,
}



def run_bic_ucb(env, T):

    learner = BICUCBLearner(
        num_clusters        = env.num_cluster,
        num_arms_per_cluster= env.num_arms,
        dist_family         = ['gaussian', 'exponential', 'bernoulli'],
        sigma               = 1.0,
        softmax_temp        = 2.0
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
        learner.receive_feedback(i, r_)
        regret_list.append(delta[c_][a_])

    # Main loop
    for t in range(total_arms, T):
        ucb_vals   = learner.get_ucb(t)
        chosen_arm = np.argmax(ucb_vals)
        c_, a_     = arm_to_cluster[chosen_arm]
        r_         = env.feedback(c_, a_)
        learner.receive_feedback(chosen_arm, r_)
        regret_list.append(delta[c_][a_])

    return np.array(regret_list)



def run_experiments(env, T, repeat):

    ucb_results    = []
    oracle_results = []
    bic_results    = []

    for _ in range(repeat):
        ucb_results.append(run_ucb(env, T).cumsum())
        oracle_results.append(run_oracle_ucb(env, T).cumsum())
        bic_results.append(run_bic_ucb(env, T).cumsum())

    ucb_results    = np.array(ucb_results)
    oracle_results = np.array(oracle_results)
    bic_results    = np.array(bic_results)


    return {
        'ucb'   : stats(ucb_results),
        'oracle': stats(oracle_results),
        'bic'   : stats(bic_results),
    }





# Environment configuration
num_arms     = [3, 2, 3, 3]                      
theta_array  = [0.7, 0.5, 0.2, 0.3]               # shared θ per cluster
sigma_array  = [1, 1, 1, 1]                       # σ for Gaussian rewards
arm_dist_types = [
    ['gaussian', 'bernoulli', 'exponential'],
    ['bernoulli', 'gaussian'],
    ['exponential', 'gaussian', 'bernoulli'],
    ['gaussian', 'exponential', 'gaussian']
]


env = HeterogeneousClusterEnvironment(num_arms, theta_array, sigma_array,
                                        arm_dist_types, contexts=None)

T       = 10000    
repeat  = 50        
results = run_experiments(env, T, repeat)


plt.figure(figsize=(9, 6))
x_axis = np.arange(1, T + 1)


avg_ucb, ub_ucb, lb_ucb = results['ucb']
plt.plot(x_axis, avg_ucb, 'b-', label="Standard UCB")
plt.fill_between(x_axis, lb_ucb, ub_ucb, color='blue', alpha=0.15)

avg_oracle, ub_oracle, lb_oracle = results['oracle']
plt.plot(x_axis, avg_oracle, 'r-', label="UCB-C")
plt.fill_between(x_axis, lb_oracle, ub_oracle, color='red', alpha=0.15)

avg_bic, ub_bic, lb_bic = results['bic']
plt.plot(x_axis, avg_bic, 'g-', label="BIC-UCB w/ different f")
plt.fill_between(x_axis, lb_bic, ub_bic, color='green', alpha=0.15)

plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.title("Section 3.2")
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig("runs/3.2_plot.png")