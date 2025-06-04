import random
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict

class HeterogeneousCluster:
    """
    A cluster in which all arms share the same underlying parameter
    (self.theta, e.g., the mean/expectation) but each arm may follow
    a different reward‐distribution family: Gaussian, Exponential, or Bernoulli.
    """
    def __init__(self, num_arms, theta, arm_dist_types, sigma=1):
        self.num_arms       = num_arms
        self.theta          = theta
        self.sigma          = sigma             
        self.arm_dist_types = arm_dist_types   

    def feedback(self, arm):
        if arm >= self.num_arms:
            raise IndexError("Arm index out of range for this cluster")

        dist_type = self.arm_dist_types[arm]
        if dist_type == 'gaussian':
            # N(theta, sigma²)
            reward = random.gauss(self.theta, self.sigma)
        elif dist_type == 'exponential':
            # Exponential with mean = theta  ⇒  rate = 1 / theta
            reward = random.expovariate(1.0 / self.theta)
        elif dist_type == 'bernoulli':
            # Bernoulli with success-probability = theta
            reward = 1 if random.random() < self.theta else 0
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

        return reward

    def payoff(self):
        payoffs = []
        for dist_type in self.arm_dist_types:
            if dist_type in ('gaussian', 'exponential', 'bernoulli'):
                payoffs.append(self.theta)
        return np.array(payoffs)


class HeterogeneousClusterEnvironment:

    def __init__(self, num_arms, theta_array, arm_dist_types, sigma_array):

        self.num_cluster = len(num_arms)
        self.num_arms    = num_arms
        self.theta_array = theta_array
        self.sigma_array = sigma_array
        self.clusters    = []
        self.arm_dist_types = arm_dist_types

        for i in range(self.num_cluster):
            self.clusters.append(
                HeterogeneousCluster(
                    num_arms[i],
                    theta_array[i],
                    arm_dist_types[i],
                    sigma_array[i]
                )
            )


    def regret_per_action(self):

        payoffs = [c.payoff() for c in self.clusters]
        best    = np.max([np.max(p) for p in payoffs])
        delta   = [best - p for p in payoffs]
        return delta

    def feedback(self, cluster, arm):
        return self.clusters[cluster].feedback(arm)



class UCBLearner:
    def __init__(self, num_arms_total):
        self.num_arms = num_arms_total
        self.obs_avg  = np.zeros(num_arms_total)
        self.obs_cnt  = np.zeros(num_arms_total)

    def clear_obs(self):
        self.obs_avg.fill(0.0)
        self.obs_cnt.fill(0)

    def receive_feedback(self, arm_idx, reward):
        n = self.obs_cnt[arm_idx]
        self.obs_avg[arm_idx] = (self.obs_avg[arm_idx] * n + reward) / (n + 1)
        self.obs_cnt[arm_idx] += 1

    def get_ucb(self, t):
        bonus = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            if self.obs_cnt[i] == 0:
                bonus[i] = float('inf')
            else:
                bonus[i] = math.sqrt(2 * math.log(t + 1) / self.obs_cnt[i])
        return self.obs_avg + bonus


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




class OracleUCBLearner:

    def __init__(self, num_clusters, num_arms_per_cluster, arm_dist_types):
        self.num_clusters         = num_clusters
        self.num_arms_per_cluster = num_arms_per_cluster
        self.arm_dist_types       = arm_dist_types

        # Cluster-level stats
        self.cluster_data   = [[] for _ in range(num_clusters)]
        self.cluster_counts = np.zeros(num_clusters)
        self.cluster_means  = np.zeros(num_clusters)

        # Arm-level stats
        self.arm_counts = defaultdict(int)
        self.arm_means  = defaultdict(float)

    def clear_obs(self):
        self.cluster_data   = [[] for _ in range(self.num_clusters)]
        self.cluster_counts = np.zeros(self.num_clusters)
        self.cluster_means  = np.zeros(self.num_clusters)
        self.arm_counts.clear()
        self.arm_means.clear()

    def receive_feedback(self, cluster_id, arm_id, reward):
        # ------- update cluster statistics -------
        self.cluster_data[cluster_id].append(reward)
        self.cluster_counts[cluster_id] += 1
        self.cluster_means[cluster_id]   = np.mean(self.cluster_data[cluster_id])

        # ------- update arm statistics -------
        key      = (cluster_id, arm_id)
        n        = self.arm_counts[key]
        new_mean = (self.arm_means[key] * n + reward) / (n + 1)
        self.arm_means[key] = new_mean
        self.arm_counts[key] = n + 1

    def get_ucb(self, t):

        ucb_vals = []
        for c in range(self.num_clusters):
            n_c          = max(1, self.cluster_counts[c])
            cluster_mean = self.cluster_means[c]
            cluster_bonus = math.sqrt(2 * math.log(t + 1) / n_c)

            for a in range(self.num_arms_per_cluster[c]):
                key = (c, a)
                n_a = max(1, self.arm_counts[key])
                arm_mean  = self.arm_means[key]
                arm_bonus = math.sqrt(2 * math.log(t + 1) / n_a)
                bonus     = 0.5 * (cluster_bonus + arm_bonus)  # simple average
                ucb_vals.append(arm_mean + bonus)

        return np.array(ucb_vals)


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




class SubGaussianCandidate:
    def fit_MLE(self, data):     raise NotImplementedError
    def loglik(self, data):      raise NotImplementedError
    def get_mean(self):          raise NotImplementedError
    def get_num_params(self):    raise NotImplementedError


class GaussianFixedVarCandidate(SubGaussianCandidate):
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.mu_hat = 0.5
        self.n_params = 1
    def fit_MLE(self, data):
        self.mu_hat = 0.5 if len(data) == 0 else np.mean(data)
    def loglik(self, data):
        n = len(data)
        if n == 0: return 0.0
        cst = -0.5 * n * math.log(2 * math.pi * self.sigma ** 2)
        sq  = -np.sum((data - self.mu_hat) ** 2) / (2 * self.sigma ** 2)
        return cst + sq
    def get_mean(self):
        return self.mu_hat
    def get_num_params(self):
        return self.n_params


class ExponentialCandidate(SubGaussianCandidate):
    """Exponential family parameterized by λ (rate)."""
    def __init__(self):
        self.lam_hat = 1.0
        self.n_params = 1
    def fit_MLE(self, data):
        n = len(data)
        if n == 0:
            self.lam_hat = 1.0
        else:
            s = max(np.sum(data), 1e-12)
            self.lam_hat = n / s
    def loglik(self, data):
        n = len(data)
        if n == 0: return 0.0
        lam = self.lam_hat
        return n * math.log(lam) - lam * np.sum(data)
    def get_mean(self):
        return 1.0 / self.lam_hat
    def get_num_params(self):
        return self.n_params


class BernoulliCandidate(SubGaussianCandidate):
    """Bernoulli family with parameter p."""
    def __init__(self):
        self.p_hat = 0.5
        self.n_params = 1
    def fit_MLE(self, data):
        n = len(data)
        if n == 0:
            self.p_hat = 0.5
        else:
            mean = np.mean(data)
            self.p_hat = min(max(mean, 1e-12), 1 - 1e-12)
    def loglik(self, data):
        n = len(data)
        if n == 0: return 0.0
        p = self.p_hat
        return np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))
    def get_mean(self):
        return self.p_hat
    def get_num_params(self):
        return self.n_params


CANDIDATE_REGISTRY = {
    'gaussian':   GaussianFixedVarCandidate,
    'exponential': ExponentialCandidate,
    'bernoulli':  BernoulliCandidate,
}



class BICUCBLearner:

    def __init__(self, num_clusters, num_arms_per_cluster,
                 dist_family=('gaussian', 'exponential', 'bernoulli'),
                 sigma=1.0, softmax_temp=2.0):

        self.num_clusters         = num_clusters
        self.num_arms_per_cluster = num_arms_per_cluster
        self.dist_family          = dist_family
        self.sigma                = sigma
        self.temp                 = softmax_temp

        # Mapping: cluster ↔ global arms
        self.cluster_to_arms = {}
        self.arm_to_cluster  = {}
        idx = 0
        for c in range(num_clusters):
            self.cluster_to_arms[c] = []
            for _ in range(num_arms_per_cluster[c]):
                self.cluster_to_arms[c].append(idx)
                self.arm_to_cluster[idx] = c
                idx += 1

        # Cluster-level data
        self.cluster_data   = [[] for _ in range(num_clusters)]
        self.cluster_counts = np.zeros(num_clusters, dtype=int)

        self.cluster_candidates = []
        self.cluster_bic_scores = []
        self.cluster_weights    = []
        for _ in range(num_clusters):
            cands = []
            for dist_name in dist_family:
                if dist_name == 'gaussian':
                    cands.append(GaussianFixedVarCandidate(sigma=sigma))
                elif dist_name == 'exponential':
                    cands.append(ExponentialCandidate())
                elif dist_name == 'bernoulli':
                    cands.append(BernoulliCandidate())
            self.cluster_candidates.append(cands)
            self.cluster_bic_scores.append([0.0] * len(dist_family))
            self.cluster_weights.append([1.0 / len(dist_family)] * len(dist_family))

        total_arms          = sum(num_arms_per_cluster)
        self.arm_counts     = np.zeros(total_arms, dtype=int)


    def clear_obs(self):
        self.cluster_data   = [[] for _ in range(self.num_clusters)]
        self.cluster_counts = np.zeros(self.num_clusters, dtype=int)
        self.arm_counts.fill(0)

        for i in range(self.num_clusters):
            for cand in self.cluster_candidates[i]:
                cand.fit_MLE([])
            self.cluster_bic_scores[i] = [0.0] * len(self.dist_family)
            self.cluster_weights[i]    = [1.0 / len(self.dist_family)] * len(self.dist_family)

    def receive_feedback(self, global_arm_idx, reward):
        c = self.arm_to_cluster[global_arm_idx]
        self.cluster_data[c].append(reward)
        self.cluster_counts[c] += 1
        self.arm_counts[global_arm_idx] += 1

        self._update_cluster_distributions(c)

    def _update_cluster_distributions(self, c):
        data = np.array(self.cluster_data[c])
        n    = len(data)
        if n == 0:
            return

        bic_vals = []
        for i, cand in enumerate(self.cluster_candidates[c]):
            cand.fit_MLE(data)
            ll  = cand.loglik(data)
            k   = cand.get_num_params()
            bic = -2 * ll + k * math.log(n)
            self.cluster_bic_scores[c][i] = bic
            bic_vals.append(bic)

        # Softmax over (−ΔBIC) for weight assignment
        arr_bic = np.array(bic_vals)
        min_bic = arr_bic.min()
        delta   = arr_bic - min_bic
        w       = np.exp(-self.temp * delta)
        w_sum   = np.sum(w)
        self.cluster_weights[c] = w / w_sum if w_sum > 1e-12 else np.ones_like(w) / len(w)



    def get_ucb(self, t):
        total_arms = sum(self.num_arms_per_cluster)
        ucb_vals   = np.zeros(total_arms)

        for arm_idx in range(total_arms):
            c = self.arm_to_cluster[arm_idx]

            # Weighted mean across candidate models
            mean_est = sum(w * cand.get_mean() for w, cand
                           in zip(self.cluster_weights[c], self.cluster_candidates[c]))

            # Cluster-level bonus
            bonus = (float('inf') if self.arm_counts[arm_idx] == 0 else
                     math.sqrt(2.0 * math.log(t + 1) / (1 + self.cluster_counts[c])))

            ucb_vals[arm_idx] = mean_est + bonus

        return ucb_vals


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

    def stats(arr):
        avg = arr.mean(axis=0)
        ub  = np.percentile(arr, 90, axis=0)
        lb  = np.percentile(arr, 10, axis=0)
        return avg, ub, lb

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


env = HeterogeneousClusterEnvironment(num_arms, theta_array,
                                        arm_dist_types, sigma_array)

T       = 10000    
repeat  = 20        
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
plt.plot(x_axis, avg_bic, 'g-', label="BIC-UCB with diffrent f")
plt.fill_between(x_axis, lb_bic, ub_bic, color='green', alpha=0.15)

plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.title("3.2")
plt.legend()
plt.tight_layout()
plt.show()
