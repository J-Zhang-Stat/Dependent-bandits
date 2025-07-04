import random
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

import torch

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

    def __init__(self, num_arms, contexts, theta_array, sigma_array, arm_dist_types):

        self.num_cluster = len(num_arms)
        self.num_arms    = num_arms
        self.contexts    = contexts
        
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
        print(f"payoffs:{payoffs}")
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





class ClusteredKernelUCB:
    """
    A simple implementation of the Clustered Kernel-UCB algorithm (CK-UCB).
    Each cluster maintains its own kernel-ridge regressor to estimate the mean and uncertainty.
    """
    def __init__(self, num_arms, contexts, lambda_=1.0, gamma=1.0, B=1.0, sigma=1.0, delta=0.1):
        """
        :param contexts: np.ndarray, shape (n_arms, d) – feature vectors x_i for each arm
        :param num_arms: array-like, length n_arms – cluster assignment C(i) for each arm i e.g.([3,2,2] 3 clusters, 7 total)
        :param lambda_: float – regularization parameter (λ)
        :param gamma: float – bandwidth parameter for Gaussian kernel
        :param B: float – RKHS norm bound
        :param sigma: float – sub-Gaussian noise parameter
        :param delta: float – confidence level parameter
        """
        self.contexts = np.array(contexts) # (n_arms, d)
        self.clusters = []
        c_idx = 0
        for c in num_arms:
            self.clusters.extend([c_idx] * c)
            c_idx += 1
        self.clusters = np.array(self.clusters)
        assert len(self.contexts) == len(self.clusters), \
            "Contexts and clusters must have the same length"

        #print(f"Contexts shape: {self.contexts.shape}, Clusters length: {len(self.clusters)}")
        
        self.lambda_ = lambda_
        self.gamma = gamma
        self.B = B
        self.sigma = sigma
        self.delta = delta
        self.n_arms = len(contexts)

        self._kernel = self._gaussian_kernel  # Use Gaussian kernel by default
        
        # Initialize empty data structures for each cluster
        self.cluster_data = {}
        for c in np.unique(self.clusters):
            self.cluster_data[c] = {
                'X': [],   # list of past contexts in cluster c
                'r': []    # list of past rewards in cluster c
            }
        
    def _gaussian_kernel(self, x1, x2):
        """Gaussian (RBF) kernel."""
        return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * self.gamma**2))
    
    def select_arm(self, t):
        """
        Compute UCB scores for all arms at round t and return the arm with highest score.
        """
        ucb_scores = np.zeros(self.n_arms)

        mean_list = []
        var_list = []
        beta_t_list = []
        
        # for each arm
        prev_c = None
        for i in range(self.n_arms):
            cur_cluster = self.clusters[i]
            if cur_cluster != prev_c:
                # only update when cluster changes
                prev_c = cur_cluster
                #print(f"Processing arm {i} in cluster {self.clusters[i]} at time {t}")
                c = cur_cluster
                X_c = np.array(self.cluster_data[c]['X'])
                r_c = np.array(self.cluster_data[c]['r'])
                n_c = len(r_c)

                #print(f"Cluster {c} at time {t}: n_c={n_c}, contexts={X_c.shape}, rewards={r_c.shape}")

                # Build Gram matrix K_c and regularized G_c = K_c + λ I
                K_c = np.array([[self._kernel(x1, x2) for x2 in X_c] for x1 in X_c])
                G_c = K_c + self.lambda_ * np.eye(n_c)
                #G_inv = np.linalg.inv(G_c)  # TODO: better inversion method, such as: Cholesky decomposition, newtons method
                G_tensor = torch.tensor(G_c, dtype=torch.float64)
                #G_inv_tensor = torch.inverse(G_tensor)
                G_inv_tensor = torch.cholesky_inverse(torch.cholesky(G_tensor))
                G_inv = G_inv_tensor.numpy()

            #print(f"K_c shape: {K_c.shape}, G_c shape: {G_c.shape}, G_inv shape: {G_inv.shape}")
            
            # Kernel vector between x_i and past contexts in cluster
            k_i = np.array([self._kernel(x, self.contexts[i]) for x in X_c])
            
            # Predictive mean: k_i^T G_inv y_c
            mean = k_i.dot(G_inv).dot(r_c)
            
            mean_list.append(mean)
            
            # Predictive variance: K(x_i,x_i) - k_i^T G_inv k_i
            var = self._kernel(self.contexts[i], self.contexts[i]) - k_i.dot(G_inv).dot(k_i)
            var = max(var, 0)  # Ensure non-negative variance
            std = np.sqrt(var)
            
            var_list.append(std)
        
            # Beta_t
            log_det_term = np.log(1/self.delta * np.linalg.det(np.eye(n_c) + (1/self.lambda_) * K_c))
            beta_t = self.B + np.sqrt(2*self.lambda_*self.B**2 + 2*(self.sigma**2)*log_det_term)

            beta_t_list.append(beta_t)
            
            # UCB score
            ucb_scores[i] = mean + beta_t * std

        
        print(f"UCB at time {t}: {np.round(ucb_scores, 2)}")
        print(f"Means: {np.round(mean_list, 2)}")
        print(f"Stds: {np.round(var_list, 2)}")
        print(f"Beta_t: {np.round(beta_t_list, 2)}")
        print(f"Selected arm: {np.argmax(ucb_scores)}")
        
        return ucb_scores #int(np.argmax(ucb_scores))
    
    def update(self, arm, reward):
        """
        After pulling `arm` and observing `reward`, update the corresponding cluster's data.
        """
        c = self.clusters[arm]
        self.cluster_data[c]['X'].append(self.contexts[arm])
        self.cluster_data[c]['r'].append(reward)



def run_ck_ucb(env, T):

    learner = ClusteredKernelUCB(
        num_arms       = env.num_arms,  
        contexts       = env.contexts,  
        lambda_        = 0.1,  # Regularization parameter
        gamma          = 1.0,  # Bandwidth for Gaussian kernel
        B              = 1.0,  # RKHS norm bound
        sigma          = 1.0,  # Sub-Gaussian noise parameter
        delta          = 0.1,  # Confidence level parameter
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
        learner.update(i, r_)
        regret_list.append(delta[c_][a_])

    # Main loop
    for t in tqdm(range(total_arms, T)):
        ucb_vals   = learner.select_arm(t)
        chosen_arm = np.argmax(ucb_vals)
        c_, a_     = arm_to_cluster[chosen_arm]
        r_         = env.feedback(c_, a_)
        learner.update(chosen_arm, r_)
        regret_list.append(delta[c_][a_])

    return np.array(regret_list)



def run_experiments(env, T, repeat):

    ucb_results    = []
    oracle_results = []
    ck_ucb_results    = []

    for _ in (range(repeat)):
        ucb_results.append(run_ucb(env, T).cumsum())
        oracle_results.append(run_oracle_ucb(env, T).cumsum())
        ck_ucb_results.append(run_ck_ucb(env, T).cumsum())

    ucb_results    = np.array(ucb_results)
    oracle_results = np.array(oracle_results)
    ck_ucb_results = np.array(ck_ucb_results)

    def stats(arr):
        avg = arr.mean(axis=0)
        ub  = np.percentile(arr, 97.5, axis=0)
        lb  = np.percentile(arr, 2.5, axis=0)
        return avg, ub, lb

    return {
        'ucb'   : stats(ucb_results),
        'oracle': stats(oracle_results),
        'ck-ucb'   : stats(ck_ucb_results),
    }




# Environment configuration
num_arms     = [3, 2, 3]
contexts     = np.array([
                [1, 0.2, 0],
                [1, 0.4, 0],
                [1, 0.6, 0],
                
                [0, 0.5, 0.1],
                [0, 0.8, 0.2],
                
                [0.2, 0.8, 1],
                [0.3, 0.7, 1],
                [0.4, 0.6, 1],
                ])
theta_array  = [0.7, 0.5, 0.3]              # shared θ per cluster
sigma_array  = [1, 1, 1]                # σ for Gaussian rewards
arm_dist_types = [
    ['gaussian', 'bernoulli', 'exponential'],
    ['bernoulli', 'gaussian'],
    ['exponential', 'gaussian', 'bernoulli'],
]


env = HeterogeneousClusterEnvironment(num_arms, contexts,  
                                      theta_array, sigma_array,
                                      arm_dist_types)

T       = 2000
repeat  = 5
results = run_experiments(env, T, repeat)


plt.figure(figsize=(9, 6))
x_axis = np.arange(1, T + 1)


avg_ucb, ub_ucb, lb_ucb = results['ucb']
plt.plot(x_axis, avg_ucb, 'b-', label="Standard UCB")
plt.fill_between(x_axis, lb_ucb, ub_ucb, color='blue', alpha=0.15)

avg_oracle, ub_oracle, lb_oracle = results['oracle']
plt.plot(x_axis, avg_oracle, 'r-', label="UCB-C")
plt.fill_between(x_axis, lb_oracle, ub_oracle, color='red', alpha=0.15)

avg_bic, ub_bic, lb_bic = results['ck-ucb']
plt.plot(x_axis, avg_bic, 'g-', label="CK-UCB")
plt.fill_between(x_axis, lb_bic, ub_bic, color='green', alpha=0.15)

plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.title("3.3: Cumulative Regret of UCB, UCB-C, and CK-UCB")
plt.legend()
plt.tight_layout()
plt.show()
