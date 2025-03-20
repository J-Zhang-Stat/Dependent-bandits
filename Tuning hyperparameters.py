import math
import random
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
class NormalCluster:

    def __init__(self, num, theta, sigma=1):
        self.num = num
        self.theta = theta
        self.sigma = sigma
        self.scale = np.array(range(num)) + 1  

    def feedback(self, arm):
        if arm >= self.num:
            raise ValueError("arm index out of range in NormalCluster")
        reward = random.gauss(self.theta * self.scale[arm], self.sigma)
        return reward

    def payoff(self):
        return self.scale * self.theta

class NormalClusterEnvironment:

    def __init__(self, num_arms, theta_array, sigma_array):

        self.num_cluster = len(num_arms)
        self.num_arms = num_arms
        self.theta_array = theta_array
        self.sigma_array = sigma_array
        self.clusters = []
        for i in range(self.num_cluster):
            self.clusters.append(NormalCluster(
                num_arms[i], theta_array[i], sigma_array[i])
            )

    def feedback(self, cluster_idx, arm_idx):
        return self.clusters[cluster_idx].feedback(arm_idx)

    def regret_per_action(self):

        payoffs = [c.payoff() for c in self.clusters]
  
        global_best = max(np.max(p) for p in payoffs)
        delta = []
        for p in payoffs:
            delta.append(global_best - p)
        return delta

class NormalClusterLearner:
        
    def __init__(self, num, sigma=1):
        self.num = num
        self.sigma = sigma
        self.obs_avg = np.zeros(num)
        self.obs_cnt = np.zeros(num, dtype=int)
        self.scale = np.array(range(num)) + 1

    def receive_feedback(self, arm, reward):
        # Update observed average
        self.obs_avg[arm] = (self.obs_avg[arm] * self.obs_cnt[arm] + reward) / (self.obs_cnt[arm] + 1)
        self.obs_cnt[arm] += 1

    def clear_obs(self):
        self.obs_avg.fill(0.0)
        self.obs_cnt.fill(0)

    def mle(self):
        # Simple example
        denom = np.sum(self.scale * self.obs_cnt)
        if denom <= 0:
            return 0.0
        return np.sum(self.obs_avg * self.obs_cnt) / denom

class NormalClusterTestBed:

    def __init__(self, num_arms, theta_array, sigma_array):
        if len(num_arms) != len(theta_array):
            raise ValueError("Number of clusters doesn't match size of theta_array")
        self.num_cluster = len(num_arms)
        self.num_arms = num_arms
        self.theta_array = theta_array
        self.sigma_array = sigma_array
        self.environment = NormalClusterEnvironment(num_arms, theta_array, sigma_array)
        self.learners = [NormalClusterLearner(num_arms[i], sigma_array[i]) for i in range(self.num_cluster)]
        self.delta = self.environment.regret_per_action()

    def reset_learners(self):
        for l in self.learners:
            l.clear_obs()

class SubGaussianCandidate:
    def fit_MLE(self, data):
        raise NotImplementedError
    def loglik(self, data):
        raise NotImplementedError
    def get_mean(self):
        raise NotImplementedError
    def get_num_params(self):
        raise NotImplementedError

class GaussianFixedVarCandidate(SubGaussianCandidate):
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.mu_hat = 0.0
        self.n_params = 1
    def fit_MLE(self, data):
        if len(data) == 0:
            self.mu_hat = 0.0
        else:
            self.mu_hat = np.mean(data)
    def loglik(self, data):
        n = len(data)
        if n == 0:
            return 0.0
        s = self.sigma
        cst = -0.5 * n * math.log(2 * math.pi * s**2)
        sq = - np.sum((data - self.mu_hat)**2) / (2 * s**2)
        return cst + sq
    def get_mean(self):
        return self.mu_hat
    def get_num_params(self):
        return self.n_params

CANDIDATE_REGISTRY = {
    'gaussian': GaussianFixedVarCandidate,
}


def run_bandit_experiment(k_i, k_j, bic_temp,
                          T=3000,
                          repeat=50,
                          num_arms=[3,2,3],
                          theta_array=[0.1,0.5,0.2],
                          sigma_array=[1,1,1]):
    class Config:
        pass
    config = Config()
    config.F = ['gaussian']
    config.sigma = 1.0
    config.k_i = k_i
    config.k_j = k_j
    config.bic_temp = bic_temp

    all_regrets = []
    for _ in range(repeat):
        testbed = NormalClusterTestBed(num_arms, theta_array, sigma_array)
        bandit = DependentBandit(config, testbed.environment)
        reg = bandit.run_one_trial(T)
        all_regrets.append(reg)

    all_regrets = np.array(all_regrets) 
    final_regret_mean = all_regrets[:, -1].mean()  # Mean cumulative regret at the final time step across multiple trials
    
    return -final_regret_mean

def tune_parameters_with_bo():

    pbounds = {
        'k_i': (0.1, 5.0),
        'k_j': (0.1, 5.0),
        'bic_temp': (0.5, 5)
    }
    optimizer = BayesianOptimization(
        f=run_bandit_experiment,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    optimizer.maximize(init_points=5, n_iter=50)
    print("Bayesian Optimization best result: ", optimizer.max)
    return optimizer.max

if __name__ == "__main__":
    best_result = tune_parameters_with_bo()
    print("Optimal (k_i, k_j, bic_temp) = ", best_result["params"])
    best_params = best_result["params"]
    final_score = run_bandit_experiment(**best_params, T=3000, repeat=50)
    print(f"Mean cumulative regret with optimal parameters = {-final_score:.4f}")
