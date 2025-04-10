import random
import numpy as np
import math
import matplotlib.pyplot as plt

# Environment and base distribution classes

class NormalCluster:
    def __init__(self, num, theta, sigma=1):
        self.num = num
        self.theta = theta
        self.sigma = sigma
        self.scale = np.arange(num) + 1

    def feedback(self, arm):
        """Sample reward from arm."""
        if arm >= self.num:
            print("NormalCluster.feedback: index out of range")
            return None
        reward = random.gauss(self.theta * self.scale[arm], self.sigma)
        return reward

    def payoff(self):
        """Return true means for each arm."""
        return self.scale * self.theta


class NormalClusterEnvironment:
    """
    Multi-cluster environment of NormalCluster.
    """
    def __init__(self, num_arms, theta_array, sigma_array):
        if len(num_arms) != len(theta_array):
            print("Number of clusters does not match.")
        self.num_cluster = len(num_arms)
        self.num_arms = num_arms
        self.theta_array = theta_array
        self.sigma_array = sigma_array

        self.clusters = [
            NormalCluster(num_arms[i], theta_array[i], sigma_array[i])
            for i in range(self.num_cluster)
        ]

    def regret_per_action(self):
        """
        Regret = (max payoff among all cluster-arms) - (current cluster-arm payoff).
        """
        payoffs = [c.payoff() for c in self.clusters]
        largest = np.max([np.max(p) for p in payoffs])
        delta = [largest - p for p in payoffs]
        return delta

    def feedback(self, cluster, arm):
        """Return one reward sample from a cluster-arm."""
        return self.clusters[cluster].feedback(arm)




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
        var = self.sigma ** 2
        cst = -0.5 * n * math.log(2 * math.pi * var)
        sq = -np.sum((data - self.mu_hat) ** 2) / (2 * var)
        return cst + sq

    def get_mean(self):
        return self.mu_hat

    def get_num_params(self):
        return self.n_params


class ExponentialCandidate(SubGaussianCandidate):
    def __init__(self):
        self.lam_hat = 1.0
        self.n_params = 1

    def fit_MLE(self, data):
        n = len(data)
        if n == 0:
            self.lam_hat = 1.0
        else:
            s = np.sum(data)
            if s < 1e-12:
                s = 1e-12
            self.lam_hat = n / s

    def loglik(self, data):
        n = len(data)
        if n == 0:
            return 0.0
        lam = self.lam_hat
        return n * math.log(lam) - lam * np.sum(data)

    def get_mean(self):
        return 1.0 / self.lam_hat

    def get_num_params(self):
        return self.n_params


class BernoulliCandidate(SubGaussianCandidate):
    def __init__(self):
        self.p_hat = 0.5
        self.n_params = 1

    def fit_MLE(self, data):
        n = len(data)
        if n == 0:
            self.p_hat = 0.5
        else:
            m = np.mean(data)
            m = min(max(m, 1e-12), 1 - 1e-12)
            self.p_hat = m

    def loglik(self, data):
        n = len(data)
        if n == 0:
            return 0.0
        p = self.p_hat
        ll = np.sum(data * math.log(p) + (1 - data) * math.log(1 - p))
        return ll

    def get_mean(self):
        return self.p_hat

    def get_num_params(self):
        return self.n_params


CANDIDATE_REGISTRY = {
    'gaussian': GaussianFixedVarCandidate,
    'exponential': ExponentialCandidate,
    'bernoulli': BernoulliCandidate,
}




class DepCluster:
    """
    BIC-UCB cluster: use BIC to fit data, keep arm stats.
    """
    def __init__(self, cluster_id, sigma, dist_family):
        self.cluster_id = cluster_id
        self.sigma = sigma
        self.dist_family = dist_family

        self.data = []
        self.N = 0
        self.weights = {dist_name: 1.0 / len(dist_family) for dist_name in dist_family}
        self.best_distribution = None
        self.empirical_mean = 0.0

        self.arm_N = {}
        self.arm_mean = {}

    def reset(self):
        self.data.clear()
        self.N = 0
        self.weights = {dist_name: 1.0 / len(self.dist_family) for dist_name in self.dist_family}
        self.best_distribution = None
        self.empirical_mean = 0.0
        self.arm_N.clear()
        self.arm_mean.clear()

    def compute_BIC_scores(self, config):
        """Compute BIC for candidate distributions."""
        n = len(self.data)
        if n == 0:
            self.best_distribution = None
            self.empirical_mean = 0.0
            return

        data_arr = np.array(self.data)
        bic_map = {}

        for dist_name in self.dist_family:
            DistClass = CANDIDATE_REGISTRY[dist_name]
            if dist_name == 'gaussian':
                candidate = DistClass(sigma=config.sigma)
            else:
                candidate = DistClass()

            candidate.fit_MLE(data_arr)
            ll = candidate.loglik(data_arr)
            k = candidate.get_num_params()
            bic_val = -2 * ll + k * math.log(n)
            bic_map[dist_name] = (bic_val, candidate.get_mean())

        all_bic = np.array([val[0] for val in bic_map.values()])
        min_bic = np.min(all_bic)
        exp_val = np.exp(-4.993 * (all_bic - min_bic))

        if np.sum(exp_val) == 0 or np.isnan(exp_val).any() or np.isinf(exp_val).any():
            w_arr = np.ones_like(exp_val) / len(exp_val)
        else:
            w_arr = exp_val / np.sum(exp_val)

        dist_list = list(self.dist_family)
        for i, dist_name in enumerate(dist_list):
            self.weights[dist_name] = w_arr[i]

        best_dist_name, (best_bic, best_mean) = min(bic_map.items(), key=lambda x: x[1][0])
        self.best_distribution = best_dist_name
        self.empirical_mean = best_mean


class DependentBandit:
    """
    BIC-UCB bandit: 2-layer UCB (cluster, arm).
    """
    def __init__(self, config, environment):
        self.config = config
        self.env = environment
        self.num_cluster = environment.num_cluster

        self.clusters = [
            DepCluster(cluster_id=c_id, sigma=self.config.sigma, dist_family=self.config.F)
            for c_id in range(self.num_cluster)
        ]
        self.delta = self.env.regret_per_action()
        self.total_regret = 0.0
        self.regret_history = []

    def reset(self):
        for c in self.clusters:
            c.reset()
        self.total_regret = 0.0
        self.regret_history = []

    def run_one_trial(self, T):
        """Run T steps, return cumulative regret."""
        for t in range(1, T + 1):
            for c in self.clusters:
                c.compute_BIC_scores(self.config)

            chosen_cluster = self.select_cluster(t)
            chosen_arm = self.select_arm_in_cluster(chosen_cluster, t)

            r = self.env.feedback(chosen_cluster.cluster_id, chosen_arm)
            chosen_cluster.data.append(r)
            chosen_cluster.N += 1

            old_cnt = chosen_cluster.arm_N.get(chosen_arm, 0)
            chosen_cluster.arm_N[chosen_arm] = old_cnt + 1
            old_mean = chosen_cluster.arm_mean.get(chosen_arm, 0.0)
            new_cnt = chosen_cluster.arm_N[chosen_arm]
            chosen_cluster.arm_mean[chosen_arm] = (old_mean * (new_cnt - 1) + r) / new_cnt

            inst_r = self.delta[chosen_cluster.cluster_id][chosen_arm]
            self.total_regret += inst_r
            self.regret_history.append(self.total_regret)

        return np.array(self.regret_history)

    def select_cluster(self, t):
        """
        Cluster-level UCB.
        If N=0, pick it first.
        """
        best_c = None
        best_val = -1e9
        for c in self.clusters:
            if c.N == 0:
                return c
            avg_w = np.mean(list(c.weights.values()))
            bonus = math.sqrt(1.7 * math.log(t + 1) / c.N) * (1 - avg_w)
            val = c.empirical_mean + bonus
            if val > best_val:
                best_val = val
                best_c = c
        return best_c

    def select_arm_in_cluster(self, cluster, t):
        """
        Arm-level UCB.
        If an arm was never selected, pick it first.
        """
        best_arm = None
        best_val = -1e9
        c_id = cluster.cluster_id
        arm_cnt = self.env.num_arms[c_id]

        for arm_id in range(arm_cnt):
            n_arm = cluster.arm_N.get(arm_id, 0)
            if n_arm == 0:
                return arm_id
            mean_arm = cluster.arm_mean.get(arm_id, cluster.empirical_mean)
            bonus = math.sqrt(1.7 * math.log(t + 1) / n_arm)
            val = mean_arm + bonus
            if val > best_val:
                best_val = val
                best_arm = arm_id
        return best_arm


# Main: Compare arms in one cluster

if __name__ == "__main__":
    base_num_arms = [2, 2, 3, 2, 3]
    theta_array = [0.1, 0.5, 0.2, 0.3, 0.1]
    sigma_array = [1, 1, 1, 1, 1]

    arms_candidate = [2, 4, 6, 8, 10]
    T = 3000
    repeat = 50

    class Config: 
        pass

    config = Config()
    config.F = ['gaussian', 'exponential', 'bernoulli']
    config.sigma = 1.0

    plt.figure(figsize=(10, 6))
    
    for arms_val in arms_candidate:
        new_num_arms = base_num_arms.copy()
        new_num_arms[1] = arms_val

        cum_reg_list = []
        for _ in range(repeat):
            env = NormalClusterEnvironment(new_num_arms, theta_array, sigma_array)
            bandit = DependentBandit(config, env)
            reg_arr = bandit.run_one_trial(T)
            cum_reg_list.append(reg_arr)

        cum_reg_list = np.array(cum_reg_list)
        avg_reg = cum_reg_list.mean(axis=0)

        plt.plot(range(T), avg_reg, label=f'Cluster[1] arms={arms_val}')

    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('BIC-UCB: Increasing Arms in One Cluster')
    plt.legend()
    plt.tight_layout()
    plt.show()
