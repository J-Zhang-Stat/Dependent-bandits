import random
import numpy as np
import math
import matplotlib.pyplot as plt
class NormalCluster:
    def __init__(self, num, theta, sigma =1):
        self.num = num
        self.theta = theta
        self.sigma = sigma
        self.scale = np.array(range(num))+1
        
    def feedback(self, arm):
        if arm >= self.num:
            print("NormalCluster.feedback: index out of num_arms within the cluster")
            return None
        reward = random.gauss(self.theta * self.scale[arm], self.sigma)
        return reward
    
    def payoff(self):
        return self.scale * self.theta

class NormalClusterEnvironment:
    def __init__(self, num_arms, theta_array, sigma_array):
        if len(num_arms) != len(theta_array):
            print("num of clusters doesn't match")
        self.num_cluster = len(num_arms)
        self.theta_array = theta_array
        self.num_arms = num_arms
        self.sigma_array = sigma_array
        self.clusters = []
        for i in range(self.num_cluster):
            self.clusters.append(NormalCluster(num_arms[i],theta_array[i],sigma_array[i]))
    
    def regret_per_action(self):
        payoffs = [c.payoff() for c in self.clusters]
        largest = np.max([np.max(p) for p in payoffs])
        delta = [largest - p for p in payoffs]
        return delta
        
    def feedback(self, cluster, arm):
        return self.clusters[cluster].feedback(arm)


class NormalClusterLearner:
    def __init__(self, num, sigma=1):
        self.num = num
        self.sigma = sigma
        self.obs_avg = np.array([0.0 for _ in range(num)])
        self.obs_cnt = np.array([0 for _ in range(num)])
        self.scale = np.array(range(num)) + 1
        
    def receive_feedback(self, arm, reward):
        if arm >= self.num:
            print("NormalClusterLearner.receive_feedback: index out of num_arms within the cluster")
            return
        self.obs_avg[arm] = (self.obs_avg[arm] * self.obs_cnt[arm] + reward)/(self.obs_cnt[arm] + 1)
        self.obs_cnt[arm] += 1
        
    def clear_obs(self):
        self.obs_avg = np.array([0.0 for _ in range(self.num)])
        self.obs_cnt = np.array([0 for _ in range(self.num)])
        
    def mle(self):
        denom = np.sum(self.scale * self.obs_cnt)
        if denom <= 0:
            return 0.0
        return np.sum(self.obs_avg * self.obs_cnt) / denom
    
    def explore_bonus(self, t):
        return np.log(t+1)  
    
    def ucb(self, t):
        bonus = np.zeros(self.num)
        for i in range(self.num):
            if self.obs_cnt[i] > 0:
                bonus[i] = np.sqrt(2*self.sigma*self.sigma*self.explore_bonus(t)/self.obs_cnt[i])
            else:
                bonus[i] = float('inf')
        return self.obs_avg + bonus
    
    def ucb_c(self, t):
        est = self.mle()
        sum_cnt_scale2 = np.sum(self.obs_cnt * (self.scale**2))
        if sum_cnt_scale2 <= 0:
            ci = float('inf')
        else:
            ci = np.sqrt(2*self.sigma*self.sigma*self.explore_bonus(t)/sum_cnt_scale2)
        upb = est + ci
        return upb * self.scale

class NormalClusterTestBed:
    def __init__(self, num_arms, theta_array, sigma_array):
        if len(num_arms) != len(theta_array):
            print("num of clusters doesn't match")
        self.num_cluster = len(num_arms)
        self.theta_array = theta_array
        self.num_arms = num_arms
        self.sigma_array = sigma_array
        self.environment = NormalClusterEnvironment(num_arms, theta_array, sigma_array)
        self.learners = [NormalClusterLearner(num_arms[i], sigma_array[i]) 
                         for i in range(self.num_cluster)]
        self.delta = self.environment.regret_per_action()
        
    def reset_learners(self):
        for l in self.learners:
            l.clear_obs()
            
    def ucb(self, T):
        regret = []
        #init
        for c in range(self.num_cluster):
            for arm in range(self.num_arms[c]):
                reward = self.environment.feedback(c, arm)
                self.learners[c].receive_feedback(arm, reward)
                regret.append(self.delta[c][arm])
        for t in range(sum(self.num_arms), T):
            ucbs = [cl.ucb(t) for cl in self.learners]
            largests = [np.max(ucb) for ucb in ucbs]
            dec_c = np.argmax(largests)
            dec_arm = np.argmax(ucbs[dec_c])
            reward = self.environment.feedback(dec_c, dec_arm)
            self.learners[dec_c].receive_feedback(dec_arm, reward)
            regret.append(self.delta[dec_c][dec_arm])
        return np.array(regret)
    
    def ucb_c(self, T):
        regret = []
        # init
        for c in range(self.num_cluster):
            arm = self.num_arms[c]-1
            reward = self.environment.feedback(c, arm)
            self.learners[c].receive_feedback(arm, reward)
            regret.append(self.delta[c][arm])
        for t in range(self.num_cluster, T):
            ucbs = [cl.ucb_c(t) for cl in self.learners]
            largest_vals = [np.max(ucb) for ucb in ucbs]
            dec_c = np.argmax(largest_vals)
            dec_arm = np.argmax(ucbs[dec_c])
            reward = self.environment.feedback(dec_c, dec_arm)
            self.learners[dec_c].receive_feedback(dec_arm, reward)
            regret.append(self.delta[dec_c][dec_arm])
        return np.array(regret)
    
    
    def ucb_perf_percentile(self, T, repeat):
        all_regrets = []
        for _ in range(repeat):
            self.reset_learners()
            r = self.ucb(T).cumsum()
            all_regrets.append(r)
        all_regrets = np.array(all_regrets)
        avg_regret = all_regrets.mean(axis=0)
        ub_regret = np.percentile(all_regrets, 90, axis=0)
        lb_regret = np.percentile(all_regrets, 10, axis=0)
        return avg_regret, ub_regret, lb_regret

    def ucb_c_perf_percentile(self, T, repeat):
        all_regrets = []
        for _ in range(repeat):
            self.reset_learners()
            r = self.ucb_c(T).cumsum()
            all_regrets.append(r)
        all_regrets = np.array(all_regrets)
        avg_regret = all_regrets.mean(axis=0)
        ub_regret = np.percentile(all_regrets, 90, axis=0)
        lb_regret = np.percentile(all_regrets, 10, axis=0)
        return avg_regret, ub_regret, lb_regret



class DependentBandit:
    def __init__(self, config, environment):
        self.config = config
        self.env = environment
        self.num_cluster = environment.num_cluster
        self.num_arms = sum(environment.num_arms)

        
        self.clusters = []
        for c_id in range(self.num_cluster):
            c_obj = DepCluster(
                cluster_id=c_id,
                sigma=self.config.sigma,
                dist_family=self.config.F
            )
            self.clusters.append(c_obj)

        self.delta = self.env.regret_per_action()  
        self.total_regret = 0.0
        self.regret_history = []

    def reset(self):
        for c in self.clusters:
            c.reset()
        self.total_regret = 0.0
        self.regret_history = []

    def run_one_trial(self, T):

        for t in range(1, T+1):
            
            for c in self.clusters:
                c.compute_BIC_scores(self.config)

            
            chosen_cluster = self.select_cluster(t)
            
            chosen_arm = self.select_arm_in_cluster(chosen_cluster, t)
            
            r = self.env.feedback(chosen_cluster.cluster_id, chosen_arm)
            chosen_cluster.data.append(r)
            chosen_cluster.N += 1
            chosen_cluster.arm_N[chosen_arm] = chosen_cluster.arm_N.get(chosen_arm,0)+1
        
            old_mean = chosen_cluster.arm_mean.get(chosen_arm,0.0)
            ccount = chosen_cluster.arm_N[chosen_arm]
            chosen_cluster.arm_mean[chosen_arm] = (old_mean*(ccount-1) + r)/ccount

            # regret
            inst_r = self.delta[chosen_cluster.cluster_id][chosen_arm]
            self.total_regret += inst_r
            self.regret_history.append(self.total_regret)

        return np.array(self.regret_history)

    def select_cluster(self, t):
        best_c = None
        best_val = -1e9
        for c in self.clusters:
            if c.N == 0:
                return c
            avg_w = np.mean(list(c.weights.values()))
            expl = math.sqrt(1.7*math.log(t+1)/c.N)*(1-avg_w)
            # UCB: c.empirical_mean + expl
            val = c.empirical_mean + expl
            if val>best_val:
                best_val = val
                best_c = c
        return best_c

    def select_arm_in_cluster(self, cluster, t):
        best_arm = None
        best_val = -1e9
        c_id = cluster.cluster_id
        arm_cnt = self.env.num_arms[c_id]

        for arm_id in range(arm_cnt):
            n_arm = cluster.arm_N.get(arm_id,0)
            mean_arm = cluster.arm_mean.get(arm_id, cluster.empirical_mean)
            if n_arm==0:
                return arm_id
            bonus = math.sqrt(1.7*math.log(t+1)/n_arm)
            val = mean_arm + bonus
            if val>best_val:
                best_val = val
                best_arm = arm_id
        return best_arm
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
        """
        data: np.array([...])
        """
        if len(data)==0:
            self.mu_hat = 0.0
        else:
            self.mu_hat = np.mean(data)
    
    def loglik(self, data):
        n = len(data)
        if n==0:
            return 0.0
        s = self.sigma
        cst = -0.5*n*math.log(2*math.pi*s*s)
        sq = - np.sum((data - self.mu_hat)**2)/(2*s*s)
        ll = cst + sq
        return ll
    
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
        if n==0:
            self.lam_hat = 1.0
        else:
            sum_data = np.sum(data)
            if sum_data < 1e-12:
                sum_data = 1e-12
            self.lam_hat = n / sum_data

    def loglik(self, data):
        n = len(data)
        if n==0:
            return 0.0
        lam = self.lam_hat
        return n*math.log(lam) - lam*np.sum(data)

    def get_mean(self):
        return 1.0 / self.lam_hat
    
    def get_num_params(self):
        return self.n_params

class BernoulliCandidate(SubGaussianCandidate):
    def __init__(self):
        self.p_hat = 0.5
        self.n_params = 1

    def fit_MLE(self, data):
        if len(data)==0:
            self.p_hat = 0.5
        else:
            mean_ = np.mean(data)
            
            mean_ = min(max(mean_,1e-12),1-1e-12)
            self.p_hat = mean_

    def loglik(self, data):
        n = len(data)
        if n==0:
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
    # to be continued
}


class DepCluster:

    def __init__(self, cluster_id, sigma, dist_family):
        self.cluster_id = cluster_id
        self.sigma = sigma
        self.dist_family = dist_family

        self.data = []
        self.N = 0
        self.weights = {dist_name: 1.0/len(dist_family) for dist_name in dist_family}
        self.best_distribution = None
        self.empirical_mean = 0.0

        self.arm_N = {}
        self.arm_mean = {}

    def reset(self):
        self.data.clear()
        self.N = 0
        self.weights = {dist_name: 1.0/len(self.dist_family) for dist_name in self.dist_family}
        self.best_distribution = None
        self.empirical_mean = 0.0
        self.arm_N.clear()
        self.arm_mean.clear()

    def compute_BIC_scores(self, config):
   
        n = len(self.data)
        if n == 0:
            self.best_distribution = None
            self.empirical_mean = 0.0
            return

        data_arr = np.array(self.data)

        
        bic_map = {}

        for dist_name in self.dist_family:
            
            DistClass = CANDIDATE_REGISTRY[dist_name]
            if dist_name=='gaussian':
                candidate = DistClass(sigma=config.sigma)
            else:
                candidate = DistClass()

            candidate.fit_MLE(data_arr)
            
            ll = candidate.loglik(data_arr)
            k = candidate.get_num_params()
            bic_val = -2*ll + k*math.log(n)
    
            bic_map[dist_name] = (bic_val, candidate.get_mean())

        all_bic = np.array([val[0] for val in bic_map.values()])
        min_bic = np.min(all_bic)
        exp_val = np.exp(-4.993*(all_bic - min_bic))
        if np.sum(exp_val)==0 or np.isnan(exp_val).any() or np.isinf(exp_val).any():
            w_arr = np.ones_like(exp_val)/len(exp_val)
        else:
            w_arr = exp_val/np.sum(exp_val)

       
        dist_list = list(self.dist_family)  # e.g. ['gaussian', 'exponential', 'bernoulli']
        for i, dist_name in enumerate(dist_list):
            self.weights[dist_name] = w_arr[i]

        
        best_dist_name, (best_bic, best_mean) = min(bic_map.items(), key=lambda x: x[1][0])
        self.best_distribution = best_dist_name
        self.empirical_mean = best_mean



if __name__=="__main__":
    
    num_arms = [3,2,3,2,4]
    theta_array = [0.1, 0.5, 0.2, 0.3, 0.1]
    sigma_array = [1,1,1,1,1]

    
    T = 5000
    repeat = 50
    plt_sample = range(1,T+1,int(T/20))

    # 1) UCB & UCB-C with NormalClusterTestBed
    testbed = NormalClusterTestBed(num_arms, theta_array, sigma_array)
    avg_regret_ucb, ub_regret_ucb, lb_regret_ucb = testbed.ucb_perf_percentile(T, repeat)
    avg_regret_ucbc, ub_regret_ucbc, lb_regret_ucbc = testbed.ucb_c_perf_percentile(T, repeat)

    # 2) DependentBandit(BIC-UCB)
    class Config:
        pass
    config = Config()
    config.F = ['gaussian','exponential','bernoulli']  # candidate distributions
    config.sigma = 1.0

    
    all_reg_bic = []
    for _ in range(repeat):
        bandit = DependentBandit(config, testbed.environment)
        reg = bandit.run_one_trial(T)
        all_reg_bic.append(reg)
    all_reg_bic = np.array(all_reg_bic)
    avg_regret_bic = all_reg_bic.mean(axis=0)
    ub_regret_bic = np.percentile(all_reg_bic, 90, axis=0)
    lb_regret_bic = np.percentile(all_reg_bic, 10, axis=0)

    
    # UCB
    plt.plot(plt_sample, [avg_regret_ucb[i-1] for i in plt_sample], 'b-', label='UCB')
    plt.fill_between(np.arange(T), ub_regret_ucb, lb_regret_ucb, color='blue', alpha=0.2)

    # UCB-C
    plt.plot(plt_sample, [avg_regret_ucbc[i-1] for i in plt_sample], 'r-', label='UCB-C')
    plt.fill_between(np.arange(T), ub_regret_ucbc, lb_regret_ucbc, color='red', alpha=0.2)

    # BIC-UCB 
    plt.plot(plt_sample, [avg_regret_bic[i-1] for i in plt_sample], 'g-', label='BIC-UCB')
    plt.fill_between(np.arange(T), ub_regret_bic, lb_regret_bic, color='green', alpha=0.2)

    plt.legend(fontsize=12)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Cumulative Regret', fontsize=14)
    plt.title('UCB vs UCB-C vs BIC-UCB', fontsize=14)
    plt.tight_layout()
    plt.show()
