import numpy as np
import math
import matplotlib.pyplot as plt
import logging

from environment.candidates import GaussianFixedVarCandidate, ExponentialCandidate, BernoulliCandidate
from environment.normal_cluster import NormalClusterEnvironment
from experiments.utils import stats


class NormalClusterLearner:
    def __init__(self, num, sigma=1):
        self.num = num
        self.sigma = sigma
        self.obs_avg = np.array([0.0 for _ in range(num)])
        self.obs_cnt = np.array([0 for _ in range(num)])
        self.scale = np.array(range(num)) + 1
        
    def receive_feedback(self, arm, reward):
        if arm >= self.num:
            logging.warning("NormalClusterLearner.receive_feedback: index out of num_arms within the cluster")
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
            logging.error("num of clusters doesn't match")
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
        sem = all_regrets.std(axis=0, ddof=1) / np.sqrt(all_regrets.shape[0])
        z = 1.96
        lb_regret = avg_regret - z * sem
        ub_regret = avg_regret + z * sem        

        return avg_regret, ub_regret, lb_regret
    

    def ucb_c_perf_percentile(self, T, repeat):
        all_regrets = []
        for _ in range(repeat):
            self.reset_learners()
            r = self.ucb_c(T).cumsum()
            all_regrets.append(r)
        all_regrets = np.array(all_regrets)
        avg_regret = all_regrets.mean(axis=0)
        sem = all_regrets.std(axis=0, ddof=1) / np.sqrt(all_regrets.shape[0])
        z = 1.96
        lb_regret = avg_regret - z * sem
        ub_regret = avg_regret + z * sem
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
            
            # TODO:DEBUG
            logging.info(f" t={t}, cluster={c.cluster_id}, N={c.N}, emp_mean={c.empirical_mean:.4f}, "
                  f"avg_w={avg_w:.4f}, expl={expl:.4f}, val={val:.4f}")
            
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
            logging.info(f"   arm={arm_id}, n_arm={n_arm}, mean_arm={mean_arm:.4f}, bonus={bonus:.4f}, val={val:.4f}")
            if val>best_val:
                best_val = val
                best_arm = arm_id
        return best_arm


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
        
        # TODO: DEBUG
        logging.info(f"[Cluster {self.cluster_id}] n={n}, bic_map={bic_map}")
        logging.info(f"[Cluster {self.cluster_id}] all_bic={all_bic}, min_bic={min_bic}")
        logging.info(f"[Cluster {self.cluster_id}] exp_val={exp_val}, sum_exp={np.sum(exp_val)}")
        
        if np.sum(exp_val)==0 or np.isnan(exp_val).any() or np.isinf(exp_val).any():
            w_arr = np.ones_like(exp_val)/len(exp_val)
        else:
            w_arr = exp_val/np.sum(exp_val)

       
        dist_list = list(self.dist_family)  # e.g. ['gaussian', 'exponential', 'bernoulli']
        for i, dist_name in enumerate(dist_list):
            self.weights[dist_name] = w_arr[i]

        #TODO:DEBUG
        logging.info(f"[Cluster {self.cluster_id}] weights={self.weights}, sum={sum(self.weights.values())}")


        
        best_dist_name, (best_bic, best_mean) = min(bic_map.items(), key=lambda x: x[1][0])
        self.best_distribution = best_dist_name
        self.empirical_mean = best_mean


# Configure logging to write to a file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='experiment_3_1.log',
    filemode='w'  # 'w' overwrites the file for each new run
)

num_arms = [3,2,3]
theta_array = [0.1, 0.5, 0.3]
sigma_array = [1,1,1]


T = 10000
repeat = 50
plt_sample = range(1,T+1,int(T/20))

# 1) UCB & UCB-C with NormalClusterTestBed
testbed = NormalClusterTestBed(num_arms, theta_array, sigma_array)
avg_ucb, ub_ucb, lb_ucb = testbed.ucb_perf_percentile(T, repeat)
avg_ucbc, ub_ucbc, lb_ucbc = testbed.ucb_c_perf_percentile(T, repeat)

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
avg_bic, ub_bic, lb_bic = stats(all_reg_bic)


plt.figure(figsize=(9, 6))
x_axis = np.arange(1, T + 1)

avg_regret_ucb, ub_regret_ucb, lb_regret_ucb = testbed.ucb_perf_percentile(T, repeat)
avg_regret_ucbc, ub_regret_ucbc, lb_regret_ucbc = testbed.ucb_c_perf_percentile(T, repeat)

plt.plot(x_axis, avg_ucb, 'b-', label="Standard UCB")
plt.fill_between(x_axis, lb_ucb, ub_ucb, color='blue', alpha=0.15)

plt.plot(x_axis, avg_ucbc, 'r-', label="UCB-C")
plt.fill_between(x_axis, lb_ucbc, ub_ucbc, color='red', alpha=0.15)

plt.plot(x_axis, avg_bic, 'g-', label="BIC-UCB")
plt.fill_between(x_axis, lb_bic, ub_bic, color='green', alpha=0.15)


plt.legend()
plt.xlabel('Time')
plt.ylabel('Cumulative Regret')
plt.title('Section 3.1')
plt.tight_layout()
plt.show()

plt.savefig("runs/3.1_plot.png")
