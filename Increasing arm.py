import random
import numpy as np
import math
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1) 环境与基础分布类的定义
# ---------------------------------------------

class NormalCluster:
    def __init__(self, num, theta, sigma=1):
        self.num = num
        self.theta = theta
        self.sigma = sigma
        # 假设真实期望为 theta * (1,2,3,...)
        self.scale = np.arange(num) + 1

    def feedback(self, arm):
        """在给定 arm 上抽取一次随机奖励"""
        if arm >= self.num:
            print("NormalCluster.feedback: index out of num_arms within the cluster")
            return None
        # 高斯分布，均值=theta*scale[arm], 方差=sigma^2
        reward = random.gauss(self.theta * self.scale[arm], self.sigma)
        return reward

    def payoff(self):
        """返回所有 arms 的真实期望，用于后续计算 regret"""
        return self.scale * self.theta


class NormalClusterEnvironment:
    """
    整个多 cluster 环境：每个 cluster 都是一个 NormalCluster。
    """
    def __init__(self, num_arms, theta_array, sigma_array):
        """
        num_arms: list, e.g. [3,2,4]，表示每个 cluster 的臂数
        theta_array, sigma_array: 每个 cluster 对应的真实参数
        """
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
        对每个 cluster 的每个 arm 计算单轮决策时的 regret:
          regret = (环境中所有 cluster 所有 arm 的最大期望) - (当前 cluster-arm 的真实期望)
        """
        payoffs = [c.payoff() for c in self.clusters]
        # largest 是所有 cluster 中最高的期望值
        largest = np.max([np.max(p) for p in payoffs])
        # delta[c][arm] = largest - payoff_of_cluster[c, arm]
        delta = [largest - p for p in payoffs]
        return delta

    def feedback(self, cluster, arm):
        """返回某个 cluster、某条臂的随机奖励"""
        return self.clusters[cluster].feedback(arm)


# ---------------------------------------------
# 2) 定义用于 BIC-UCB 的分布候选类
# ---------------------------------------------

class SubGaussianCandidate:
    """
    通用接口：fit_MLE / loglik / get_mean / get_num_params
    """
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
        self.n_params = 1  # 只估计一个参数 mu

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
        # 对应 N(mu_hat, var) 的对数似然
        cst = -0.5 * n * math.log(2 * math.pi * var)
        sq = - np.sum((data - self.mu_hat) ** 2) / (2 * var)
        return cst + sq

    def get_mean(self):
        return self.mu_hat

    def get_num_params(self):
        return self.n_params


class ExponentialCandidate(SubGaussianCandidate):
    def __init__(self):
        self.lam_hat = 1.0
        self.n_params = 1  # 只估计 lam

    def fit_MLE(self, data):
        n = len(data)
        if n == 0:
            self.lam_hat = 1.0
        else:
            sum_data = np.sum(data)
            if sum_data < 1e-12:
                sum_data = 1e-12
            self.lam_hat = n / sum_data  # MLE: lam = n / sum(x)

    def loglik(self, data):
        n = len(data)
        if n == 0:
            return 0.0
        lam = self.lam_hat
        # 对应 Exp(lam) 分布，loglik = n log lam - lam sum(x)
        return n * math.log(lam) - lam * np.sum(data)

    def get_mean(self):
        return 1.0 / self.lam_hat

    def get_num_params(self):
        return self.n_params


class BernoulliCandidate(SubGaussianCandidate):
    def __init__(self):
        self.p_hat = 0.5
        self.n_params = 1  # 只估计 p

    def fit_MLE(self, data):
        n = len(data)
        if n == 0:
            self.p_hat = 0.5
        else:
            mean_ = np.mean(data)
            # 避免出现 p=0 或 p=1
            mean_ = min(max(mean_, 1e-12), 1 - 1e-12)
            self.p_hat = mean_

    def loglik(self, data):
        n = len(data)
        if n == 0:
            return 0.0
        p = self.p_hat
        # Bernoulli 的对数似然： sum( x_i log p + (1-x_i) log (1-p) )
        ll = np.sum(data * math.log(p) + (1 - data) * math.log(1 - p))
        return ll

    def get_mean(self):
        return self.p_hat

    def get_num_params(self):
        return self.n_params


# 注册可用的分布类型
CANDIDATE_REGISTRY = {
    'gaussian': GaussianFixedVarCandidate,
    'exponential': ExponentialCandidate,
    'bernoulli': BernoulliCandidate,
}


# ---------------------------------------------
# 3) BIC-UCB 算法：核心类
# ---------------------------------------------

class DepCluster:
    """
    表示 BIC-UCB 下的单个 cluster，对其内部数据进行分布匹配 (BIC)，
    同时维护对各个 arm 的观测次数与均值，用于做两层 UCB (cluster-level + arm-level)。
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

        # 记录这个 cluster 内每条 arm 的拉取次数与均值
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
        """
        用当前 cluster 的全部观测 data 去拟合 dist_family 中的不同分布；
        计算 BIC 分数并做加权 (或直接选最优)。
        """
        n = len(self.data)
        if n == 0:
            self.best_distribution = None
            self.empirical_mean = 0.0
            return

        data_arr = np.array(self.data)
        bic_map = {}

        for dist_name in self.dist_family:
            DistClass = CANDIDATE_REGISTRY[dist_name]
            # 如果是 Gauss 分布，需要知道 sigma
            if dist_name == 'gaussian':
                candidate = DistClass(sigma=config.sigma)
            else:
                candidate = DistClass()

            candidate.fit_MLE(data_arr)
            ll = candidate.loglik(data_arr)
            k = candidate.get_num_params()
            bic_val = -2 * ll + k * math.log(n)
            bic_map[dist_name] = (bic_val, candidate.get_mean())

        # Softmax in BIC space（可选）
        all_bic = np.array([val[0] for val in bic_map.values()])
        min_bic = np.min(all_bic)
        # 这里的 -4.993 是一个可选的温度系数，具体可自行调节；也可以直接选 best_distribution
        exp_val = np.exp(-4.993 * (all_bic - min_bic))

        if np.sum(exp_val) == 0 or np.isnan(exp_val).any() or np.isinf(exp_val).any():
            w_arr = np.ones_like(exp_val) / len(exp_val)
        else:
            w_arr = exp_val / np.sum(exp_val)

        dist_list = list(self.dist_family)
        for i, dist_name in enumerate(dist_list):
            self.weights[dist_name] = w_arr[i]

        # 也可以直接选 BIC 最优的分布
        best_dist_name, (best_bic, best_mean) = min(bic_map.items(), key=lambda x: x[1][0])
        self.best_distribution = best_dist_name
        self.empirical_mean = best_mean


class DependentBandit:
    """
    BIC-UCB 的主类：每一轮做以下事情：
      1) 每个 cluster 基于已有数据更新 BIC 最优分布 (或混合权重)
      2) 选一个 cluster (cluster-level UCB)
      3) 在该 cluster 内选一个 arm (arm-level UCB)
      4) 获得奖励并更新
    """
    def __init__(self, config, environment):
        self.config = config
        self.env = environment
        self.num_cluster = environment.num_cluster

        # 为每个 cluster 创建 DepCluster
        self.clusters = [
            DepCluster(cluster_id=c_id, sigma=self.config.sigma, dist_family=self.config.F)
            for c_id in range(self.num_cluster)
        ]
        self.delta = self.env.regret_per_action()  # 用于计算(集群,臂)的 regret
        self.total_regret = 0.0
        self.regret_history = []

    def reset(self):
        for c in self.clusters:
            c.reset()
        self.total_regret = 0.0
        self.regret_history = []

    def run_one_trial(self, T):
        """
        跑 T 步决策，返回长度为 T 的“累计懊悔”序列。
        """
        for t in range(1, T + 1):
            # 1) 更新每个 cluster 的 BIC 分布选择
            for c in self.clusters:
                c.compute_BIC_scores(self.config)

            # 2) 选 cluster
            chosen_cluster = self.select_cluster(t)

            # 3) 选该 cluster 下的 arm
            chosen_arm = self.select_arm_in_cluster(chosen_cluster, t)

            # 4) 和环境交互，得到奖励
            r = self.env.feedback(chosen_cluster.cluster_id, chosen_arm)
            chosen_cluster.data.append(r)
            chosen_cluster.N += 1

            # 更新该 arm 的统计量
            old_cnt = chosen_cluster.arm_N.get(chosen_arm, 0)
            chosen_cluster.arm_N[chosen_arm] = old_cnt + 1
            old_mean = chosen_cluster.arm_mean.get(chosen_arm, 0.0)
            new_cnt = chosen_cluster.arm_N[chosen_arm]
            chosen_cluster.arm_mean[chosen_arm] = (old_mean * (new_cnt - 1) + r) / new_cnt

            # 5) 计算这一轮的 instant regret，加到 total_regret
            inst_r = self.delta[chosen_cluster.cluster_id][chosen_arm]
            self.total_regret += inst_r
            self.regret_history.append(self.total_regret)

        return np.array(self.regret_history)

    def select_cluster(self, t):
        """
        cluster-level UCB: 这里简单用
          score = cluster.empirical_mean + sqrt( c * log(t+1) / cluster.N ) * (1 - avg_weight)
        如果某 cluster 还没被拉过，就优先选它。
        """
        best_c = None
        best_val = -1e9
        for c in self.clusters:
            # 如果一个 cluster 没有被选过，先尝试它
            if c.N == 0:
                return c
            # 用 weights 的平均值来调节置信区间，你也可以只用 sqrt(...) 而不用 (1 - avg_w)
            avg_w = np.mean(list(c.weights.values()))
            bonus = math.sqrt(1.7 * math.log(t + 1) / c.N) * (1 - avg_w)
            val = c.empirical_mean + bonus
            if val > best_val:
                best_val = val
                best_c = c
        return best_c

    def select_arm_in_cluster(self, cluster, t):
        """
        arm-level UCB: mean + sqrt( c * log(t+1) / n_arm )
        如果某条臂从未选过，则直接选它。
        """
        best_arm = None
        best_val = -1e9
        c_id = cluster.cluster_id
        arm_cnt = self.env.num_arms[c_id]

        for arm_id in range(arm_cnt):
            n_arm = cluster.arm_N.get(arm_id, 0)
            if n_arm == 0:
                return arm_id  # 优先尝试没被拉过的臂

            mean_arm = cluster.arm_mean.get(arm_id, cluster.empirical_mean)
            bonus = math.sqrt(1.7 * math.log(t + 1) / n_arm)
            val = mean_arm + bonus
            if val > best_val:
                best_val = val
                best_arm = arm_id
        return best_arm


# ---------------------------------------------
# 4) Main: 增加一个 cluster 的 arms 数量来观察对 BIC-UCB 的影响
# ---------------------------------------------
if __name__ == "__main__":
    # 下面示例里假设我们有 5 个 clusters，其中第 1 个 cluster (索引=0) 数量不变，
    # 第 2 个 cluster (索引=1) 将尝试不同的 arms 数值进行对比，其它 cluster 都固定。
    # 你可以根据自己的场景灵活调整。

    base_num_arms = [2, 2, 3, 2, 3]   # 其余 clusters 的臂数设为固定
    theta_array = [0.1, 0.5, 0.2, 0.3, 0.1]  # 每个 cluster 的真实 theta
    sigma_array = [1, 1, 1, 1, 1]           # 噪声标准差

    # 测试下面这些 arms 数量：比如 [2,4,6,8,10]
    arms_candidate = [2, 4, 6, 8, 10]

    T = 3000      # 每次实验的回合数
    repeat = 50   # 每种 arms 配置重复做多少次实验

    # BIC-UCB 配置
    class Config: 
        pass

    config = Config()
    config.F = ['gaussian', 'exponential', 'bernoulli']  # 候选分布
    config.sigma = 1.0

    plt.figure(figsize=(10, 6))
    
    for arms_val in arms_candidate:
        # 让第 2 个 cluster (即索引=1) 的臂数设为 arms_val
        new_num_arms = base_num_arms.copy()
        new_num_arms[1] = arms_val

        # 重复试验，记录累计懊悔
        cum_reg_list = []
        for _ in range(repeat):
            # 1) 创建环境
            env = NormalClusterEnvironment(new_num_arms, theta_array, sigma_array)
            # 2) 创建 BIC-UCB 算法
            bandit = DependentBandit(config, env)
            # 3) 运行 T 步
            reg_arr = bandit.run_one_trial(T)  # shape: (T,)
            cum_reg_list.append(reg_arr)

        # 计算平均累计懊悔
        cum_reg_list = np.array(cum_reg_list)   # shape: (repeat, T)
        avg_reg = cum_reg_list.mean(axis=0)     # shape: (T,)

        # 绘制曲线
        plt.plot(range(T), avg_reg, label=f'Cluster[1] arms={arms_val}')

    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Regret')
    plt.title('BIC-UCB: Impact of Increasing Arms in One Cluster')
    plt.legend()
    plt.tight_layout()
    plt.show()
