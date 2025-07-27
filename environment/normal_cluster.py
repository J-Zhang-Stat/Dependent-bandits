import numpy as np


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
        reward = np.random.normal(self.theta * self.scale[arm], self.sigma)
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