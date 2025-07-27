import numpy as np
import math
from collections import defaultdict


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