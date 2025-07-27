import numpy as np
import math


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
                bonus[i] = np.sqrt(2 * np.log(t + 1) / self.obs_cnt[i])
        return self.obs_avg + bonus