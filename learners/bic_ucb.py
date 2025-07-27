import numpy as np

from environment.candidates import GaussianFixedVarCandidate, ExponentialCandidate, BernoulliCandidate


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
            bic = -2 * ll + k * np.log(n)
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
                     np.sqrt(2.0 * np.log(t + 1) / (1 + self.cluster_counts[c])))

            ucb_vals[arm_idx] = mean_est + bonus

        return ucb_vals