import numpy as np

class HeterogeneousCluster:
    """
    A cluster in which each arm may follow a different reward distribution.
    We precompute feasible (mu, var) for each arm based on desired thetas and sigmas.
    Feedback and payoff then rely on these adjusted parameters.
    """
    def __init__(self, num_arms, thetas, sigmas, dist_types, gaussian_default=(0.5,0.25)):
        assert len(thetas) == num_arms, "len(thetas) must == num_arms"
        assert len(sigmas) == num_arms, "len(sigmas) must == num_arms"
        assert len(dist_types) == num_arms, "dist_types length must match arms"

        self.num_arms = num_arms
        self.requested_thetas = np.array(thetas, dtype=float)
        self.requested_sigmas = np.array(sigmas, dtype=float)
        self.dist_types = dist_types

        # default for Gaussian
        self.gauss_mu0, self.gauss_var0 = gaussian_default

        # Placeholders for actual implemented parameters
        self.thetas = np.empty(num_arms, dtype=float)
        self.vars = np.empty(num_arms, dtype=float)

        # Precompute feasible mu/var per arm
        for i in range(num_arms):
            self.thetas[i], self.vars[i] = self._compute_feasible_params(
                self.requested_thetas[i],
                self.requested_sigmas[i],
                self.dist_types[i]
            )

    def _compute_feasible_params(self, mu_req, sigma_req, dist):
        """
        Given requested mean (mu_req) and sigma (stddev) sigma_req,
        return an achievable (mu, var) pair for the specified dist.
        """
        var_req = sigma_req**2

        if dist == 'gaussian':
            mu = mu_req if not np.isnan(mu_req) else self.gauss_mu0
            var = var_req if not np.isnan(var_req) and var_req > 0 else self.gauss_var0
            return mu, var

        if dist == 'exponential':
            # E[X]=θ, Var[X]=θ^2
            if np.isnan(mu_req):
                theta = np.sqrt(var_req)
                var = var_req
            elif np.isnan(var_req):
                theta = mu_req
                var = theta**2
            else:
                theta = mu_req
                var = var_req
            return theta, var

        if dist == 'bernoulli':
            # E[X]=p, Var[X]=p(1-p)
            if np.isnan(mu_req):
                # solve p(1-p)=var_req
                discriminant = 1 - 4*var_req
                if discriminant < 0:
                    # too large: clamp var to max .25
                    p = 0.5
                    var = var_req
                else:
                    p = (1 + np.sqrt(discriminant)) / 2
                    var = p * (1-p)
            else:
                if mu_req > 1:
                    p = 1.0
                    var = 0.0
                elif mu_req < 0:
                    p = 0.0
                    var = 0.0
                else:
                    p = mu_req
                    var = p*(1-p)
            return p, var

        raise ValueError(f"Unknown dist type: {dist}")

    def _affine_transform(self, base_sample, base_mean, base_var, target_mean, target_var):
        b = np.sqrt(target_var / base_var)
        a = target_mean - b * base_mean
        return a + b * base_sample

    def feedback(self, arm):
        if arm < 0 or arm >= self.num_arms:
            raise IndexError("Arm index out of range")

        mu, var = self.thetas[arm], self.vars[arm]
        dist = self.dist_types[arm]

        if dist == 'gaussian':
            return np.random.normal(loc=mu, scale=np.sqrt(var))

        if dist == 'exponential':
            # Check if var = mu^2 (valid exponential parameters)
            if np.isclose(var, mu**2):
                # Valid exponential parameters, sample directly
                return np.random.exponential(scale=mu)
            else:
                # Invalid parameters, use affine transform
                # base Exp(1)
                z = np.random.exponential(scale=1.0)
                return self._affine_transform(z, 1.0, 1.0, mu, var)

        if dist == 'bernoulli':
            # Check if var = p*(1-p) (valid Bernoulli parameters)
            if np.isclose(var, mu*(1-mu)):
                # Valid Bernoulli parameters, sample directly
                return np.random.binomial(n=1, p=mu)
            else:
                # Invalid parameters, use affine transform
                # base Bernoulli(0.5)
                if mu < 0 or mu > 1:
                    base_p = 0.5
                    z = np.random.binomial(n=1, p=base_p)
                    return self._affine_transform(z, base_p, base_p*(1-base_p), mu, var)
                else:
                    z = np.random.binomial(n=1, p=mu)
                    return self._affine_transform(z, mu, mu*(1-mu), mu, var)

        raise ValueError(f"Unsupported distribution: {dist}")

    def payoff(self):
        # Return the true mean reward for each arm in this cluster.
        return self.thetas.copy()



class HeterogeneousClusterEnvironment:
    def __init__(self, num_arms, theta_array, sigma_array, arm_dist_types, contexts=None):
        """
        Manages multiple HeterogeneousClusters, ensuring per-arm parameters and providing feedback/regret.
        :param num_arms: List[int], number of arms in each cluster.
        :param theta_array: List[float or List[float]], desired means per cluster/arm.
        :param sigma_array: List[float or List[float]], desired std-devs per cluster/arm.
        :param arm_dist_types: List[List[str]], distribution types for each arm in each cluster.
        :param contexts: Optional contexts for contextual bandits.
        """
        self.num_cluster = len(num_arms)
        assert len(theta_array) == self.num_cluster, "theta_array length must match num clusters"
        assert len(sigma_array) == self.num_cluster, "sigma_array length must match num clusters"
        assert len(arm_dist_types) == self.num_cluster, "arm_dist_types length must match num clusters"

        self.num_arms = list(num_arms)
        # Expand scalar theta/sigma to per-arm lists
        self.theta_array = [
            (ta if isinstance(ta, list) else [ta]*self.num_arms[i])
            for i, ta in enumerate(theta_array)
        ]
        self.sigma_array = [
            (sa if isinstance(sa, list) else [sa]*self.num_arms[i])
            for i, sa in enumerate(sigma_array)
        ]

        self.arm_dist_types = arm_dist_types
        self.contexts = contexts

        # Initialize clusters
        self.clusters = []
        for i in range(self.num_cluster):
            self.clusters.append(HeterogeneousCluster(
                num_arms=self.num_arms[i],
                thetas=self.theta_array[i],
                sigmas=self.sigma_array[i],
                dist_types=self.arm_dist_types[i]
            ))

    def regret_per_action(self):
        """
        Compute the per-arm regret for each cluster as best_mean - arm_mean.
        Returns: List[np.ndarray] of length `num_cluster`, each array of shape (num_arms[i],)
        """
        payoffs = [c.payoff() for c in self.clusters]
        best    = np.max([np.max(p) for p in payoffs])
        delta   = [best - p for p in payoffs]
        return delta

    def feedback(self, cluster_idx, arm_idx):
        """
        Pull arm `arm_idx` in cluster `cluster_idx` and return the sampled reward.
        """
        return self.clusters[cluster_idx].feedback(arm_idx)