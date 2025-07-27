import numpy as np
import math
import random


class SubGaussianCandidate:
    def fit_MLE(self, data):     raise NotImplementedError
    def loglik(self, data):      raise NotImplementedError
    def get_mean(self):          raise NotImplementedError
    def get_num_params(self):    raise NotImplementedError


class GaussianFixedVarCandidate(SubGaussianCandidate):
    def __init__(self, sigma=1.0):
        self.sigma = sigma
        self.mu_hat = 0.5
        self.n_params = 1
    def fit_MLE(self, data):
        self.mu_hat = 0.5 if len(data) == 0 else np.mean(data)
    def loglik(self, data):
        n = len(data)
        if n == 0: return 0.0
        cst = -0.5 * n * math.log(2 * math.pi * self.sigma ** 2)
        sq  = -np.sum((data - self.mu_hat) ** 2) / (2 * self.sigma ** 2)
        return cst + sq
    def get_mean(self):
        return self.mu_hat
    def get_num_params(self):
        return self.n_params


class ExponentialCandidate(SubGaussianCandidate):
    """Exponential family parameterized by λ (rate)."""
    def __init__(self):
        self.lam_hat = 1.0
        self.n_params = 1
    def fit_MLE(self, data):
        n = len(data)
        if n == 0:
            self.lam_hat = 1.0
        else:
            s = max(np.sum(data), 1e-12)
            self.lam_hat = n / s
    def loglik(self, data):
        n = len(data)
        if n == 0: return 0.0
        lam = self.lam_hat
        return n * math.log(lam) - lam * np.sum(data)
    def get_mean(self):
        return 1.0 / self.lam_hat
    def get_num_params(self):
        return self.n_params


class BernoulliCandidate(SubGaussianCandidate):
    """Bernoulli family with parameter p."""
    def __init__(self):
        self.p_hat = 0.5
        self.n_params = 1
    def fit_MLE(self, data):
        n = len(data)
        if n == 0:
            self.p_hat = 0.5
        else:
            mean = np.mean(data)
            self.p_hat = min(max(mean, 1e-12), 1 - 1e-12)
    def loglik(self, data):
        n = len(data)
        if n == 0: return 0.0
        p = self.p_hat
        return np.sum(data * np.log(p) + (1 - data) * np.log(1 - p))
    def get_mean(self):
        return self.p_hat
    def get_num_params(self):
        return self.n_params