import numpy as np
import torch
import math
import time
import logging


class ClusteredKernelUCB:
    """
    A simple implementation of the Clustered Kernel-UCB algorithm (CK-UCB).
    Each cluster maintains its own kernel-ridge regressor to estimate the mean and uncertainty.
    """
    def __init__(self, num_arms, contexts, lambda_=0.2, gamma=1.0, B=1.0, sigma=1.0, delta=0.1):
        """
        :param contexts: np.ndarray, shape (n_arms, d) – feature vectors x_i for each arm
        :param num_arms: array-like, length n_arms – cluster assignment C(i) for each arm i e.g.([3,2,2] 3 clusters, 7 total)
        :param lambda_: float – ridge regularization parameter (λ)
        :param gamma: float – bandwidth parameter for Gaussian kernel
        :param B: float – RKHS norm bound, ||θ^* _k || <=B for all clusters k 
        :param sigma: float – sub-Gaussian noise parameter
        :param delta: float – confidence level parameter
        """
        self.contexts = np.array(contexts) # (n_arms, d)
        self.clusters = []
        c_idx = 0
        for c in num_arms:
            self.clusters.extend([c_idx] * c)
            c_idx += 1
        self.clusters = np.array(self.clusters)
        assert len(self.contexts) == len(self.clusters), \
            "Contexts and clusters must have the same length"

        #print(f"Contexts shape: {self.contexts.shape}, Clusters length: {len(self.clusters)}")
        
        self.lambda_ = lambda_
        self.gamma = gamma
        self.B = B
        self.sigma = sigma
        self.delta = delta
        self.n_arms = len(contexts)
        # set up GPU device and mixed-precision context tensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # keep a half-precision tensor of all contexts on GPU
        self.contexts_tensor = torch.tensor(self.contexts, device=self.device, dtype=torch.bfloat16)

        #self._kernel = self._gaussian_kernel  # Use Gaussian kernel by default
        
        # Store history as GPU tensors to avoid CPU-GPU transfers
        self.cluster_data = {}
        d = self.contexts.shape[1]
        for c in np.unique(self.clusters):
            self.cluster_data[c] = {
                'X': torch.empty((0, d), device=self.device, dtype=torch.float32),  # contexts
                'r': torch.empty((0, 1), device=self.device, dtype=torch.float32)   # rewards
            }
        
    def _gaussian_kernel(self, x1, x2):
        """Gaussian (RBF) kernel."""
        #return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * self.gamma**2))
        raise NotImplementedError("Kernel function should not be called directly. Use select_arm() instead.")
    
    def select_arm(self, t):
        """
        Compute UCB scores for all arms at round t and return the UCB scores for all arms.
        """
        # synchronize and start timing
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
        start_time = time.time()

        # Preallocate UCB scores
        ucb_scores = torch.zeros(self.n_arms, device=self.device, dtype=torch.float32)

        # For each cluster, do batch processing
        for c in torch.unique(torch.tensor(self.clusters, device=self.device)):
            arms_idx = (self.clusters == c.item()).nonzero()[0]  # indices of arms in cluster c
            X_c = self.cluster_data[c.item()]['X']    # (n_c, d)
            r_c = self.cluster_data[c.item()]['r']    # (n_c, 1)
            n_c = X_c.size(0)

            if n_c == 0:
                # no data, UCB = beta * 1 for all arms
                beta_t = self.B + math.sqrt(2*self.sigma**2 * math.log(1/self.delta))
                ucb_scores[arms_idx] = beta_t
                continue

            # 1) Compute Gram and factor once
            diff = X_c.unsqueeze(1) - X_c.unsqueeze(0)
            K_c = torch.exp(-diff.pow(2).sum(-1) / (2 * self.gamma**2))
            G_c = K_c + self.lambda_ * torch.eye(n_c, device=self.device)
            L = torch.linalg.cholesky(G_c)

            # 2) Solve for alpha: G_c @ alpha = r_c
            alpha = torch.cholesky_solve(r_c, L).squeeze(1)  # (n_c,)

            # 3) Query contexts batch
            X_q = self.contexts_tensor[arms_idx].to(torch.float32)  # (m, d)
            # kernel vectors: (n_c, m)
            k_mat = torch.exp(-((X_c.unsqueeze(1) - X_q.unsqueeze(0)).pow(2).sum(-1)) / (2 * self.gamma**2))

            # 4) predictive means and variances
            means = (alpha.unsqueeze(1) * k_mat).sum(0)  # (m,)
            v = torch.cholesky_solve(k_mat, L)           # (n_c, m)
            vars = 1.0 - (k_mat * v).sum(0)               # (m,)

            # 5) compute log_det_term and beta_t using torch
            M = torch.eye(n_c, device=self.device) + (1.0 / self.lambda_) * K_c
            sign, logabsdet = torch.linalg.slogdet(M)
            log_det_term = torch.log(torch.tensor(1.0/self.delta, device=self.device)) + logabsdet
            # combine constants and log_det_term
            beta_t = self.B + torch.sqrt(
                2 * self.lambda_ * self.B**2 + 2 * (self.sigma**2) * log_det_term
            )
            stds = torch.sqrt(vars.clamp(min=0.0))

            # 6) UCB scores for this cluster
            ucb_scores[arms_idx] = means + beta_t * stds

            # Detailed logging per arm in cluster
            if t % 100 == 0:
                for j, arm_idx in enumerate(arms_idx.tolist()):
                    logging.info(
                        f"t={t}, cluster={int(c)}, arm={arm_idx}, "
                        f"mean={means[j].item():.4f}, std={stds[j].item():.4f}, "
                        f"beta_t={beta_t.item():.4f}, ucb={ucb_scores[arm_idx].item():.4f}"
                    )
        # end cluster loop

        # synchronize and measure elapsed time
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
        elapsed = time.time() - start_time
        if t % 100 == 0:
            logging.info(f"t={t}: elapsed={elapsed:.4f}s, UCB scores computed for {self.n_arms} arms")
            logging.info(f"Time {t}: full UCB scores: {ucb_scores.cpu().numpy()}")

        # Move to CPU numpy for compatibility
        return ucb_scores.cpu().numpy()
    
    def update(self, arm, reward):
        """
        After pulling `arm` and observing `reward`, update the corresponding cluster's data.
        """
        c = self.clusters[arm]
        x_new = self.contexts_tensor[arm].to(torch.float32).unsqueeze(0)  # (1, d)
        r_new = torch.tensor([[reward]], device=self.device, dtype=torch.float32)  # (1,1)
        self.cluster_data[c]['X'] = torch.cat([self.cluster_data[c]['X'], x_new], dim=0)
        self.cluster_data[c]['r'] = torch.cat([self.cluster_data[c]['r'], r_new], dim=0)

