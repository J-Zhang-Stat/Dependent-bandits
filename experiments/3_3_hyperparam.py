# Expected cfg keys:
#  ├─ experiment_name : str
#  ├─ num_arms, theta_array, sigma_array, arm_dist_types, contexts
#  ├─ T : int           – horizon
#  ├─ repeat : int      – #replications (≥10)
#  ├─ ck_params : dict  – default CK‑UCB hyper‑parameters
#  └─ sweep (optional) :
#        { "param": "<lambda_|gamma|B|sigma|delta>",
#          "values": [v1, v2, …] }  – list size ≥5

import os
import pickle
import logging
import json
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from datetime import datetime

from environment.heterogeneous_cluster import HeterogeneousClusterEnvironment
from experiments.utils import stats
from learners import UCBLearner, OracleUCBLearner, ClusteredKernelUCB

def run_ucb(env, T):
    total_arms = sum(env.num_arms)
    learner = UCBLearner(total_arms)
    regret, delta = [], env.regret_per_action()
    # build global arm→(cluster,arm) map
    arm_to_cluster = {i:(c,a)
        for c in range(env.num_cluster)
        for a in range(env.num_arms[c])
        for i in [sum(env.num_arms[:c])+a]}
    # init
    for i in range(total_arms):
        c,a = arm_to_cluster[i]
        r   = env.feedback(c,a)
        learner.receive_feedback(i,r)
        regret.append(delta[c][a])
    # main loop
    for t in range(total_arms, T):
        chosen = np.argmax(learner.get_ucb(t))
        c,a    = arm_to_cluster[chosen]
        r      = env.feedback(c,a)
        learner.receive_feedback(chosen, r)
        regret.append(delta[c][a])
    return np.array(regret)

def run_oracle_ucb(env, T):
    learner = OracleUCBLearner(env.num_cluster, env.num_arms, env.arm_dist_types)
    regret, delta = [], env.regret_per_action()
    arm_to_cluster = {i:(c,a)
        for c in range(env.num_cluster)
        for a in range(env.num_arms[c])
        for i in [sum(env.num_arms[:c])+a]}
    total_arms = sum(env.num_arms)
    # init
    for i in range(total_arms):
        c,a = arm_to_cluster[i]
        r   = env.feedback(c,a)
        learner.receive_feedback(c,a,r)
        regret.append(delta[c][a])
    # main loop
    for t in range(total_arms, T):
        chosen = np.argmax(learner.get_ucb(t))
        c,a    = arm_to_cluster[chosen]
        r      = env.feedback(c,a)
        learner.receive_feedback(c,a,r)
        regret.append(delta[c][a])
    return np.array(regret)

def run_ck_ucb(env, T, rep=None, ck_params=None):
    if rep is not None:
        logging.info(f"CK-UCB run #{rep+1}")
    if ck_params is None:
        ck_params = {}
    learner = ClusteredKernelUCB(
        num_arms=env.num_arms,
        contexts=env.contexts,
        lambda_=ck_params.get("lambda_", 0.1),
        gamma=ck_params.get("gamma", 1.0),
        B=ck_params.get("B", 1.0),
        sigma=ck_params.get("sigma", 1.0),
        delta=ck_params.get("delta", 0.1)
    )
    regret, delta = [], env.regret_per_action()
    arm_to_cluster = {i:(c,a)
        for c in range(env.num_cluster)
        for a in range(env.num_arms[c])
        for i in [sum(env.num_arms[:c])+a]}
    total_arms = sum(env.num_arms)
    # init
    for i in range(total_arms):
        c,a = arm_to_cluster[i]
        r   = env.feedback(c,a)
        learner.update(i,r)
        regret.append(delta[c][a])
    # main loop
    for t in tqdm(range(total_arms, T), desc="CK-UCB", leave=False, disable=True):
        chosen = np.argmax(learner.select_arm(t))
        c,a    = arm_to_cluster[chosen]
        r      = env.feedback(c,a)
        learner.update(chosen, r)
        regret.append(delta[c][a])
    return np.array(regret)

def _final_stats(cum_regret_arr):
    """
    Return mean and 95 % CI half‑width of the final cumulative regret.
    `cum_regret_arr` shape: (replications, T)
    """
    finals = cum_regret_arr[:, -1]
    mean   = finals.mean()
    # 95 % CI using t≈1.96
    hw     = 1.96 * finals.std(ddof=1) / np.sqrt(len(finals))
    return mean, hw

def sweep_hyperparam(param_name, values, base_cfg):
    """
    Sweep over `param_name` for CK‑UCB.  Other hyper‑parameters are
    kept fixed as in base_cfg['ck_params'].
    Returns aggregated statistics for plotting.
    """
    means = {k: [] for k in ["ucb", "oracle", "ck-ucb"]}
    hws   = {k: [] for k in ["ucb", "oracle", "ck-ucb"]}

    for v in values:
        ck_params = base_cfg["ck_params"].copy()
        ck_params[param_name] = v
        res = run_experiments(base_cfg, base_cfg["T"],
                              base_cfg["repeat"], ck_params)

        for algo in ["ucb", "oracle", "ck-ucb"]:
            m, hw = _final_stats(res[algo])
            means[algo].append(m)
            hws[algo].append(hw)

    return means, hws

def plot_sweep(values, means, hws, param_name, fixed_ck_params, cfg):
    fig = plt.figure(figsize=(7,4))
    for algo, color in [("ucb","b"), ("oracle","r"), ("ck-ucb","g")]:
        plt.errorbar(values, means[algo], yerr=hws[algo],
                     label=algo, marker="o", linestyle="-", color=color, capsize=4)
    plt.xlabel(param_name)
    # Ensure x‑axis ticks exactly match the swept parameter values
    plt.xticks(values, [str(v) for v in values])
    plt.ylabel("Mean final cumulative regret")
    title_fixed = ", ".join([f"{k}={v}" for k,v in fixed_ck_params.items() if k!=param_name])
    plt.title(title_fixed, fontsize=8)
    plt.suptitle(f"Sweep of {param_name}  |  T={cfg['T']}  reps={cfg['repeat']}")
    plt.legend()
    plt.tight_layout()
    return fig

def run_experiments(base_cfg, T, repeat, ck_params):
    """
    Run `repeat` independent replications, each with a *fresh*
    HeterogeneousClusterEnvironment so that bandit state does not leak.
    Returns dict of shape‑(repeat, T) arrays for each algorithm.
    """
    ucbs, oracles, cks = [], [], []
    for rep in tqdm(range(repeat), desc="Reps", disable=False):
        env = HeterogeneousClusterEnvironment(
            base_cfg["num_arms"],
            base_cfg["theta_array"],
            base_cfg["sigma_array"],
            base_cfg["arm_dist_types"],
            base_cfg.get("contexts", None)
        )
        # Log the ground‑truth reward statistics for this fresh environment
        logging.info(
            f"Replication {rep+1}: "
            f"true means per cluster = {[c.thetas for c in env.clusters]}"
        )
        logging.info(
            f"Replication {rep+1}: "
            f"true variances per cluster = {[c.vars for c in env.clusters]}"
        )
        logging.info(f"---- Replication {rep+1}/{repeat} ----")
        ucbs.append(   run_ucb(env, T).cumsum()      )
        oracles.append(run_oracle_ucb(env, T).cumsum())
        cks.append(    run_ck_ucb(env, T, rep, ck_params).cumsum())
        logging.info(
            f"Replication {rep+1} summary: "
            f"final cumulative regret - UCB: {np.round(ucbs[-1][-1], 4)}, "
            f"Oracle: {np.round(oracles[-1][-1], 4)}, CK-UCB: {np.round(cks[-1][-1], 4)}"
        )

    return {
        "ucb":    np.asarray(ucbs),
        "oracle": np.asarray(oracles),
        "ck-ucb": np.asarray(cks)
    }

def main():

    cfg = json.load(open(sys.argv[1], "r"))

    if "sweep" not in cfg:
        raise ValueError("This experiment file only supports hyper‑parameter sweeps. "
                         "Add a 'sweep' block to your config.")

    # ==== plain run or sweep ===========================================
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join("runs", cfg["experiment_name"], ts)
    os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logdir, "log.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    logging.info(f"Experiment: {cfg['experiment_name']}")

    param_name   = cfg["sweep"]["param"]
    values       = cfg["sweep"]["values"]          # list
    assert len(values) >= 5, "Need ≥5 values for a sweep"

    logging.info(f"Running experiments on device: {device.upper()} with AMP dtype: {amp_dtype}")
    with torch.amp.autocast(device_type=device, dtype=amp_dtype):
        means, hws = sweep_hyperparam(param_name, values, cfg)
        
    fig = plot_sweep(values, means, hws, param_name,
                        cfg["ck_params"], cfg)
    plot_path = os.path.join(logdir, f"sweep_{param_name}.png")
    fig.savefig(plot_path, dpi=300)
    logging.info(f"Sweep plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    main()