# core_experiment.py

import os
import pickle
import json
import sys
import logging

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

def run_ck_ucb(env, T, rep=None):
    if rep is not None:
        logging.info(f"CK-UCB run #{rep+1}")
    learner = ClusteredKernelUCB(
        num_arms=env.num_arms,
        contexts=env.contexts,
        lambda_=0.1, gamma=1.0, B=1.0, sigma=1.0, delta=0.1
    )
    logging.info(f"CK-UCB params: λ={learner.lambda_}, γ={learner.gamma}, B={learner.B}")
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

def run_experiments(env, T, repeat):
    ucbs, oracles, cks = [], [], []
    for rep in tqdm(range(repeat), desc="Reps", disable=True):
        logging.info(f"---- Replication {rep+1}/{repeat} ----")
        ucbs.append(   run_ucb(env, T).cumsum()      )
        oracles.append(run_oracle_ucb(env, T).cumsum())
        cks.append(   run_ck_ucb(env, T, rep).cumsum())

        logging.info(
            f"Replication {rep+1} summary: "
            f"final cumulative regret - UCB: {np.round(ucbs[-1][-1], 4)}, "
            f"Oracle: {np.round(oracles[-1][-1], 4)}, CK-UCB: {np.round(cks[-1][-1], 4)}"
        )
        
    return {"ucb": np.array(ucbs),
            "oracle": np.array(oracles),
            "ck-ucb": np.array(cks)}

def main(config):
    # 1) Logging
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join("runs", config["experiment_name"], ts)
    os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(logdir, "log.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    logging.info(f"Experiment: {config['experiment_name']}")
    # 2) Env
    env = HeterogeneousClusterEnvironment(
        config["num_arms"],
        config["theta_array"],
        config["sigma_array"],
        config["arm_dist_types"],
        config.get("contexts", None)
    )
    logging.info(f"Env settings: {config}")
    logging.info(f"actual settings:")
    logging.info(f"   Thetas: {[c.thetas for c in env.clusters]}")
    logging.info(f"   Vars: {[c.vars for c in env.clusters]}")
    logging.info(f"   Arm types: {[c.dist_types for c in env.clusters]}")
    
    # 3) Run w/ amp
    logging.info(f"Running experiments on device: {device.upper()} with AMP dtype: {amp_dtype}")
    with torch.amp.autocast(device_type=device, dtype=amp_dtype):
        results = run_experiments(env, config["T"], config["repeat"])
        
    # 4) Save
    with open(os.path.join(logdir, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    logging.info("Results saved")
    
    # 5) Plot
    x = np.arange(1, config["T"]+1)
    plt.figure(figsize=(8,5))
    plot_map = {
            "ucb": ("Standard UCB", "b"),
            "oracle": ("Oracle", "r"),
            "ck-ucb": ("CK-UCB", "g")
        }
    for label, (plot_label, color) in plot_map.items():
        avg, ub, lb = stats(results[label])
        plt.plot(x, avg, color, label=plot_label)
        plt.fill_between(x, lb, ub, alpha=0.2, color=color)
    plt.xlabel("t")
    plt.ylabel("Cumulative Regret")
    plt.legend(loc='upper left')
    plt.suptitle(config["experiment_name"])
    subtitle = (f"Thetas: {[(c.thetas) for c in env.clusters]}\n"
                f"Vars: {[(c.vars) for c in env.clusters]}\n"
                f"Arm types: {[c.dist_types for c in env.clusters]}")
    plt.title(subtitle, fontsize=8)
    plt.tight_layout()
    plot_path = os.path.join(logdir, "cumulative_regret.png")
    plt.savefig(plot_path)
    logging.info(f"Plot saved to {plot_path}")
    #plt.show()

if __name__ == "__main__":
    # --- AMP Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16
    
    cfg = json.load(open(sys.argv[1], "r"))
    main(cfg)
