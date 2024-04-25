import os
import pandas as pd
import json
from os.path import join
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np
from utils import pickle_load, pickle_dump, csv_load, csv_dump
import matplotlib.pyplot as plt

from path_constants import RESULTS_PATH, PLOT_PATH

# Which experiment to visualize
run_name = "test_experiment_paper"

results_dir = RESULTS_PATH
results_dir = "./results"
plots_dir = PLOT_PATH

run_dir = join(results_dir, run_name)
save_dir = join(plots_dir, run_name, "iteration_plots")
os.makedirs(save_dir, exist_ok=True)

joined_results_csv = csv_load(join(run_dir, "joined_results_arr.csv"))
iter_results_csv = csv_load(join(run_dir, "iter_results_arr.csv"))

m = "mahalanobis"
d = "ksdd2"
perc_images = [0, 1, 5, 15, 25]
pp = -1
s_f = "auc"
eval_strat = 1
its = list(range(5))


def get_score_mean_std(c, m, d, pi, pp, eval_strat, s_f, iter_ix):
    ix = (c["method"] == m) & (c["dataset"] == d) & (c["perc_images"] == pi) & (c["perc_pixels"] == pp) & (c["eval_strat"] == eval_strat) & (c["score_function"] == s_f) & (c["iter_ix"] == iter_ix)
    return list(c.loc[ix]["score"])[0]


# %%
def plot_runs(m, d, pp, save_dir):
    plt.clf()
    plt.figure(figsize=(5, 5))

    pis = perc_images[1:] if pp != -1 and 0 in perc_images else perc_images
    for iter_ix in its:
        scores = [get_score_mean_std(iter_results_csv, m, d, pi, pp, 2 if pp == -1 else 3, s_f, iter_ix) for pi in pis]
        plt.plot(range(len(scores)), scores)

    title = f"{m}_{d}_{pp}_pixels"
    plt.title(title)
    plt.ylabel("auc")
    plt.xlabel("% images")
    plt.xticks(range(len(scores)), list(map(str, pis)))
    plt.savefig(join(save_dir, f"{title}.png"), dpi=200, bbox_inches="tight")


for m in ["mahalanobis", "padim", "patchcore"]:
    # for m in ["mahalanobis"]:
    for d in ["ksdd2"]:
        for pp in [-1, 1, 5, 25, 50]:
            plot_runs(m, d, pp, save_dir)
