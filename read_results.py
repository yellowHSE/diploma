import os
from os.path import join
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np
from utils import pickle_load, pickle_dump, csv_load, csv_dump
import matplotlib.pyplot as plt
import pandas as pd

from path_constants import RESULTS_PATH, PLOT_PATH

# Selection of the experiment to visualize
#run_name = "test_experiment_paper"
run_name = "test_multiple_methods_multiple_datasets_2"

# Which methods to visualize
# methods = ["riad", "padim", "mahalanobis", "patchcore", "cutpaste", "draem", "cflow"]
methods = ["mahalanobis", "draem"]

# Which datasets to visualize
# datasets = ["ksdd2", "dagm10", "bsdata", "softgel"]
datasets = ["ksdd2", "bsdata"]

# What percentages
# perc_images = [0, 1, 5, 15, 25]
perc_images = [0, 1, 5, 15, 25]

# Leftover
# perc_pixels = [-1, 1, 5, 25, 50]
perc_pixels = [-1]
its = list(range(5))

eval_strats = [1, 2, 3]  # 1 => bad+perlin, 2 => bad only, 3 => perlin only
eval_strats = [1, 2, 3]  # 1 => bad+perlin, 2 => bad only, 3 => perlin only
# score_functions = ["ap", "auc", "fp+fn", "f1_opt", "cls_acc", "fp@0fn", ]
score_functions = ["auc"]

plot_single_runs = True
use_cached_results = False

linestyle_tuple = [
    ('solid', (0, ())),
    ('dotted', (0, (1, 1))),
    ('densely dotted', (0, (1, 1))),

    ('loosely dashed', (0, (5, 10))),
    ('dashed', (0, (5, 5))),
    ('densely dashed', (0, (5, 1))),

    ('loosely dashdotted', (0, (3, 10, 1, 10))),
    ('dashdotted', (0, (3, 5, 1, 5))),
    ('densely dashdotted', (0, (3, 1, 1, 1))),

    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]


def calc_confusion_mat(gts, scores):
    FP = (scores != gts) & (gts.astype(np.bool) == False)
    FN = (scores != gts) & (gts.astype(np.bool) == True)
    TN = (scores == gts) & (gts.astype(np.bool) == False)
    TP = (scores == gts) & (gts.astype(np.bool) == True)

    return sum(FP), sum(FN), sum(TN), sum(TP)


def get_confusion_mat(gts, scores):
    precision_, recall_, thresholds = precision_recall_curve(gts, scores)
    f_measures = 2 * (precision_ * recall_) / (precision_ + recall_ + 0.0000000001)
    ix_best = np.argmax(f_measures)
    if ix_best > 0:
        best_threshold = (thresholds[ix_best] + thresholds[ix_best - 1]) / 2
    else:
        best_threshold = thresholds[ix_best]
    FP, FN, TN, TP = calc_confusion_mat(gts, scores >= best_threshold)
    return FP, FN, TN, TP


def fp_0fn_score(gts, scores):
    precision_, recall_, thresholds = precision_recall_curve(gts, scores)
    fn0_threshold = thresholds[np.where(recall_ >= 1)][0]
    FP, FN, TN, TP = calc_confusion_mat(gts, scores >= fn0_threshold)
    return FP


def f1_opt_score(gts, scores):
    precision_, recall_, thresholds = precision_recall_curve(gts, scores)
    f_measures = 2 * (precision_ * recall_) / (precision_ + recall_ + 0.0000000001)
    return np.max(f_measures)


def fp_fn_score(gts, scores):
    FP, FN, TN, TP = get_confusion_mat(gts, scores)
    return FP + FN


def cls_acc_score(gts, scores):
    FP, FN, TN, TP = get_confusion_mat(gts, scores)
    return (TP + TN) / (TP + TN + FP + FN)


def filter_samples(gts, scores, eval_strat):
    if eval_strat == 1:
        gts = (gts > 0) * 1
    elif eval_strat == 2:
        ix = np.logical_or(gts == 0, gts == 1)
        gts = (gts[ix] > 0) * 1
        scores = scores[ix]
    elif eval_strat == 3:
        ix = np.logical_or(gts == 0, gts == 2)
        gts = (gts[ix] > 0) * 1
        scores = scores[ix]
    else:
        raise Exception("eval_strat must be one of [1,2,3] ")
    return gts, scores


def get_score(score_function, gts, scores):
    if score_function == "auc":
        score_fn = roc_auc_score
    elif score_function == "ap":
        score_fn = average_precision_score
    elif score_function == "fp+fn":
        score_fn = fp_fn_score
    elif score_function == "f1_opt":
        score_fn = f1_opt_score
    elif score_function == "cls_acc":
        score_fn = cls_acc_score
    elif score_function == "fp@0fn":
        score_fn = fp_0fn_score
    else:
        raise Exception(f"Unknown score: {score_function}")

    return score_fn(gts, scores)


def plot_raw_scores(gts, scores, ds, method, perc_images, perc_pixels, i, save_dir):
    plt.clf()
    plt.figure(figsize=(4, 4))
    zipped_sorted = sorted(zip(scores, gts), key=lambda x: x[0])
    gts = list(map(lambda x: x[1], zipped_sorted))
    scores = list(map(lambda x: x[0], zipped_sorted))
    colors = [["g", "r", "b"][gt] for gt in gts]
    plt.scatter(range(len(scores)), sorted(scores), c=colors)
    plt.xticks([])
    plt.yticks([])
    plt.grid()
    plt.savefig(join(save_dir, f"{ds}_{method}_{perc_images}_{perc_pixels}_{i}.png"), bbox_inches="tight", dpi=200)


def plot_histograms(gts, scores, ds, method, perc_images, perc_pixels, i, save_dir):
    plt.clf()
    plt.figure(figsize=(8, 4))
    n_bins = 30
    x = 2 * np.array(list(range(n_bins)))
    width = 0.5
    for label in range(3):
        ixs = gts == label
        scs = scores[ixs]
        hst, bs = np.histogram(scs, bins=n_bins, range=(min(scores), max(scores)))
        plt.bar(x + (width * (label - 1)), hst, width=width, color=["g", "r", "b"][label], alpha=0.7)

    plt.xticks([])
    plt.yticks([])
    plt.grid()
    plt.savefig(join(save_dir, f"{ds}_{method}_{perc_images}_{perc_pixels}_{i}.png"), bbox_inches="tight", dpi=200)


def plot_histograms3d(gts, scores, ds, method, perc_images, perc_pixels, i, save_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n_bins = 30
    x = np.array(list(range(n_bins)))
    for label, (c, z) in enumerate(zip(["g", "r", "b"], [0, 8, 16])):
        ixs = gts == label
        scs = scores[ixs]
        hst, bs = np.histogram(scs, bins=n_bins, range=(min(scores), max(scores)))
        ax.bar(x, hst, zs=z, zdir='y', color=c, ec=c, alpha=0.8)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    plt.grid()
    # plt.show()

    plt.savefig(join(save_dir, f"{ds}_{method}_{perc_images}_{perc_pixels}_{i}.png"), bbox_inches="tight", dpi=200)


def do_single_run_plots(gts, scores, d, m, perc_images, perc_pixels, i, raw_scores_dir):
    plot_raw_scores(gts, scores, d, m, perc_images, perc_pixels, i, raw_scores_dir)
    plot_histograms(gts, scores, d, m, perc_images, perc_pixels, i, histograms_dir)
    plot_histograms3d(gts, scores, d, m, perc_images, perc_pixels, i, histograms3d_dir)

results_dir = RESULTS_PATH
plots_dir = PLOT_PATH
run_dir = join(results_dir, run_name)
save_dir = join(plots_dir, run_name)

raw_scores_dir = join(save_dir, "raw_scores")
histograms_dir = join(save_dir, "histograms")
histograms3d_dir = join(save_dir, "histograms_3d")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(raw_scores_dir, exist_ok=True)
os.makedirs(histograms_dir, exist_ok=True)
os.makedirs(histograms3d_dir, exist_ok=True)

def load_cflow_gts(names, ds):
    split_path = os.path.join("splits", ds if ds != "dagm10" else "dagm", f"split_-1_iteration_TEST.csv")
    df = pd.read_csv(split_path)

    gts = []
    for n in names:
        gt = list(df[df["image"] == n]["label"])[0]
        gts.append(gt)
    return np.array(gts)

if use_cached_results:
    joined_results_csv = csv_load(join(run_dir, "joined_results_arr.csv"))
    iter_results_csv = csv_load(join(run_dir, "iter_results_arr.csv"))

else:
    joined_results_csv = []
    iter_results_csv = []
    for m in methods:
        for d in datasets:
            for pi in perc_images:
                for pp in perc_pixels:
                    if pi == 0 and pp != -1:
                        continue
                    joined_metrics = {s: {eval_strat: [] for eval_strat in eval_strats} for s in score_functions}
                    for i in its:
                        cur_run_dir = join(run_dir, f"{m}_{d}_{pi}_{pp}", f"iter_{i}")
                        res = pickle_load(join(cur_run_dir, "evaluation.pkl"))
                        gts, scores = np.array(res["ts_gts"]), np.array(res["ts_scores"])
                        if m == "cflow":
                            gts = load_cflow_gts(res["ts_names"], d)
                        if plot_single_runs:
                            do_single_run_plots(gts, scores, d, m, perc_images, perc_pixels, i, raw_scores_dir)
                        for eval_strat in eval_strats:
                            es_gts, es_scores = filter_samples(gts, scores, eval_strat)
                            for score_function in score_functions:
                                score = get_score(score_function, es_gts, es_scores)
                                iter_results_csv.append([m, d, pi, pp, i, eval_strat, score_function, score])
                                joined_metrics[score_function][eval_strat].append(score)
                    for s, dct in joined_metrics.items():
                        for e, arr in dct.items():
                            joined_results_csv.append([m, d, pi, pp, e, s, np.mean(arr), np.std(arr)])
    csv_dump(join(run_dir, "joined_results_arr.csv"), joined_results_csv, ["method", "dataset", "perc_images", "perc_pixels", "eval_strat", "score_function", "score_mean", "score_std"])
    csv_dump(join(run_dir, "iter_results_arr.csv"), iter_results_csv, ["method", "dataset", "perc_images", "perc_pixels", "iter_ix", "eval_strat", "score_function", "score"])
    joined_results_csv = csv_load(join(run_dir, "joined_results_arr.csv"))
    iter_results_csv = csv_load(join(run_dir, "iter_results_arr.csv"))


# %%

def get_score_mean_std(c, m, d, pi, pp, eval_strat, s_f):
    ix = (c["method"] == m) & (c["dataset"] == d) & (c["perc_images"] == pi) & (c["perc_pixels"] == pp) & (c["eval_strat"] == eval_strat) & (c["score_function"] == s_f)
    return list(c.loc[ix]["score_mean"])[0], list(c.loc[ix]["score_std"])[0]


if True:
    draw_stds = True
    csv = joined_results_csv
    for m in methods:
        for d in datasets:
            for s_f in score_functions:
                score_save_dir = join(save_dir, s_f)
                os.makedirs(score_save_dir, exist_ok=True)

                plt.clf()
                plt.figure(figsize=(5, 5))

                baseline_mean_def, baseline_std_def = get_score_mean_std(csv, m, d, 0, -1, 2, s_f)
                baseline_mean_perlin, baseline_std_perlin = get_score_mean_std(csv, m, d, 0, -1, 3, s_f)

                plot_means = np.zeros((len(perc_pixels), len(perc_images) - 1))
                plot_stds = np.zeros((len(perc_pixels), len(perc_images) - 1))
                for ipi, pi in enumerate(perc_images[1:]):  # Ignore pi==0
                    for ipp, pp in enumerate(perc_pixels):
                        mean, std = get_score_mean_std(csv, m, d, pi, pp, 2 if pp == -1 else 3, s_f)
                        plot_means[ipp, ipi] = mean
                        plot_stds[ipp, ipi] = std

                plot_means = np.concatenate((np.array([baseline_mean_def] + [baseline_mean_perlin] * (len(perc_pixels) - 1)).reshape((-1, 1)), plot_means), axis=1)
                plot_stds = np.concatenate((np.array([baseline_std_def] + [baseline_std_perlin] * (len(perc_pixels) - 1)).reshape((-1, 1)), plot_stds), axis=1)

                x_len = plot_means.shape[1]

                for ix, (row_m, row_s) in enumerate(zip(plot_means, plot_stds)):
                    if draw_stds:
                        plt.errorbar(np.array(range(x_len)) + ((-0.1 + ix * 0.04) if ix > 0 else 0), row_m, row_s)
                    else:
                        plt.plot(range(x_len), row_m, ls=linestyle_tuple[ix][1])

                leg_els = ["Real" if pp == -1 else f"{str(pp)} %" for pp in perc_pixels]
                plt.legend(leg_els, loc="center", bbox_to_anchor=(0.5, 1.05), ncol=len(leg_els))
                plt.figtext(0.5, 0.96, f"{m}, {d}", ha="center")

                xticks = list(map(lambda x: f"{str(x)} %", [0] + perc_images[1:]))
                plt.xticks(ticks=range(x_len), labels=xticks)

                plt.xlabel(f"% defective images")
                plt.ylabel(s_f)

                plot_title = f"{m} {d}"
                # plt.title(plot_title)
                plt.grid()
                # plt.axhline(y=baseline_score_def, linestyle='-', alpha=0.5)
                # plt.axhline(y=baseline_score_perlin, linestyle='--', alpha=0.5)
                plt.savefig(join(score_save_dir, f"{plot_title}.png"), dpi=200)

# %%
integrals = []
for m in methods:
    for d in datasets:
        for s_f in score_functions:

            for pp in perc_pixels:
                baseline, _ = get_score_mean_std(csv, m, d, 0, -1, 2 if pp == -1 else 3, s_f)
                scores = [get_score_mean_std(csv, m, d, pi, pp, 2 if pp == -1 else 3, s_f)[0] for pi in perc_images[1:]]
                heights = np.array([baseline - s for s in scores])
                integral = sum(heights[:-1]) + heights[-1] / 2
                integral = integral / len(heights)
                integrals.append([m, d, pp, s_f, integral])

csv_dump(join(run_dir, "integrals.csv"), integrals, ["method", "dataset", "perc_pixels", "score_function", "integral"])
integrals_csv = csv_load(join(run_dir, "integrals.csv"))
# %%
join_datasets = False


def integral_score(c, m, d, pp, s_f):
    ix = (c["method"] == m) & (c["dataset"] == d) & (c["perc_pixels"] == pp) & (c["score_function"] == s_f)
    return list(c.loc[ix]["integral"])[0]


def get_method_robustness_score(csv, method, ds, perc_pixels, s_f, robustness_defective):
    score_def = integral_score(csv, method, ds, -1, s_f)
    scores_perlin = [integral_score(csv, method, ds, pp, s_f) for pp in perc_pixels[1:]]

    if robustness_defective:
        return -score_def
    else:
        return -np.mean(scores_perlin)


def get_method_accuracy_score(csv, m, d, s_f, def_acc_only):
    score_def = get_score_mean_std(csv, m, d, 0, -1, 2, s_f)[0]
    score_perlin = get_score_mean_std(csv, m, d, 0, -1, 3, s_f)[0]

    score = score_def if def_acc_only else score_perlin
    return score


scatter_dir = os.path.join(save_dir, "scatter")
os.makedirs(scatter_dir, exist_ok=True)

if join_datasets:
    for s_f in score_functions:
        plt.clf()
        fig = plt.figure(figsize=(10, 10))
        fig, ax = plt.subplots()
        plot_items = []
        for m in methods:
            robustness_scores = [get_method_robustness_score(integrals_csv, m, ds, perc_pixels, s_f, mean_perlin_score, robustness_defective_only) for ds in datasets]
            accuracy_scores = [get_method_accuracy_score(joined_results_csv, m, ds, s_f, mean_perlin_score) for ds in datasets]
            plot_items.append([m, np.mean(robustness_scores), np.mean(accuracy_scores)])
        names, xs, ys = list(zip(*plot_items))
        plt.scatter(xs, ys)
        for n, x, y in zip(names, xs, ys):
            ax.text(x, y, n)
        plt.xlabel("Robustness")
        plt.ylabel("Accuracy")
        plt.show(dpi=200, bbox_inches="tight")
        # plt.savefig("v1_auc.png", bbox_inches="tight", dpi=200)
else:
    for d in datasets:
        for s_f in score_functions:
            plt.clf()
            fig = plt.figure(figsize=(10, 10))
            fig, ax = plt.subplots()
            plot_items = []
            for m in methods:
                robustness_scores = get_method_robustness_score(integrals_csv, m, d, perc_pixels, s_f, False)
                accuracy_scores = get_method_accuracy_score(joined_results_csv, m, d, s_f, False)
                plot_items.append([m, np.mean(robustness_scores), np.mean(accuracy_scores)])
            names, xs, ys = list(zip(*plot_items))
            plt.scatter(xs, ys)
            for n, x, y in zip(names, xs, ys):
                ax.text(x, y, n)

            plot_items = []
            for m in methods:
                robustness_scores = get_method_robustness_score(integrals_csv, m, d, perc_pixels, s_f, True)
                accuracy_scores = get_method_accuracy_score(joined_results_csv, m, d, s_f, True)
                plot_items.append([m, np.mean(robustness_scores), np.mean(accuracy_scores)])
            names, xs, ys = list(zip(*plot_items))
            plt.scatter(xs, ys)
            for n, x, y in zip(names, xs, ys):
                ax.text(x, y, n)

            plt.xlabel("Robustness")
            plt.ylabel("Accuracy")
            plt.title(d)
            # plt.show(dpi=200, bbox_inches="tight")
            plt.savefig(os.path.join(scatter_dir, f"{d}.png"), bbox_inches="tight", dpi=200)
