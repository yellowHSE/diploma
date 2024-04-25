import os
from utils import pickle_load
import pandas
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# root_results_path = "results/KOLEKTOR_MOLD2_F"
# run_name = "UNSUPERVISED_KOLEKTOR_SYNT"
# run_name = "KM2_SMALL_FIXED"
run_name = "unsupervised_real_only"
# root_results_path = f"results/{run_name}"
root_results_path = f"/storage/private/kolektor/Rezultati/anomaly_detection/{run_name}"

runs = os.listdir(root_results_path)
methods = set(map(lambda x: "_".join(x.split("_")[:2]), runs))


# %%
def get_metrics(res):
    gts, scores = res["ts_gts"], res["ts_scores"]

    precision_, recall_, thresholds = precision_recall_curve(gts, scores)
    f_measures = 2 * (precision_ * recall_) / (precision_ + recall_ + 0.0000000001)

    ix_best = np.argmax(f_measures)
    best_threshold = thresholds[ix_best]

    acc = ((scores >= best_threshold) == gts).sum() / len(gts)
    auc = roc_auc_score(gts, scores)
    ap = average_precision_score(gts, scores)

    return acc, auc, ap


# %%
N_FOLDS = 4
mf = 100
joined_csv = []
for method in methods:
    # for method in ["riad_kolektor"]:
    try:
        m_acc, m_auc, m_ap = [], [], []
        for fold_ix in range(N_FOLDS):
            # eval_path = os.path.join(root_results_path, f"{method}_{fold_ix}_0_-1", "iter_0", "evaluation.pkl")
            eval_path = os.path.join(root_results_path, f"{method}_{fold_ix}{'_0' if N_FOLDS == 4 else ''}_-1", "iter_0", "evaluation.pkl")
            res = pickle_load(eval_path)
            f_acc, f_auc, f_ap = get_metrics(res)
            m_acc.append(f_acc * mf)
            m_auc.append(f_auc * mf)
            m_ap.append(f_ap * mf)
        joined_csv.append([method.split("_")[0]] + [np.mean(m_acc), np.mean(m_auc), np.mean(m_ap)] + m_acc + m_auc + m_ap)
    except Exception as e:
        print(f"Error during {method}")
# %%
col_names = ["method", "acc", "auc", "ap"] + sum([[f"{mt}_{i}" for i in range(N_FOLDS)] for mt in ["acc", "auc", "ap"]], [])
pd.DataFrame(joined_csv, columns=col_names).round(2).to_csv(f"kolektor_{run_name}.csv", index=False)
