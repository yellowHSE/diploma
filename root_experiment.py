from methods.riad import Riad
from methods.padim import Padim
from methods.mahalanobis import Mahalanobis
from methods.patch_core import PatchCore
from methods.cut_paste import CutPaste
from methods.cflow import CFLOW
from methods.draem import Draem
from methods.classification import Classification
import os
from datasets_python.perlin_datasets import KSDD2_PerlinDataset, DAGM_PerlinDataset, BSDATA_PerlinDataset, \
    SoftgelPerlinDataset
from datasets_python.ds_mvtec_orig import MVTecDataset
from datasets_python.ds_kolektor import KolektorDataset
from datasets_python.ds_kolblend import KolBlendDataset
from datasets_python.debug_mvtec_draem import MVTecDRAEMTrainDataset, MVTecDRAEMTestDataset
from datasets_python.ds_ploscice import TilesDataset
from torch.utils.data import DataLoader
from utils import get_auc, json_dump
from utils import pickle_dump
from config import Config


def get_method_class(method):
    if method.lower() == "riad":
        return Riad
    elif method.lower() == "padim":
        return Padim
    elif method.lower() == "mahalanobis":
        return Mahalanobis
    elif method.lower() == "patchcore":
        return PatchCore
    elif method.lower() == "cutpaste":
        return CutPaste
    elif method.lower() == "draem":
        return Draem
    elif method.lower() == "cflow":
        return CFLOW
    elif method.lower() == "classification":
        return Classification
    else:
        raise Exception(f"Unknown method:{method}")


def get_dataloader(kind, c, extra_params):
    dataset = c.DATASET
    if dataset.lower() == "bsdata":
        ds = BSDATA_PerlinDataset(kind, c, extra_params)
    elif dataset.lower() == "ksdd2":
        ds = KSDD2_PerlinDataset(kind, c, extra_params)
    elif dataset.lower() == "dagm10":
        ds = DAGM_PerlinDataset(kind, c, extra_params)
    elif dataset.lower() == "softgel":
        ds = SoftgelPerlinDataset(kind, c, extra_params)
    elif dataset.lower().startswith("mvtec"):
        extra_params["object_class"] = dataset.lower().split("_")[-1]
        ds = MVTecDataset(kind, c, extra_params)
    elif dataset.lower().startswith("kolektor"):
        extra_params["fold"] = int(dataset.lower().split("_")[-1])
        ds = KolektorDataset(kind, c, extra_params)
    elif dataset.lower() == "kolblend":
        ds = KolBlendDataset(kind, c, extra_params)
    elif dataset.lower().startswith("dmvt"):
        extra_params["object_class"] = dataset.lower().split("_")[-1]
        if kind == "train":
            ds = MVTecDRAEMTrainDataset(kind, c, extra_params)
        else:
            ds = MVTecDRAEMTestDataset(kind, c, extra_params)
    elif dataset.lower() == "tiles":
        ds = TilesDataset(kind, c, extra_params)
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    dl = DataLoader(ds, batch_size=c.BATCH_SIZE if kind == "train" else 1, shuffle=kind == "train",
                    num_workers=c.N_WORKERS, pin_memory=True)
    # dl = DataLoader(ds, batch_size=c.BATCH_SIZE if kind == "train" else 1, shuffle=False, num_workers=c.N_WORKERS, pin_memory=True)

    return dl


def compute_step_metrics(step_data):
    tr_sc, tr_gt, tr_n = step_data["tr_scores"], step_data["tr_gts"], step_data["tr_names"]
    ts_sc, ts_gt, ts_n = step_data["ts_scores"], step_data["ts_gts"], step_data["ts_names"]

    tr_auc, ts_auc = get_auc(tr_gt, tr_sc), get_auc(ts_gt, ts_sc)
    num_tr_pos, num_tr_neg = sum(tr_gt > 0), sum(tr_gt == 0)

    return tr_auc, ts_auc, num_tr_pos, num_tr_neg


def run_single_class(run_path: str, c: Config):
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(os.path.join(run_path, "segmentation"), exist_ok=True)
    json_dump(os.path.join(run_path, "config.json"), c._get_props())

    method_class = get_method_class(c.METHOD)

    ds_extra_params = {"train_ds_no_aug": False}
    if c.METHOD == "classification":
        ds_extra_params["load_bad_training"] = True
    train_dataloader = get_dataloader("train", c, ds_extra_params)
    test_dataloader = get_dataloader("test", c, ds_extra_params)
    extra_args = {}
    if c.METHOD == "cutpaste":
        extra_args["train_fit_dataloader"] = get_dataloader("train", c, {"train_ds_no_aug": True})

    step_data = method_class().train_and_eval(run_path, c, train_dataloader, test_dataloader, extra_args)
    pickle_dump(os.path.join(run_path, f"evaluation.pkl"), step_data)
    tr_auc, ts_auc, n_tr_pos, n_tr_neg = compute_step_metrics(step_data)
    print(f"RUN: {run_path} ||| Train AUC: {tr_auc}, Test AUC: {ts_auc}, TRAIN_POS: {n_tr_pos}, TRAIN_NEG: {n_tr_neg}")
