import argparse
from config import Config
import os
import socket
import sys

from path_constants import RESULTS_PATH

if sys.gettrace() is not None:
    print("Running FIXED configuration from python")

    GPU = 5

    c: Config = Config()
    c.OUTPUT_PATH = RESULTS_PATH
    c.RUN_NAME = "ksdd2_test"
    c.DATASET = "ksdd2"
    c.METHOD = "draem"
    c.PERC_IMAGES = 0
    c.PERC_PIXELS = -1
    c.ITERATION_IX = 2
    c.SINGLE_TEST = 1

    c.SAVE_SEGMENTATION = True
    c.EVAL_IF_MODEL_SAVED = True
    c.KOLBLEND_EVAL_SYNT = True
    # c._merge_from_args(args)

    c.LEARNING_RATE = 5e-5

    c.N_EPOCHS = 50
    c.BATCH_SIZE = 2
    c.N_WORKERS = 4
    c.PATCHCORE_CORESET_SAMPLING_RATIO = 0.0005
    c.DEVICE = f"cuda:{GPU}"
    print(c.DEVICE)
    run_id = f"{c.METHOD}_{c.DATASET}_{c.PERC_IMAGES}_{c.PERC_PIXELS}"
    run_path = os.path.join(c.OUTPUT_PATH, c.RUN_NAME, run_id, f"iter_{c.ITERATION_IX}")

    print(f"Running {run_path}")
    print(c._get_props())
    from root_experiment import run_single_class

    run_single_class(run_path, c)



else:
    print("Running arguments configuration from bash")

    parser = argparse.ArgumentParser()

    parser.add_argument('--OUTPUT_PATH', type=str, required=True)
    parser.add_argument('--DATASETS_PATH', type=str, required=False, default=None)
    parser.add_argument('--PERLIN_SOURCE_IMAGES_PATH', type=str, required=False, default=None)
    parser.add_argument('--RUN_NAME', type=str, required=True)

    parser.add_argument('--GPU', type=int, required=True)
    parser.add_argument('--DATASET', type=str, required=True)
    parser.add_argument('--METHOD', type=str, required=True)
    parser.add_argument('--PERC_IMAGES', type=int, required=True)
    parser.add_argument('--PERC_PIXELS', type=int, required=True)
    parser.add_argument('--ITERATION_IX', type=int, required=True)
    parser.add_argument('--SINGLE_TEST', type=int, required=True)

    parser.add_argument("--EXTRA_PARAMS", type=str, required=False, default=None)

    args = parser.parse_args()
    c: Config = Config()

    
    c.SAVE_SEGMENTATION = True

    c._merge_from_args(args)

    c.DEVICE = f"cuda:{args.GPU}"
    print(c.DEVICE)
    run_id = f"{c.METHOD}_{c.DATASET}_{c.PERC_IMAGES}_{c.PERC_PIXELS}"
    run_path = os.path.join(c.OUTPUT_PATH, c.RUN_NAME, run_id, f"iter_{c.ITERATION_IX}")

    print(f"Running {run_path}")
    print(c._get_props())
    from root_experiment import run_single_class

    run_single_class(run_path, c)
