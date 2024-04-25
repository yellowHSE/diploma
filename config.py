import socket
import os

from path_constants import DATASET_PATH, PERLIN_SOURCE_DATASET_PATH, SPLITS_PATH


class Config:
    _on_vicos = True

    METHOD = None
    DATASET = None
    PERC_IMAGES = None
    PERC_PIXELS = None
    ITERATION_IX = None
    SINGLE_TEST = None

    RUN_NAME = None
    OUTPUT_PATH = None
    DATASETS_PATH = DATASET_PATH
    PERLIN_SOURCE_IMAGES_PATH = os.path.join(PERLIN_SOURCE_DATASET_PATH, "dagm1")
    SPLITS_DIR = SPLITS_PATH

    SAVE_SEGMENTATION = True
    EVAL_IF_MODEL_SAVED = True

    DEVICE = None
    SIZE_MULT_F = 1

    N_EPOCHS = 100  # RIAD and CutPaste

    N_WORKERS = 12
    BATCH_SIZE = 6
    LEARNING_RATE = 1e-4

    KOLBLEND_EVAL_SYNT = True

    """
    RIAD specific configuration
    """
    RIAD_CUTOUT_SIZES = [2, 4, 8, 16]
    RIAD_NUM_DISJOINT_MASKS = 3
    RIAD_LR_STEP = 10  # If list will decay at every element, if int will decay at every specified epochs
    RIAD_LR_GAMMA = 0.1
    RIAD_OPTIMIZER = "adam"  # sgd, adam
    RIAD_MOMENTUM = 0
    RIAD_WEIGHT_DECAY = 1e-5
    RIAD_NESTEROV = False
    RIAD_LOSS_FN = "ce"  # ce, bce, l2

    """
    PaDiM specific configuration
    """
    PADIM_ARCH = "resnet18"

    """
    PatchCore specific configuration
    """
    PATCHCORE_CORESET_SAMPLING_RATIO = 0.001

    """
    Various 
    """
    ORIG_MVTEC_DIR = "/storage/datasets/mvtec"
    DRAEM_ANOMALY_SOURCE_PATH = "/storage/datasets/DTD/images/*/*.jpg"

    """
    Draem ploscice
    """
    W_SSIM_LOSS=1
    W_FOCAL_LOSS=1
    DRAEM_AUG_FLIP=0
    DRAEM_AUG_CJ = 0
    DRAEM_AUG_CJ_VAL = 0.1
    DRAEM_AUG_NORM = 0
    DRAEM_AUG_BLUR = 0
    DRAEM_AUG_BLUR_KERNEL = 15
    DRAEM_AUG_BLUR_SIGMA = 0.7


    def _get_props(self) -> dict:
        props = {prop: self.__getattribute__(prop) for prop in filter(lambda x: not x.startswith("_"), self.__dir__())}
        return {k: v for k, v in sorted(props.items(), key=lambda x: x[0])}

    def _merge_from_args(self, args):
        prop_dict = self._get_props()

        for arg_prop in args.__dir__():
            if arg_prop in prop_dict:
                arg_val = args.__getattribute__(arg_prop)
                if arg_val is not None:
                    self.__setattr__(arg_prop, arg_val)

        self._merge_extra_params(args.EXTRA_PARAMS)

    def _merge_extra_params(self, extra_params):
        if extra_params is not None and extra_params != "":
            pars = extra_params.split(";")
            for p in pars:
                par_name, par_val = p.split(":")

                if par_val.isnumeric():
                    par_val = int(par_val)
                elif par_val.replace(".", "").isnumeric():
                    par_val = float(par_val)
                elif par_val in ["True", "true"]:
                    par_val = True
                elif par_val in ["False", "false"]:
                    par_val = False
                self.__setattr__(par_name, par_val)
