import os
import numpy as np
from path_constants import RESULTS_PATH

# Run name of the experiment
run_name = "test_multiple_methods_multiple_datasets_2"

# Chose which methods to use in the experiment
# Supported methods:
# methods = ["riad", "cflow", "draem", "cutpaste", "patchcore", "mahalanobis", "padim"]
methods = ["mahalanobis", "padim"]

# Chose which datasets to use in the experiment
# Supported datasets:
# datasets = ["softgel", "bsdata", "dagm10", "ksdd2"]
# songeun test
#datasets = ["ksdd2", "bsdata"]
datasets = ["ksdd2"]

# Chose the percent of anomalous images inside the training set
# perc_images = [0, 1, 5, 15, 25]
# songeun test
perc_images = [0, 5]

# Number of iterations for each experiment setup
# songeun  test
#its = list(range(5))
its = list(range(1))

# Leftover from previous experiments, just leave it at -1 for now
perc_pixels = [-1]

# GPUs that are going to be used for the experiment
# https://vicos:patroller_v1c0s@patroller.proxy.vicos.si/
# The webpage shows the currently unsused gpus
# The experiments take quite some time, so do not use up all the gpus, but also don't run it on only one
# (Tip: Calculon is mostly empty nowadays)
#gpus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
gpus = [0, 1, 2, 3]
n_gpus = len(gpus)

# Some methods require extra parames, which can be set up here
method_extra_params = {"cutpaste": "BATCH_SIZE:12;N_EPOCHS:256",
                       "riad": "BATCH_SIZE:4;N_EPOCHS:60",
                       "cflow": "BATCH_SIZE:32;N_EPOCHS:10",
                       "draem": "BATCH_SIZE:4;N_EPOCHS:100"}

num_total = np.prod(list(map(len, [methods, datasets, perc_images, perc_pixels, its])))

prefix = "python "
script_name = "shell_runner.py"
output_path = RESULTS_PATH

single_test = 1
# %%
to_file = True

def out(s, f, to_file):
    if to_file:
        with open(f, "w+") as fl:
            fl.writelines("\n".join(s))
    else:
        list(map(print, s))


# %%
lines_to_print = []
iter_counter = 0
for m in methods:
    for d in datasets:
        for pi in perc_images:
            for pp in perc_pixels:
                if pi == 0 and pp != perc_pixels[0]:  # For perc_images == 0 generate only 1 run
                    continue
                for i in its:
                    extra_params_string = f"--EXTRA_PARAMS=\"{method_extra_params[m]}\"" if m in method_extra_params.keys() else ""
                    command = f"CUDA_VISIBLE_DEVICES={gpus[iter_counter % n_gpus]} {prefix}{script_name} --RUN_NAME={run_name} --OUTPUT_PATH={output_path} --SINGLE_TEST={single_test} {extra_params_string} --METHOD={m:12} --DATASET={d:8} --PERC_IMAGES={str(pi):2} --PERC_PIXELS={str(pp):2} --ITERATION_IX={i} --GPU=0 {'&' if n_gpus > 1 else ''}"
                    lines_to_print.append(command)
                    iter_counter += 1
                    if iter_counter % n_gpus == 0 and n_gpus > 1:
                        lines_to_print.append("wait")
if iter_counter % n_gpus != 0 and n_gpus > 1:
    lines_to_print.append("wait")

out(lines_to_print, f"experiment.sh", to_file)
