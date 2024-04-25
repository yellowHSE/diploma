import os
from os.path import join
from shutil import copy2

import sys
sys.path.append("..")
from path_constants import PERLIN_SOURCE_DATASET_PATH

"""
Reads all negative images from a single dataset, then saves them to a separate folder.
"""


# %%
def read_good_dagm(kind):
    source_ds_path = join("/storage/datasets/GOSTOP/DAGM/Class1", kind)
    good_images = []
    for img_n in os.listdir(source_ds_path):
        if "Label" in img_n or "label" in img_n:
            continue

        ix = img_n[:4]
        label_name = f"{ix}_label.PNG"
        if os.path.exists(os.path.join(source_ds_path, label_name)):
            continue
        good_images.append([img_n, join(source_ds_path, img_n)])

    return good_images


def copy_good(ds_name, kind, images):
    ds_path = join(PERLIN_SOURCE_DATASET_PATH, ds_name, kind)
    os.makedirs(ds_path, exist_ok=True)

    for img_n, img_path in images:
        dst = join(ds_path, img_n)
        copy2(img_path, dst)


# %%
good_train = read_good_dagm("Train")
good_test = read_good_dagm("Test")

copy_good("dagm1", "train", good_train)
copy_good("dagm1", "test", good_test)


