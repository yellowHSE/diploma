import os
import random
import json
import numpy as np
import cv2
from os.path import join
from shutil import copy2
import pandas as pd

import sys
sys.path.append("..")
from path_constants import DATASET_PATH, RENAMING_PATH

#### Baza       POZ     NEG
####
#### KSDD2	    356 	2979
#### BSData	    325 	710
#### Softgel	345 	846
#### DAGM 7-10	300	    2000
####

"""
Joins all good and bad images in a single, specified folder.
Mask are copied only for defective samples, named <ORIG_NAME>_GT.png.
Images in range 10 000 - 19 999 are defective, those in 20 000+ are negative.
"""


# %%
def json_to_mask(label_path, orig_shape, resize_shape):
    with open(label_path, "r") as f:
        label_json = json.load(f)

    lst = label_json["shapes"][0]["points"]
    mask = np.zeros(orig_shape)
    pts = np.array(lst).astype(np.int32)
    filled = cv2.drawContours(mask, [pts], -1, (255, 255, 255), thickness=cv2.FILLED)
    return cv2.resize(filled, resize_shape)



def read_bsdata():
    bsdata_orig_path = "/storage/datasets/BSData/"

    orig_images = os.listdir(join(bsdata_orig_path, "data"))
    all_good, all_bad = [], []

    for img in sorted(orig_images):
        if os.path.isdir(f"{bsdata_orig_path}data/{img}"):
            continue
        img_path = join(bsdata_orig_path, "data", img)
        lab_path = join(bsdata_orig_path, "label", f"{img[:-4]}.json")
        if os.path.exists(lab_path):
            all_bad.append((img, img_path, lab_path))
        else:
            all_good.append((img, img_path, None))

    return all_good, all_bad, "bsdata"


def read_dagm(class_ix):
    orig_path = f"/storage/datasets/GOSTOP/DAGM/Class{class_ix}"

    all_good, all_bad = [], []
    for kind in ["Train", "Test"]:
        for img_n in os.listdir(join(orig_path, kind)):
            if "Label" in img_n or "label" in img_n:
                continue
            label_path = join(orig_path, kind, f"{img_n[:-4]}_label.PNG")
            img_path = join(orig_path, kind, img_n)
            if os.path.exists(label_path):
                all_bad.append((img_n, img_path, label_path))
            else:
                all_good.append((img_n, img_path, None))

    return all_good, all_bad, "dagm"


def read_ksdd2():
    orig_path = "/storage/datasets/GOSTOP/KSDD2/no_dilate"
    train_path = os.path.join(orig_path, "train")
    test_path = os.path.join(orig_path, "test")

    train_good, train_bad, test_good, test_bad = [], [], [], []

    for img_name in os.listdir(train_path):
        if "GT" in img_name:
            continue

        img_short_name = img_name[:-4]
        img_path = join(train_path, f"{img_short_name}.png")
        mask_path = join(train_path, f"{img_short_name}_GT.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.max() > 0:
            train_bad.append((img_name, img_path, mask_path))
        else:
            train_good.append((img_name, img_path, None))

    for img_name in os.listdir(test_path):
        if "GT" in img_name:
            continue

        img_short_name = img_name[:-4]
        img_path = join(test_path, f"{img_short_name}.png")
        mask_path = join(test_path, f"{img_short_name}_GT.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.max() > 0:
            test_bad.append((img_name, img_path, mask_path))
        else:
            test_good.append((img_name, img_path, None))

    return train_good + test_good, train_bad + test_bad, "ksdd2"


def read_softgel():
    orig_path = "/storage/datasets/SensumSODF/softgel"
    all_good, all_bad = [], []
    for img in os.listdir(join(orig_path, "negative", "data")):
        all_good.append((img, join(orig_path, "negative", "data", img), None))
    for img in os.listdir(join(orig_path, "positive", "data")):
        all_bad.append((img, join(orig_path, "positive", "data", img), join(orig_path, "positive", "annotation", img)))
    return all_good, all_bad, "softgel"


# %%
bsdata_data = read_bsdata()
ksdd2_data = read_ksdd2()
dagm_data = read_dagm(10)
softgel_data = read_softgel()
# %%
ROOT_DS_PATH = DATASET_PATH
ROOT_DF_PATH = RENAMING_PATH
os.makedirs(ROOT_DS_PATH, exist_ok=True)
os.makedirs(ROOT_DF_PATH, exist_ok=True)


# %%
def process_ds(data):
    all_good, all_bad, ds_name = data
    ds_path = join(ROOT_DS_PATH, ds_name)
    os.makedirs(ds_path, exist_ok=True)

    renaming = []
    ix = 10000
    for img_n, img_path, mask_path in all_bad:
        new_image_name = f"{ix}.png"
        new_mask_name = f"{ix}_GT.png"
        copy2(img_path, join(ds_path, new_image_name))
        if mask_path.endswith("json"):
            img = cv2.imread(img_path)
            mask = json_to_mask(mask_path, img.shape[:2], img.shape[:2])
            cv2.imwrite(join(ds_path, new_mask_name), mask)
        else:
            copy2(mask_path, join(ds_path, new_mask_name))

        renaming.append([img_path, new_image_name])
        ix += 1

    ix = 20000
    for img_n, img_path, _ in all_good:
        new_image_name = f"{ix}.png"
        copy2(img_path, join(ds_path, new_image_name))
        renaming.append([img_path, new_image_name])
        ix += 1

    pd.DataFrame(renaming, columns=["from", "to"]).to_csv(join(f"{ROOT_DF_PATH}", f"{ds_name}.csv"), index=False)


# %%
process_ds(softgel_data)
process_ds(ksdd2_data)
process_ds(dagm_data)
process_ds(bsdata_data)
