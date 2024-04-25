import os
import random
import numpy as np
from os.path import join
import pandas as pd
import glob
import sys
sys.path.append("..")
from path_constants import SPLITS_PATH, DATASET_PATH, PERLIN_SOURCE_DATASET_PATH
#### Baza       POZ     NEG
####
#### KSDD2	    356 	2979
#### BSData	    325 	710
#### Softgel	345 	846
#### DAGM 7-10	300	    2000
####

"""
Creates splits for specified percentages of positive images or pixels and for mulitple iterations.
Dataset should be in format that is outputed by join_data.py.
"""

# %%
N_TEST_GOOD, N_TEST_BAD, N_TEST_PERLIN = 100, 100, 100
N_TRAIN_GOOD = 400

# PERCENTAGE_IMAGES = [0, 1, 5, 10, 15, 20, 25]
PERCENTAGE_IMAGES = [0, 1, 5, 10, 15, 20, 25]
# PERCENTAGE_IMAGES = [25]
# PERCENTAGE_PIXELS = [1, 5, 10, 25, 50, 75]
# PERCENTAGE_PIXELS = [1, 5, 10, 25, 50, 75]
PERCENTAGE_PIXELS = [-1]

N_ITERATIONS = 30

SPLITS_SAVE_PATH = SPLITS_PATH
os.makedirs(SPLITS_SAVE_PATH, exist_ok=True)

np.random.seed(1337)
random.seed(1337)

perlin_seed_range = 100000000
csv_columns = ["image", "label", "perlin_perc", "perlin_alpha", "perlin_source", "perlin_seed"]


# %%
def read_dataset(ds):
    ds_path = join(DATASET_PATH, ds)

    good_images, bad_images = [], []
    for img_n in os.listdir(ds_path):
        if "GT" in img_n:
            continue
        if os.path.exists(join(ds_path, f"{img_n[:-4]}_GT.png")):
            bad_images.append(img_n)
        else:
            good_images.append(img_n)

    return sorted(good_images), sorted(bad_images)


# %%
def read_perlin_source_images():
    train_source_images = glob.glob(f"{PERLIN_SOURCE_DATASET_PATH}/dagm1/train/*")
    test_source_images = glob.glob(f"{PERLIN_SOURCE_DATASET_PATH}/dagm1/test/*")

    train_source_images = filter(lambda x: "png" in x.lower(), train_source_images)
    test_source_images = filter(lambda x: "png" in x.lower(), test_source_images)

    shorten = lambda y: list(map(lambda x: x.split("/")[-1], y))

    return shorten(train_source_images), shorten(test_source_images)


# %%
def random_percentage_pixels(percentage_pixels, n):
    arr = []
    each = n // len(percentage_pixels)
    for p in percentage_pixels:
        arr.extend([p] * each)
    missing = n - len(arr)
    random.shuffle(percentage_pixels)
    arr.extend(percentage_pixels[:missing])
    random.shuffle(arr)
    return arr


# %%

def split_test(good_images, bad_images, perlin_source_test_images, percentage_pixels, splits_folder, iteration_ix):
    random.shuffle(good_images)
    random.shuffle(bad_images)
    """
    Generate fixed test set for the iteration, i.e. every combination of perc_images and perc_pixels has the same test set.
    For perlin noise images we also generate random perc_pixels, alpha, source_image and seed for noise generation.
    """
    test_good = good_images[:N_TEST_GOOD]
    test_perlin = good_images[N_TEST_GOOD:N_TEST_GOOD + N_TEST_PERLIN]
    test_bad = bad_images[:N_TEST_BAD]

    perlin_percentage_test_random = random_percentage_pixels(percentage_pixels, N_TEST_PERLIN)
    perlin_alphas_random = list(map(lambda x: round(x, 2), np.random.uniform(0.1, 0.9, N_TEST_PERLIN)))
    perlin_sources_random = random.choices(perlin_source_test_images, k=N_TEST_PERLIN)

    all_test_images = []
    for img in test_good:
        all_test_images.append([img, 0, -1, -1, "", -1])
    for img in test_bad:
        all_test_images.append([img, 1, -1, -1, "", -1])
    for img, perc, alpha, src in zip(test_perlin, perlin_percentage_test_random, perlin_alphas_random, perlin_sources_random):
        all_test_images.append([img, 2, perc, alpha, src, np.random.randint(perlin_seed_range)])

    pd.DataFrame(all_test_images, columns=csv_columns).to_csv(join(splits_folder, f"split_{iteration_ix}_iteration_TEST.csv"), index=False)

    remaining_good = good_images[N_TEST_GOOD + N_TEST_PERLIN:]
    remaining_bad = bad_images[N_TEST_BAD:]

    return remaining_good, remaining_bad


def make_splits(ds, percentage_images, percentage_pixels, good_images, bad_images, perlin_source_train_images, perlin_source_test_images, splits_save_path, single_test_set):
    splits_folder = join(splits_save_path, ds)
    os.makedirs(splits_folder, exist_ok=True)

    if single_test_set:
        remaining_good, remaining_bad = split_test(good_images, bad_images, perlin_source_test_images, percentage_pixels, splits_folder, -1)

    for iteration_ix in range(N_ITERATIONS):

        if not single_test_set:
            remaining_good, remaining_bad = split_test(good_images, bad_images, perlin_source_test_images, percentage_pixels, splits_folder, iteration_ix)
        else:
            random.shuffle(remaining_good)
            random.shuffle(remaining_bad)

        """
        Train set, we generate splits with perlin noise in training set and also with defective parts in training set.
        For images with perlin noise we fix alpha, source image and seed. 
        """

        remaining_perlin_alphas_random = list(map(lambda x: round(x, 2), np.random.uniform(0.1, 0.9, len(remaining_good))))
        remaining_perlin_sources_random = random.choices(perlin_source_train_images, k=len(remaining_good))
        remaining_perlin_seed_random = np.random.randint(perlin_seed_range, size=len(remaining_good))

        for current_perc_images in percentage_images:
            n_bad_images = int(current_perc_images * N_TRAIN_GOOD / 100)
            n_good_images = N_TRAIN_GOOD - n_bad_images

            if current_perc_images > 0:
                """
                Generate perlin noise training set
                """
                all_train_perlin_images = []
                for img in remaining_good[:n_good_images]:
                    all_train_perlin_images.append([img, 0, -1, -1, "", -1])
                for img, alpha, src, seed in zip(remaining_good[-n_bad_images:], remaining_perlin_alphas_random[-n_bad_images:], remaining_perlin_sources_random[-n_bad_images:], remaining_perlin_seed_random[-n_bad_images:]):
                    all_train_perlin_images.append([img, 2, -1, alpha, src, seed])
                pd.DataFrame(all_train_perlin_images, columns=csv_columns).to_csv(join(splits_folder, f"split_{iteration_ix}_iteration_{current_perc_images}_images_PERLIN_TRAIN.csv"), index=False)

            """
            Generate defective training set
            """
            all_train_bad_images = []
            for img in remaining_good[:n_good_images]:
                all_train_bad_images.append([img, 0, -1, -1, "", -1])
            for img in remaining_bad[:n_bad_images]:
                all_train_bad_images.append([img, 1, -1, -1, "", -1])
            pd.DataFrame(all_train_bad_images, columns=csv_columns).to_csv(join(splits_folder, f"split_{iteration_ix}_iteration_{current_perc_images}_images_DEFECTIVE_TRAIN.csv"), index=False)


# %%
perlin_source_train_images, perlin_source_test_images = read_perlin_source_images()
for ds in ["ksdd2","softgel", "dagm",  "bsdata"]:
    good, bad = read_dataset(ds)
    make_splits(ds, PERCENTAGE_IMAGES, PERCENTAGE_PIXELS, good, bad, perlin_source_train_images, perlin_source_test_images, SPLITS_SAVE_PATH, True)
