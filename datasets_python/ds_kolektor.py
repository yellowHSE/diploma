import glob
import math

import cv2
import os
from datasets_python.root_dataset import RootDataset
from config import Config
import random

# DS_PATH = "/storage/private/kolektor/MFA2_Top/"
DS_PATH = "/storage/private/kolektor/MFA2_sorted/"



# class KolektorDataset(RootDataset):
#     ds_name = "kolektormold"
#     resize = (416, 160)
#
#     # resize = (192, 96)
#     # cropsize = 144 * mult_f
#
#     def __init__(self, kind: str, c: Config, extra_params):
#         self.resize = (int(self.resize[0] * c.SIZE_MULT_F), int(self.resize[1] * c.SIZE_MULT_F))
#         self.resize_torchvision = (self.resize[1], self.resize[0])
#         self.extra_params = extra_params
#         self.fold = self.get_extra_param("fold")
#
#         super().__init__(kind, c, extra_params)
#
#     def read_contents(self):
#         good_path = os.path.join(DS_PATH, "good")
#         bad_path = os.path.join(DS_PATH, "bad")
#
#         good_samples = sorted(glob.glob(os.path.join(good_path, "*.png")))
#         bad_samples = sorted(glob.glob(os.path.join(bad_path, "*.png")))
#
#         good_per_fold = math.ceil(len(good_samples) / 4)
#         bad_per_fold = math.ceil(len(bad_samples) / 4)
#         random.seed(1337)
#         random.shuffle(good_samples)
#         random.shuffle(bad_samples)
#
#         selected_good_test_fold = good_samples[self.fold * good_per_fold:(self.fold + 1) * good_per_fold]
#         selected_bad_test_fold = bad_samples[self.fold * bad_per_fold:(self.fold + 1) * bad_per_fold]
#
#         selected_good = selected_good_test_fold if self.kind == "test" else list(set(good_samples) - set(selected_good_test_fold))
#         selected_bad = selected_bad_test_fold if self.kind == "test" else list(set(bad_samples) - set(selected_bad_test_fold))
#
#         samples = []
#         for image_path in selected_good:
#             image_name = image_path.split("/")[-1][:-4]
#             img = cv2.imread(image_path)
#             img = cv2.resize(img, self.resize)
#             sample = {"image": img, "label": 0, "name": image_name, "image_path": image_path}
#             samples.append(sample)
#
#         if self.kind == "test" or self.c.METHOD == "classification":
#             for image_path in selected_bad:
#                 image_name = image_path.split("/")[-1][:-4]
#                 img = cv2.imread(image_path)
#                 img = cv2.resize(img, self.resize)
#                 sample = {"image": img, "label": 1, "name": image_name, "image_path": image_path}
#                 samples.append(sample)
#
#         self.samples = samples


class KolektorDataset(RootDataset):
    ds_name = "kolektormold_sorted"
    resize = (416, 160)

    # resize = (192, 96)
    # cropsize = 144 * mult_f

    def __init__(self, kind: str, c: Config, extra_params):
        self.resize = (int(self.resize[0] * c.SIZE_MULT_F), int(self.resize[1] * c.SIZE_MULT_F))
        self.resize_torchvision = (self.resize[1], self.resize[0])
        self.extra_params = extra_params
        self.fold = self.get_extra_param("fold")

        super().__init__(kind, c, extra_params)

    def read_contents(self):
        good_path = os.path.join(DS_PATH, "good")
        bad_path = os.path.join(DS_PATH, "bad")

        good_samples = sorted(glob.glob(os.path.join(good_path, "*")))
        bad_samples = sorted(glob.glob(os.path.join(bad_path, "*")))

        good_per_fold = math.ceil(len(good_samples) / 4)
        bad_per_fold = math.ceil(len(bad_samples) / 4)
        random.seed(1337)
        random.shuffle(good_samples)
        random.shuffle(bad_samples)

        selected_good_test_fold = good_samples[self.fold * good_per_fold:(self.fold + 1) * good_per_fold]
        selected_bad_test_fold = bad_samples[self.fold * bad_per_fold:(self.fold + 1) * bad_per_fold]

        selected_good = selected_good_test_fold if self.kind == "test" else list(set(good_samples) - set(selected_good_test_fold))
        selected_bad = selected_bad_test_fold if self.kind == "test" else list(set(bad_samples) - set(selected_bad_test_fold))

        samples = []
        for image_folder in selected_good:
            for image_path_short in os.listdir(image_folder):
                image_path = os.path.join(image_folder, image_path_short)
                image_name = image_path.split("/")[-1][:-4]
                img = cv2.imread(image_path)
                img = cv2.resize(img, self.resize)
                sample = {"image": img, "label": 0, "name": image_name, "image_path": image_path}
                samples.append(sample)

        if self.kind == "test" or self.c.METHOD == "classification":
            for image_folder in selected_bad:
                for image_path_short in os.listdir(image_folder):
                    image_path = os.path.join(image_folder, image_path_short)
                    image_name = image_path.split("/")[-1][:-4]
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, self.resize)
                    sample = {"image": img, "label": 1, "name": image_name, "image_path": image_path}
                    samples.append(sample)

        self.samples = samples