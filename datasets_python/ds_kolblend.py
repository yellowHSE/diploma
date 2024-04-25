import glob
import math

import cv2
import os
from datasets_python.root_dataset import RootDataset
from config import Config
import random
import numpy as np

DS_PATH = "/storage/private/kolektor/Blender/1_vzglavnik_10000"

real_DS_PATH = "/storage/private/kolektor/MFA2_Top"


N_SAMPLES_TRAIN = 100
N_SAMPLES_TEST = 100



class KolBlendDataset(RootDataset):
    ds_name = "kolblend"
    resize = (416, 160 )
    # resize = (192, 96)

    def __init__(self, kind: str, c: Config, extra_params):
        self.resize = (int(self.resize[0] * c.SIZE_MULT_F), int(self.resize[1] * c.SIZE_MULT_F))
        self.resize_torchvision = (self.resize[1], self.resize[0])
        self.extra_params = extra_params
        super().__init__(kind, c, extra_params)


    def read_contents(self):
        if self.kind.lower() == "test" and not self.c.KOLBLEND_EVAL_SYNT:
            self.samples = self.read_actual()
        else:
            if self.kind == "test":
                DS_PATH = "/storage/private/kolektor/Blender/2_vzglavnik_za_video"
            else:
                DS_PATH = "/storage/private/kolektor/Blender/1_vzglavnik_10000"


            good_path = os.path.join(DS_PATH, "good")
            bad_path = os.path.join(DS_PATH, "bad")

            good_samples = sorted(glob.glob(os.path.join(good_path, "*.png")))
            bad_samples = sorted(glob.glob(os.path.join(bad_path, "*-seg.png")))

            random.seed(1337)
            random.shuffle(good_samples)
            random.shuffle(bad_samples)

            bad_samples = list(map(lambda x: x.replace("-seg", ""), bad_samples))

            selected_good = good_samples[:N_SAMPLES_TRAIN] if self.kind == "train" else good_samples[-N_SAMPLES_TEST:]
            selected_bad = bad_samples[:N_SAMPLES_TRAIN] if self.kind == "train" else bad_samples[-N_SAMPLES_TEST:]

            samples = []
            for image_path in selected_good:
                image_name = image_path.split("/")[-1][:-4]
                img = cv2.imread(image_path)
                # img = cv2.resize(img, self.resize)
                sample = {"image": img, "label": 0, "name": image_name, "image_path": image_path}
                samples.append(sample)

            if self.kind == "test" or self.c.METHOD == "classification":
                for image_path in selected_bad:
                    image_name = image_path.split("/")[-1][:-4]
                    img = cv2.imread(image_path)
                    # img = cv2.resize(img, self.resize)
                    mask_path = os.path.join(DS_PATH, "bad", f"{image_name}-seg.png")
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    # mask = cv2.resize(mask, self.resize)
                    sample = {"image": img, "label": 1, "name": image_name, "image_path": image_path, "mask": mask}
                    samples.append(sample)

            self.samples = samples

    def read_actual(self):
        good_path = os.path.join(real_DS_PATH, "good")
        bad_path = os.path.join(real_DS_PATH, "bad")

        good_samples = sorted(glob.glob(os.path.join(good_path, "*.png")))
        bad_samples = sorted(glob.glob(os.path.join(bad_path, "*.png")))

        samples = []
        for image_path in good_samples:
            image_name = image_path.split("/")[-1][:-4]
            img = cv2.imread(image_path)
            img = cv2.resize(img, self.resize)
            sample = {"image": img, "label": 0, "name": image_name, "image_path": image_path}
            samples.append(sample)

        for image_path in bad_samples:
            image_name = image_path.split("/")[-1][:-4]
            img = cv2.imread(image_path)
            img = cv2.resize(img, self.resize)
            sample = {"image": img, "label": 1, "name": image_name, "image_path": image_path}
            samples.append(sample)

        return samples
