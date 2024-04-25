import cv2
import os
from datasets_python.root_dataset import RootDataset
from config import Config
import numpy as np
import socket

on_vicos = not socket.gethostname().startswith("wn")

ROOT_DIR_TRAIN = "/storage/datasets/DEMO_CUBE/tiles/locene/good" if on_vicos else "/d/hpc/projects/FRI/jbozic/vicos_demo/ploscice/dataset/locene/good"
ROOT_DIR_TEST = "/storage/datasets/DEMO_CUBE/tiles/locene/" if on_vicos else "/d/hpc/projects/FRI/jbozic/vicos_demo/ploscice/dataset/locene"

LIMIT_TILES = 35

class TilesDataset(RootDataset):
    ds_name = "ploscice"
    resize = (480 , 480)

    def __init__(self, kind: str, c: Config, extra_params):
        self.extra_params = extra_params
        self.kind = kind

        super().__init__(kind, c, extra_params)

    def read_contents(self):

        samples = []

        if self.kind == "train":
            all_samples = os.listdir(ROOT_DIR_TRAIN)
            # train_samples = list(filter(lambda x: int(x.split(".")[0]) <= LIMIT_TILES, all_samples))
            train_samples =all_samples

            for img_name in train_samples:
                image_path = os.path.join(ROOT_DIR_TRAIN, img_name)
                img = cv2.imread(image_path)
                img = cv2.resize(img, self.resize)
                sample_name = f"{img_name[:-4]}"
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                label = 0
                sample = {"image": img, "label": label, "name": sample_name, "image_path": image_path, "mask": mask}
                samples.append(sample)
        else:
            kind_path = os.path.join(ROOT_DIR_TEST, self.kind)
            for defect_type in sorted(os.listdir(kind_path)):
                defect_path = os.path.join(kind_path, defect_type)
                for img_name in sorted(os.listdir(defect_path)):
                    image_path = os.path.join(defect_path, img_name)
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, self.resize)
                    sample_name = f"{defect_type}_{img_name[:-4]}"
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    if defect_type == "good":
                        label = 0
                    else:
                        label = 1
                    sample = {"image": img, "label": label, "name": sample_name, "image_path": image_path, "mask": mask}
                    samples.append(sample)

        self.samples = samples
