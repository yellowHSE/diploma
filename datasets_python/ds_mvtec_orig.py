import cv2
import os
from datasets_python.root_dataset import RootDataset
from config import Config
import numpy as np

mult_f = 1


class MVTecDataset(RootDataset):
    ds_name = "mvtec"
    resize = (256 * mult_f, 256 * mult_f)
    cropsize = 224 * mult_f

    def __init__(self, kind: str, c: Config, extra_params):
        self.extra_params = extra_params
        self.object_class = self.get_extra_param("object_class")

        super().__init__(kind, c, extra_params)

    def read_contents(self):
        data_class_path = os.path.join(self.c.ORIG_MVTEC_DIR, self.object_class)
        samples = []

        kind_path = os.path.join(data_class_path, self.kind)
        for defect_type in sorted(os.listdir(kind_path)):
            defect_path = os.path.join(kind_path, defect_type)
            for img_name in sorted(os.listdir(defect_path)):
                image_path = os.path.join(defect_path, img_name)
                img = cv2.imread(image_path)
                img = cv2.resize(img, self.resize)
                sample_name = f"{defect_type}_{img_name[:-4]}"
                if defect_type == "good":
                    label = 0
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                else:
                    label = 1
                    mask = cv2.imread(os.path.join(data_class_path, "ground_truth", defect_type, f"{img_name[:-4]}_mask.png"), cv2.IMREAD_GRAYSCALE)
                sample = {"image": img, "label": label, "name": sample_name, "image_path": image_path, "mask": mask}
                samples.append(sample)

        self.samples = samples
