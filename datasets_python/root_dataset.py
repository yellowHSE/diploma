import cv2
from torch.utils.data import Dataset
import os
import numpy as np
import json
import random
from torchvision import transforms as T
from PIL import Image
from config import Config
import pandas as pd
from datasets_python.dataset_utils import augment_cutpaste, augment_draem, augment_draem_orig
from datasets_python.perlin_noise import generate_anomalies_alpha


def read_split(splits_dir, ds_name, kind, iteration_ix, perc_images, perc_pixels, single_test):
    kind = kind.lower()

    if kind == "test":
        split_path = os.path.join(splits_dir, ds_name, f"split_{-1 if single_test != 0 else iteration_ix}_iteration_TEST.csv")
    else:
        perlin_or_bad = "DEFECTIVE" if (perc_pixels == -1 or perc_images == 0) else "PERLIN"
        split_path = os.path.join(splits_dir, ds_name, f"split_{iteration_ix}_iteration_{perc_images}_images_{perlin_or_bad}_TRAIN.csv")

    return pd.read_csv(split_path).values.tolist()


# %%

class RootDataset(Dataset):
    ds_name = None

    resize = None  # WIDHTH x HEIGHT

    transforms_riad = None
    transforms_padim = None
    transforms_cutpaste = None

    def __init__(self, kind: str, c: Config, extra_params) -> None:
        self.resize_torchvision = (self.resize[1], self.resize[0])
        self.kind = kind
        self.c: Config = c
        self.extra_params = extra_params
        self.init_transforms()

        self.images_dir = os.path.join(self.c.DATASETS_PATH, self.ds_name)
        self.read_contents()

    def get_extra_param(self, name):
        if name in self.extra_params:
            return self.extra_params[name]
        else:
            raise Exception(f"Missing {name} entry in extra_params!")

    def read_contents(self):
        samples = []

        data_points = read_split(self.c.SPLITS_DIR, self.ds_name, self.kind, self.c.ITERATION_IX, self.c.PERC_IMAGES, self.c.PERC_PIXELS, self.c.SINGLE_TEST)

        for img_n, label, p_pixels, p_alpha, p_source, p_seed in data_points:
            img = cv2.imread(os.path.join(self.images_dir, img_n))
            img = cv2.resize(img, self.resize)

            """
            Augment perlin images if necessary
            """
            if label == 2:
                src_image = cv2.imread(os.path.join(self.c.PERLIN_SOURCE_IMAGES_PATH, self.kind, p_source))
                img = generate_anomalies_alpha(img, src_image, self.c.PERC_PIXELS if self.kind == "train" else p_pixels, p_alpha, p_seed)

            samples.append({"image": img, "name": img_n, "label": label})

        print(f"Loaded {len(samples)} samples for kind:{self.kind}")
        self.samples = samples

    def transform(self, image, label, mask):
        extra_dict = None
        method = self.c.METHOD.lower()
        if method in ["padim", "mahalanobis", "patchcore", "classification"]:
            # if self.kind.lower() == "train":
            image = self.transforms_padim_test(Image.fromarray(image))
            # else:
            #     image = self.transforms_padim_test(Image.fromarray(image))

        elif method == "riad":
            image = self.transforms_riad(Image.fromarray(image))
        elif method == "cutpaste":
            if self.kind == "train" and not self.get_extra_param("train_ds_no_aug"):
                image, label = augment_cutpaste(image)
            image = self.transforms_cutpaste(Image.fromarray(image))
        elif method == "draem":
            image = cv2.resize(image, dsize=self.resize)
            if self.kind == "train":
                if True:
                    image, augmented_image, label, mask_draem = augment_draem_orig(image, self.resize)
                else:
                    image, augmented_image, label, mask_draem = augment_draem(image, self.resize)
                    augmented_image = self.transforms_draem(Image.fromarray(augmented_image.astype(np.uint8)))
                    mask_draem = self.transforms_draem(Image.fromarray(mask_draem.squeeze().astype(np.uint8)))
                extra_dict = {"mask": mask_draem, "augmented_image": augmented_image}
            else:
                image = self.transforms_draem(Image.fromarray(image))
        elif method == "cflow":
            image = self.transforms_riad(Image.fromarray(image))

        if mask is not None and method != "draem":
            mask = self.transforms_mask_resize(Image.fromarray(mask))

        return image, label, mask, extra_dict

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image, label = sample["image"], sample["label"]
        if "mask" in sample.keys():
            mask = sample["mask"]
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        image, label, mask, extra_dict = self.transform(image, label, mask)

        item_dict = {"name": sample["name"], "image": image, "label": label, "mask": mask}
        if extra_dict is not None:
            for k, v in extra_dict.items():
                item_dict[k] = v
        return item_dict

    def __len__(self) -> int:
        return len(self.samples)

    def init_transforms(self):
        self.transforms_mask_resize = T.Compose([T.ToTensor()])
        self.transforms_riad = T.Compose([T.Resize(self.resize_torchvision), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        # self.transforms_padim_train = T.Compose([T.Resize(self.resize_torchvision), T.CenterCrop(self.cropsize), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transforms_padim_test = T.Compose([T.Resize(self.resize_torchvision), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transforms_cutpaste = T.Compose([T.ColorJitter(0.1, 0.1, 0.1, 0.1), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # self.transforms_draem = T.Compose([T.ToTensor()])
        self.transforms_draem = self.get_draem_aug()

    def get_draem_aug(self):
        if self.kind == "test":
            trsfs = [T.ToTensor()]
            if self.c.DRAEM_AUG_NORM:
                trsfs.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
            return T.Compose(trsfs)

        trsfs = [T.ToTensor()]
        if self.c.DRAEM_AUG_FLIP:
            print(f"Adding flips")
            trsfs.append(T.RandomVerticalFlip())
            trsfs.append(T.RandomHorizontalFlip())
        if self.c.DRAEM_AUG_CJ:
            print(f"Adding CJ")
            cj = self.c.DRAEM_AUG_CJ
            trsfs.append(T.ColorJitter(cj, cj, cj, cj))
        if self.c.DRAEM_AUG_BLUR:
            print(f"Adding BLUR")
            trsfs.append(T.GaussianBlur(self.c.DRAEM_AUG_BLUR_KERNEL, (0.1, self.c.DRAEM_AUG_BLUR_SIGMA)))
        if self.c.DRAEM_AUG_NORM:
            print(f"Adding NORM")
            trsfs.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        return T.Compose(trsfs)

# %%
# if __name__ == '__main__':
#     spl = read_split("ksdd2", "train", 0, 10, 10)
#     for img_n, label, perc_pixels, alpha, source_image, seed in spl:
#         print(source_image)
