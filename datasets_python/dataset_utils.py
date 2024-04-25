import numpy as np
import cv2
import random
import albumentations as A
import glob
import imgaug.augmenters as iaa
import math
import torch
from config import Config


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (width / 2, height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def cut_paste(image):
    h, w = image.shape[:2]
    area_ratio = random.uniform(0.02, 0.15)
    aspect_ratio = random.uniform(0.3, 3.3)

    area = np.prod(h * w * area_ratio)
    w_c = round(np.sqrt(area * aspect_ratio))
    h_c = round(w_c / aspect_ratio)

    if w_c >= w:
        w_c = w - 1
    if h_c >= h:
        h_c = h - 1

    left_cut = random.randint(0, w - w_c)
    top_cut = random.randint(0, h - h_c)

    left_paste = random.randint(0, w - w_c)
    top_paste = random.randint(0, h - h_c)

    image[top_paste:top_paste + h_c, left_paste:left_paste + w_c, :] = image[top_cut:top_cut + h_c, left_cut:left_cut + w_c, :]
    return image


def scar(image):
    h, w = image.shape[:2]
    w_c = random.randint(2, 16)
    h_c = random.randint(10, 25)

    left_cut = random.randint(0, w - w_c)
    top_cut = random.randint(0, h - h_c)

    scar_patch = image[top_cut:top_cut + h_c, left_cut:left_cut + w_c, :]
    transforms_array = [A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, always_apply=True),
                        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=0, always_apply=True)]
    random.shuffle(transforms_array)
    scar_patch = A.Compose(transforms_array)(image=scar_patch)["image"]

    rotated_scar_patch = rotate_image(scar_patch, random.randint(-45, 45))
    patch_mask = np.sum(rotated_scar_patch, axis=2) != 0

    h_rc, w_rc = rotated_scar_patch.shape[:2]
    left_paste = random.randint(0, w - w_rc)
    top_paste = random.randint(0, h - h_rc)

    cropped_orig_image = image[top_paste:top_paste + h_rc, left_paste:left_paste + w_rc, :]

    cropped_orig_image[patch_mask] = rotated_scar_patch[patch_mask]
    image[top_paste:top_paste + h_rc, left_paste:left_paste + w_rc, :] = cropped_orig_image

    return image


def augment_cutpaste(image):
    selected_augmentation = random.randint(0, 2)
    if selected_augmentation == 1:
        image = cut_paste(image)
        label = 1
    elif selected_augmentation == 2:
        image = scar(image)
        label = 2
    else:
        label = 0

    return image, label


#
#  DRAEM
#
def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)
    dot = lambda grad, shift: (
            np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                     axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


anomaly_source_paths = sorted(glob.glob(Config.DRAEM_ANOMALY_SOURCE_PATH))
random_augmenter = iaa.SomeOf(3, [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                                  iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                                  iaa.pillike.EnhanceSharpness(),
                                  iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                                  iaa.Solarize(0.5, threshold=(32, 128)),
                                  iaa.Posterize(),
                                  iaa.Invert(),
                                  iaa.pillike.Autocontrast(),
                                  iaa.pillike.Equalize(),
                                  iaa.Affine(rotate=(-45, 45))
                                  ])
rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


def _dream_augment_image(image, resize_shape):
    anomaly_source_path = random.sample(anomaly_source_paths, 1)[0]
    aug = random_augmenter

    perlin_scale = 6
    min_perlin_scale = 0
    anomaly_source_img = cv2.imread(anomaly_source_path)
    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=resize_shape)

    anomaly_img_augmented = aug(image=anomaly_source_img)
    perlin_scalex = 2 ** random.randint(min_perlin_scale, perlin_scale - 1)
    perlin_scaley = 2 ** random.randint(min_perlin_scale, perlin_scale - 1)

    perlin_noise = rand_perlin_2d_np((resize_shape[1], resize_shape[0]), (perlin_scalex, perlin_scaley))
    perlin_noise = rot(image=perlin_noise)
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = np.expand_dims(perlin_thr, axis=2)

    # img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
    img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr

    beta = random.random() * 0.8

    augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
        perlin_thr)

    no_anomaly = random.random()
    # no_anomaly = 0

    if no_anomaly > 0.5:
        image = image.astype(np.float32)
        return image, np.zeros_like(perlin_thr, dtype=np.float32), 0
    else:
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1 - msk) * image
        has_anomaly = 1
        if np.sum(msk) == 0:
            has_anomaly = 0
        return augmented_image, msk, has_anomaly


def augment_draem(image, resize_shape):
    do_aug_orig = random.random() > 0.7
    if do_aug_orig:
        image = rot(image=image)
    # image = image.astype(np.float32) / 255.0
    augmented_image, anomaly_mask, label = _dream_augment_image(image, resize_shape)

    return image, augmented_image, label, anomaly_mask

###
###
###
###

draem_augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]

draem_rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


def randAugmenter():
    aug_ind = np.random.choice(np.arange(len(draem_augmenters)), 3, replace=False)
    aug = iaa.Sequential([draem_augmenters[aug_ind[0]],
                          draem_augmenters[aug_ind[1]],
                          draem_augmenters[aug_ind[2]]]
                         )
    return aug

def _dream_augment_orig(image,resize_shape, anomaly_source_path):
    aug = randAugmenter()
    perlin_scale = 6
    min_perlin_scale = 0
    anomaly_source_img = cv2.imread(anomaly_source_path)
    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(resize_shape[1], resize_shape[0]))

    anomaly_img_augmented = aug(image=anomaly_source_img)
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    perlin_noise = rand_perlin_2d_np((resize_shape[0], resize_shape[1]), (perlin_scalex, perlin_scaley))
    perlin_noise = draem_rot(image=perlin_noise)
    threshold = 0.5
    perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = np.expand_dims(perlin_thr, axis=2)

    img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

    beta = torch.rand(1).numpy()[0] * 0.8

    augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
        perlin_thr)

    no_anomaly = torch.rand(1).numpy()[0]
    # no_anomaly = 0
    if no_anomaly > 0.5:
        image = image.astype(np.float32)
        return image, np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
    else:
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1 - msk) * image
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly = 0.0
        return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

def augment_draem_orig(image, resize_shape):
    image = cv2.resize(image, dsize=(resize_shape[1], resize_shape[0]))
    do_aug_orig = torch.rand(1).numpy()[0] > 0.7
    if do_aug_orig:
        image = draem_rot(image=image)

    image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
    anomaly_source_path = random.sample(anomaly_source_paths, 1)[0]
    augmented_image, anomaly_mask, has_anomaly = _dream_augment_orig(image, resize_shape, anomaly_source_path)
    augmented_image = np.transpose(augmented_image, (2, 0, 1))
    image = np.transpose(image, (2, 0, 1))
    anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))

    return image, augmented_image, has_anomaly, anomaly_mask