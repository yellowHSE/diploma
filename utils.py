import numpy as np
from sklearn.metrics import roc_auc_score
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import cv2

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_auc(ground_truth, predictions):
    if len(np.unique(ground_truth)) == 1:
        return 1.0

    # return roc_auc_score(ground_truth, predictions)
    return roc_auc_score((ground_truth > 0) * 1, predictions)


def trim_training_percentage(samples, scores, percentage):
    sort_ixs = np.argsort(scores)
    samples, scores = samples[sort_ixs], scores[sort_ixs]
    n_dropped = int(len(samples) * percentage)
    return samples[:-n_dropped]


def pickle_load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def json_load(p):
    with open(p, "r") as f:
        return pickle.load(f)


def pickle_dump(p, obj):
    with open(p, "wb") as f:
        pickle.dump(obj, f)


def json_dump(p, obj):
    with open(p, "w") as f:
        json.dump(obj, f)


def csv_load(p) -> pd.DataFrame:
    return pd.read_csv(p)


def csv_dump(p, obj, cols):
    pd.DataFrame(obj, columns=cols).to_csv(p, index=False)


def plot_score_only(image, score, name, label,save_dir,  normalize_imagenet=True, normalize_1=False):
    if normalize_imagenet:
        image = (image * imagenet_std) + imagenet_mean
    if normalize_1:
        image = (image / 2) + 0.5

    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.title(f"{name}: {label}")
    plt.xlabel(f"{(score * 100):.2f}" if score > 0.0001 else "<0.0001")
    plt.imshow(image)

    out_prefix = '{:.5f}_'.format(score)

    plt.savefig(f"{save_dir}/{out_prefix}result_{name}.jpg", bbox_inches='tight', dpi=300)
    plt.close()


def plot_sample(image, mask, segmentation, gt_label, image_name, score, save_dir, normalize_imagenet=True, normalize_1=False, convert_channels=True):
    if normalize_imagenet:
        image = (image * imagenet_std) + imagenet_mean
    if normalize_1:
        image = (image / 2) + 0.5

    plt.figure()
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"{image_name}: {gt_label}")
    if image.shape[0] < image.shape[1]:
        image = np.transpose(image, axes=[1, 0, 2])
        segmentation = np.transpose(segmentation)
        if mask is not None:
            mask = np.transpose(mask)
    if image.shape[2] == 1:
        plt.imshow(image, cmap="gray")
    else:
        if convert_channels:
            image = cv2.cvtColor((image*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
        plt.imshow(image)

    if mask is not None:
        plt.subplot(1, 3, 2)
        plt.xticks([])
        plt.yticks([])
        plt.title('Groundtruth')
        plt.imshow(mask, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Output: {score * 100:.2f}")
    # display max
    vmax_value = max(1, np.max(segmentation))
    plt.imshow(segmentation, cmap="jet", vmax=vmax_value)

    out_prefix = '{:.5f}_'.format(score)

    plt.savefig(f"{save_dir}/{out_prefix}result_{image_name}.jpg", bbox_inches='tight', dpi=300)
    plt.close()
