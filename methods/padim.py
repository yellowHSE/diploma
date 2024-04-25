import random
from random import sample
import numpy as np
import os
import pickle

import utils
from methods.unsupervised_method import UnsupervisedMethod
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, resnet18
from config import Config
from utils import plot_sample

random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed_all(1024)


def save_pickle(fn, data):
    with open(fn, "wb") as f:
        pickle.dump(data, f)


class Padim(UnsupervisedMethod):

    def train_and_eval(self, curr_run_path, cfg: Config, train_dataloader, test_dataloader, extra_args):
        device = cfg.DEVICE
        model, t_d, d = load_model(cfg.PADIM_ARCH, device)

        idx = torch.tensor(sample(range(0, t_d), d))

        # set model's intermediate outputs
        outputs = [[]]

        def hook(module, input, output):
            outputs[0].append(output)

        model.layer1[-1].register_forward_hook(hook)
        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

        train_cov, train_mean = get_train_mean_cov(idx, model, outputs, train_dataloader, device)

        save_pickle(os.path.join(curr_run_path, "train_cov_mean.pkl"), (train_mean, train_cov))

        test_scores, test_gts, test_names = get_test_scores(idx, model, outputs, test_dataloader, train_cov, train_mean, device, cfg.SAVE_SEGMENTATION, os.path.join(curr_run_path, "segmentation"))
        train_scores, train_gts, train_names = get_test_scores(idx, model, outputs, train_dataloader, train_cov, train_mean, device, False, None)

        return_dict = {
            "tr_gts": np.array(train_gts), "tr_scores": train_scores, "tr_names": np.array(train_names),
            "ts_gts": np.array(test_gts), "ts_scores": test_scores, "ts_names": np.array(test_names),
        }
        return return_dict


def load_model(arch, device):
    # load model
    if arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        t_d = 1792
        d = 550

    model.to(device)
    model.eval()
    return model, t_d, d


def trim_training_percentage(samples, scores, percentage):
    samples, scores = np.array(samples), np.array(scores)
    sort_ixs = np.argsort(scores)
    samples, scores = samples[sort_ixs], scores[sort_ixs]
    n_dropped = int(len(samples) * percentage)
    return samples[:-n_dropped]


def get_test_auc_score(img_scores, test_ys):
    test_ys = np.asarray(test_ys)
    fpr, tpr, _ = roc_curve(test_ys, img_scores)
    if len(np.unique(test_ys)) == 1:
        return -1
    img_roc_auc = roc_auc_score(test_ys, img_scores)
    return img_roc_auc


def get_train_mean_cov(idx, model, outputs, train_dataloader, device):
    train_features_dict, _ = extract_features(model, train_dataloader, OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])]), outputs, device)
    train_embeddings, train_embedding_dims = get_embedding_vectors(train_features_dict, idx)
    train_mean, train_cov = get_mean_cov(train_embeddings)
    return train_cov, train_mean


def get_test_scores(idx, model, outputs, test_dataloader, train_cov, train_mean, device, save_segmentation, segmentation_save_dir):
    test_features_dict = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_features_dict, (test_images, test_ys, test_names, test_masks) = extract_features(model, test_dataloader, test_features_dict, outputs, device)
    test_embeddings, test_embeddings_dims = get_embedding_vectors(test_features_dict, idx)
    B, C, H, W = test_embeddings_dims
    embedding_vectors = test_embeddings.numpy()
    dist_list = []
    for i in range(H * W):
        mean = train_mean[:, i]
        conv_inv = np.linalg.inv(train_cov[:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)
    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
    # upsample
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=(test_images[0].shape[1], test_images[0].shape[2]), mode='bilinear', align_corners=False).squeeze().numpy()
    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    if save_segmentation:
        for img, seg, name, im_score, gt_label, mask in zip(test_images, scores, test_names, img_scores, test_ys, test_masks):
            plot_sample(img.transpose((1, 2, 0)), mask[0], seg,gt_label,  name, im_score, segmentation_save_dir, True)

    return img_scores, test_ys, test_names


def extract_features(model, dataloader, features_dict, outputs, device):
    ys = []
    images = []
    names = []
    masks = []

    for sample_dict in tqdm(dataloader, '| feature extraction | '):
        x, y, name, mask = sample_dict["image"], sample_dict["label"], sample_dict["name"], sample_dict["mask"]
        images.extend(x.cpu().detach().numpy())
        ys.extend(y.cpu().detach().numpy())
        masks.extend(mask.cpu().detach().numpy())
        names.extend(name)

        outputs[0] = []
        # model prediction
        with torch.no_grad():
            _ = model(x.to(device))
        # get intermediate layer outputs
        for k, v in zip(features_dict.keys(), outputs[0]):
            features_dict[k].append(v.cpu().detach())
        # initialize hook outputs
    for k, v in features_dict.items():
        features_dict[k] = torch.cat(v, 0)

    return features_dict, (images, ys, names, masks)


def get_embedding_vectors(features_dict, idx):
    # Embedding concat
    embedding_vectors = features_dict['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, features_dict[layer_name])
    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    return embedding_vectors, (B, C, H, W)


def get_mean_cov(embedding_vectors):
    _, C, HW = embedding_vectors.size()
    mean = torch.mean(embedding_vectors, dim=0).numpy()
    cov = torch.zeros(C, C, HW).numpy()
    I = np.identity(C)
    for i in range(HW):
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
    # save learned distribution
    return mean, cov


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z
