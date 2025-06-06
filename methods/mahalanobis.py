import numpy as np
import os
import pickle
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis

import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from methods.unsupervised_method import UnsupervisedMethod
from utils import plot_score_only


def save_pickle(fn, data):
    with open(fn, "wb") as f:
        pickle.dump(data, f)


class Mahalanobis(UnsupervisedMethod):

    def train_and_eval(self, curr_run_path, cfg, train_dataloader, test_dataloader, extra_args):
        self.c = cfg
        self.device = self.c.DEVICE
        print(f"Using device: {self.device}")

        model = EfficientNetModified.from_pretrained('efficientnet-b4')
        model.to(self.device)
        if next(model.parameters()).is_cuda:
            print(f"Model is on GPU: {self.device}")
        else:
            print("Model is not on GPU")
        model.eval()

        train_outputs = [[] for _ in range(9)]

        # for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
        for sample_dict in train_dataloader:
            x, y = sample_dict["image"], sample_dict["label"]
            print(f"Input tensor original device: {x.device}")
            x = x.to(self.device)
            print(f"Input tensor moved to device: {x.device}")
            # model prediction
            with torch.no_grad():
                feats = model.extract_features(x)
            for f_idx, feat in enumerate(feats):
                train_outputs[f_idx].append(feat)

        # fitting a multivariate gaussian to features extracted from every level of ImageNet pre-trained model
        for t_idx, train_output in enumerate(train_outputs):
            mean = torch.mean(torch.cat(train_output, 0).squeeze(), dim=0).cpu().detach().numpy()
            # covariance estimation by using the Ledoit. Wolf et al. method
            cov = LedoitWolf().fit(torch.cat(train_output, 0).squeeze().cpu().detach().numpy()).covariance_
            train_outputs[t_idx] = [mean, cov]

        save_pickle(os.path.join(curr_run_path, "train_outputs.pkl"), train_outputs)

        test_scores, test_gts, test_names = get_test_scores(test_dataloader, self.device, model, train_outputs, cfg.SAVE_SEGMENTATION, os.path.join(curr_run_path, "segmentation"))
        train_scores, train_gts, train_names = get_test_scores(train_dataloader, self.device, model, train_outputs, False, None)

        return_dict = {
            "tr_gts": np.array(train_gts), "tr_scores": train_scores, "tr_names": np.array(train_names),
            "ts_gts": np.array(test_gts), "ts_scores": test_scores, "ts_names": np.array(test_names),
        }
        return return_dict


def get_test_scores(dataloader, device, model, train_outputs, save_segmentation, segmentation_save_dir):
    gt_list, names = [], []
    test_outputs = [[] for _ in range(9)]

    # extract test set features
    for sample_dict in dataloader:
        x, y, name = sample_dict["image"], sample_dict["label"], sample_dict["name"]
        gt_list.extend(y.cpu().detach().numpy())
        names.extend(name)
        x = x.to(device)
        print(f"Test input tensor moved to device: {x.device}")
        # model prediction
        with torch.no_grad():
            feats = model.extract_features(x)
        for f_idx, feat in enumerate(feats):
            test_outputs[f_idx].append(feat)
    for t_idx, test_output in enumerate(test_outputs):
        test_outputs[t_idx] = torch.cat(test_output, 0).squeeze().cpu().detach().numpy()

    # calculate Mahalanobis distance per each level of EfficientNet
    dist_list = []
    for t_idx, test_output in enumerate(test_outputs):
        mean = train_outputs[t_idx][0]
        cov_inv = np.linalg.inv(train_outputs[t_idx][1])
        dist = [mahalanobis(sample, mean, cov_inv) for sample in test_output]
        dist_list.append(np.array(dist))

    # Anomaly score is followed by unweighted summation of the Mahalanobis distances
    scores = np.sum(np.array(dist_list), axis=0)

    if save_segmentation:
        for img, name, im_score, gt_label in zip(dataloader.dataset, names, scores, gt_list):
            plot_score_only(img["image"].detach().cpu().numpy().transpose((1, 2, 0)), im_score, name, gt_label, segmentation_save_dir, True)

    return scores, gt_list, names


class EfficientNetModified(EfficientNet):

    def extract_features(self, inputs):
        """ Returns list of the feature at each level of the EfficientNet """

        feat_list = []

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        print(f"Feature after stem on device: {x.device}")
        feat_list.append(F.adaptive_avg_pool2d(x, 1))

        # Blocks
        x_prev = x
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            print(f"Feature after block {idx} on device: {x.device}")
            if (x_prev.shape[1] != x.shape[1] and idx != 0) or idx == (len(self._blocks) - 1):
                feat_list.append(F.adaptive_avg_pool2d(x_prev, 1))
            x_prev = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        print(f"Feature after head on device: {x.device}")
        feat_list.append(F.adaptive_avg_pool2d(x, 1))

        return feat_list
