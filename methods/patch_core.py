import os

import cv2
import torch
from torch.nn import functional as F
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import pairwise_distances
import numpy as np

import abc
from utils import plot_sample

from methods.unsupervised_method import UnsupervisedMethod
from utils import pickle_dump
from config import Config
from sklearn.neighbors import KNeighborsClassifier


class SamplingMethod(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, X, y, seed, **kwargs):
        self.X = X
        self.y = y
        self.seed = seed

    def flatten_X(self):
        shape = self.X.shape
        flat_X = self.X
        if len(shape) > 2:
            flat_X = np.reshape(self.X, (shape[0], np.product(shape[1:])))
        return flat_X

    @abc.abstractmethod
    def select_batch_(self):
        return

    def select_batch(self, **kwargs):
        return self.select_batch_(**kwargs)

    def to_dict(self):
        return None


class kCenterGreedy(SamplingMethod):

    def __init__(self, X, y, seed, metric='euclidean'):
        self.X = X
        self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.

        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, model, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.

        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size

        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
            # Assumes that the transform function takes in original data and not
            # flattened data.
            print('Getting transformed features...')
            self.features = model.transform(self.X)
            print('Calculating distances...')
            self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
            print('Using flat_X as features.')
            self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_obs))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
              % max(self.min_distances))

        self.already_selected = already_selected

        return new_batch


def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist


class NN():

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]


class KNN(NN):

    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):
        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)

        knn = dist.topk(self.k, largest=False)

        return knn


class PatchCore(UnsupervisedMethod):

    def train_and_eval(self, curr_run_path, cfg: Config, train_dataloader, test_dataloader, extra_args):
        device = cfg.DEVICE
        model = torch.hub.load('pytorch/vision:v0.8.2', 'wide_resnet50_2', pretrained=True)
        model.to(device)
        for param in model.parameters():
            param.requires_grad = False

        # set model's intermediate outputs
        outputs = [[]]

        def hook(module, input, output):
            outputs[0].append(output)

        model.layer2[-1].register_forward_hook(hook)
        model.layer3[-1].register_forward_hook(hook)

        embedding_coreset = extract_embedding_coreset(model, outputs, train_dataloader, device, cfg.PATCHCORE_CORESET_SAMPLING_RATIO)

        pickle_dump(os.path.join(curr_run_path, f"embedding_coreset.pickle"), embedding_coreset)

        test_scores, test_gts, test_names = get_test_scores(model, outputs, embedding_coreset, test_dataloader, device, cfg.SAVE_SEGMENTATION, os.path.join(curr_run_path, "segmentation"))
        train_scores, train_gts, train_names = get_test_scores(model, outputs, embedding_coreset, train_dataloader, device, False, None)

        return_dict = {
            "tr_gts": np.array(train_gts), "tr_scores": np.array(train_scores), "tr_names": np.array(train_names),
            "ts_gts": np.array(test_gts), "ts_scores": np.array(test_scores), "ts_names": np.array(test_names),
        }
        return return_dict


def get_test_scores(model, outputs, embedding_coreset, dataloader, device, save_segmentation, segmentation_save_dir):
    model.eval()
    scores, ys, names = [], [], []
    embedding_coreset_tensor = torch.from_numpy(embedding_coreset)

    for sample_dict in dataloader:
        x, y, name = sample_dict["image"], sample_dict["label"], sample_dict["name"]
        ys.extend(y.cpu().detach().numpy())
        names.extend(name)

        outputs[0] = []
        embeddings = []

        # model prediction
        with torch.no_grad():
            _ = model(x.to(device))
        for feature in outputs[0]:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))

        knn = KNN(embedding_coreset_tensor.to(device), k=9)
        score_patches = knn(torch.from_numpy(embedding_test).to(device))[0].cpu().detach().numpy()

        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
        score = w * max(score_patches[:, 0])  # Image-level score
        scores.append(score)

        if save_segmentation:
            seg = score_patches.max(axis=1).reshape((embedding_.shape[2], -1))
            # seg = score_patches[:, 0].reshape((embedding_.shape[2], -1))
            seg = cv2.resize(seg, (x.shape[3], x.shape[2]))
            plot_sample(x[0].detach().cpu().numpy().transpose((1, 2, 0)), None, seg, y.item(), name[0], score, segmentation_save_dir, True)

    return scores, ys, names


def extract_embedding_coreset(model, outputs, dataloader, device, coreset_sampling_ratio):
    model.eval()
    embedding_list = []

    for sample_dict in dataloader:
        x = sample_dict["image"]

        outputs[0] = []
        embeddings = []

        # model prediction
        with torch.no_grad():
            _ = model(x.to(device))
        for feature in outputs[0]:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        embedding_list.extend(reshape_embedding(np.array(embedding)))

    total_embeddings = np.array(embedding_list)
    # Random projection
    randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)  # 'auto' => Johnson-Lindenstrauss lemma
    randomprojector.fit(total_embeddings)
    # Coreset Subsampling
    selector = kCenterGreedy(total_embeddings, 0, 0)
    selected_idx = selector.select_batch(model=randomprojector, already_selected=[], N=int(total_embeddings.shape[0] * coreset_sampling_ratio))
    embedding_coreset = total_embeddings[selected_idx]

    return embedding_coreset


def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list


def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
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
