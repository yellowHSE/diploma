import torch
from torch import nn
import os
import numpy as np
from tqdm import tqdm

from methods.unsupervised_method import UnsupervisedMethod
from utils import pickle_dump

from efficientnet_pytorch import EfficientNet
from config import Config


def cov_mtx(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1 / (D - 1) * X @ X.transpose(-1, -2)


class CutPaste(UnsupervisedMethod):

    def train_and_eval(self, curr_run_path, c: Config, train_dataloader, test_dataloader, extra_args):
        device = c.DEVICE
        model = self.train_pretext(train_dataloader, c.N_EPOCHS, device)
        torch.save(model.state_dict(), os.path.join(curr_run_path, f"model_FINAL.pth"))

        model._dropout = nn.Identity()
        model._fc = nn.Identity()
        model.eval()
        model.to(device)

        mean, cov = self.mc_det(model, extra_args["train_fit_dataloader"], device)

        pickle_dump(os.path.join(curr_run_path, f"mean_cov.pickle"), [mean, cov])

        test_scores, test_gts, test_names = self.get_test_scores(model, mean, cov, test_dataloader, device)
        train_scores, train_gts, train_names = self.get_test_scores(model, mean, cov, extra_args["train_fit_dataloader"], device)

        return_dict = {
            "tr_gts": np.array(train_gts), "tr_scores": np.array(train_scores), "tr_names": np.array(train_names),
            "ts_gts": np.array(test_gts), "ts_scores": np.array(test_scores), "ts_names": np.array(test_names),
        }
        return return_dict

    def train_pretext(self, dataloader, n_epochs, device, epoch_mult_f=1):
        model = self.get_model().to(device)
        criterion = self.get_criterion()

        model.train()

        ### Freeze all layers but last one
        for _, p in model.named_parameters():
            p.requires_grad = False
        model._fc.weight.requires_grad = True
        model._fc.bias.requires_grad = True

        optimizer = torch.optim.SGD(model.parameters(), 0.03, momentum=0.9, weight_decay=3e-5)
        lr_scheduler = self.get_lr_scheduler(optimizer)
        n_ep = int(10 * epoch_mult_f * n_epochs / len(dataloader))
        iteration_counter = 0
        for ix_ep in (pbar := tqdm(range(n_ep))):
            loss_ep = 0
            for batch_samples in dataloader:
                images, labels = batch_samples["image"].to(device), batch_samples["label"].to(device)
                optimizer.zero_grad()
                outs = model(images)

                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()
                loss_ep += loss.item()
                iteration_counter += 1
                if iteration_counter >= 256:
                    lr_scheduler.step()
                    iteration_counter = 0
            pbar.set_description(f"Epoch {ix_ep}/{n_ep}: Loss={loss_ep}")

        ### Unfreeze layers except for batch norm
        for n, p in model.named_parameters():
            if "bn" not in n:
                p.requires_grad = True
        optimizer = torch.optim.SGD(model.parameters(), 0.0001, momentum=0.9, weight_decay=3e-5)

        n_ep = int(64 * epoch_mult_f * n_epochs / len(dataloader))
        for ix_ep in (pbar := tqdm(range(n_ep))):
            loss_ep = 0
            for batch_samples in dataloader:
                images, labels = batch_samples["image"].to(device), batch_samples["label"].to(device)
                optimizer.zero_grad()
                outs = model(images)

                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()
                loss_ep += loss.item()
            pbar.set_description(f"Epoch {ix_ep}/{n_ep}: Loss={loss_ep}")

        return model

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, int(1e6))

    def get_model(self):
        self.feature_dim = 1792
        m = EfficientNet.from_pretrained('efficientnet-b4')
        return m

    def mc_det(self, pretext_model, dataloader, device):
        n_images = len(dataloader.dataset)
        features_tensor = torch.zeros(size=(n_images, self.feature_dim), device=device, dtype=torch.float32, requires_grad=False)
        ix_f = 0
        with torch.no_grad():
            for batch_samples in tqdm(dataloader):
                images = batch_samples["image"].to(device)
                features = pretext_model(images)
                features_tensor[ix_f:ix_f + len(images)] = features.detach()
                ix_f += len(images)

        mean = features_tensor.mean(dim=0)
        cov = cov_mtx(features_tensor.T)

        return mean, cov

    def get_test_scores(self, model, mean, cov, dataloader, device, epsilon_inv=1e-4):
        model.eval()
        n_images = len(dataloader.dataset)
        features_tensor = torch.zeros(size=(n_images, self.feature_dim), device=device, dtype=torch.float32, requires_grad=False)
        gts = torch.zeros(size=(n_images,), device=device, dtype=torch.int8, requires_grad=False)
        names = []
        ix_f = 0
        with torch.no_grad():
            for sample_dict in tqdm(dataloader):
                images, labels, names = sample_dict["image"].to(device), sample_dict["label"], sample_dict["name"]
                features = model(images)
                features_tensor[ix_f:ix_f + len(images)] = features.detach()
                gts[ix_f:ix_f + len(images)] = labels.detach()
                ix_f += len(images)

        l = (features_tensor - mean) @ torch.inverse(cov + (torch.eye(self.feature_dim, device=device) * epsilon_inv))
        scores = (l * (features_tensor - mean)).sum(dim=1)

        return scores.detach().cpu().numpy(), gts.detach().cpu().numpy(), names
