import torch
import timm
from methods.unsupervised_method import UnsupervisedMethod
from config import Config
import torch.nn.functional as F
import numpy as np
import os
from utils import plot_score_only


class Classification(UnsupervisedMethod):

    def train_and_eval(self, curr_run_path, cfg: Config, train_dataloader, test_dataloader, extra_args):
        device = cfg.DEVICE
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)

        model.train()
        for i in range(cfg.N_EPOCHS):
            ep_loss = 0
            for sample_batch in train_dataloader:
                images, labels = sample_batch["image"].to(device), sample_batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs.reshape(-1), labels.float())

                ep_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {i + 1}/{cfg.N_EPOCHS}: Loss: {ep_loss}")

        ts_scores, ts_gts, ts_names = self.eval_test(model, device, test_dataloader, cfg.SAVE_SEGMENTATION, os.path.join(curr_run_path, "segmentation"))
        tr_scores, tr_gts, tr_names = self.eval_test(model, device, train_dataloader, False, None)

        return_dict = {
            "tr_gts": np.array(tr_gts), "tr_scores": np.array(tr_scores), "tr_names": np.array(tr_names),
            "ts_gts": np.array(ts_gts), "ts_scores": np.array(ts_scores), "ts_names": np.array(ts_names),
        }

        return return_dict

    def eval_test(self, model, device, loader, save_segmentation, segmentation_save_dir):
        model.eval()
        scores, gts, names = [], [], []

        for sample_batch in loader:
            images, labels, ns = sample_batch["image"].to(device), sample_batch["label"].to(device), sample_batch["name"]
            outputs = model(images)
            batch_scores = F.sigmoid(outputs).reshape(-1).tolist()
            scores.extend(batch_scores)
            gts.extend(labels.tolist())
            names.extend(ns)

            if save_segmentation:
                for img, name, im_score, gt_label in zip(images, ns, batch_scores, labels):
                    plot_score_only(img.detach().cpu().numpy().transpose((1, 2, 0)), im_score, name, gt_label.item(), segmentation_save_dir, True)

        return scores, gts, names
