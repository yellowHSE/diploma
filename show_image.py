import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np


def show_image(image, is_bgr=True, save_to=None, return_only=False, log=False):
    """
    torch.Tensor -> C x H x W
    cv2 image:   -> H x W x C
    """
    if isinstance(image, torch.Tensor):
        if log:
            print("Converting torch.Tensor to numpy")
        image = image.detach().cpu().numpy()

    if len(image.shape) == 4:
        image = image[0, :, :, :]

    if len(image.shape) == 3:
        if image.shape[0] <= 3:
            if log:
                print("Converting image to channels last")
            image = np.transpose(image, (1, 2, 0))

        if image.shape[2] == 1:
            if log:
                print("Removing dimension from HxWx1 image")
            image = image[:, :, 0]

    if image.dtype == np.uint8:
        if log:
            print("Scaling uint8 image")
        image = (image / np.max(image) * 255).astype(np.uint8)
    elif image.dtype == np.bool:
        if log:
            print("Converting bool image")
        image = image * 255

    if save_to is not None:
        if log:
            print(f"Saving image to ./{save_to}.png")
        cv2.imwrite(f"{save_to}.png", image)

    if is_bgr and len(image.shape) == 3:
        if log:
            print("Converting to RGB for displaying")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.dtype == np.float and np.min(image) < 0 or np.max(image) > 1:
        if log:
            print("Scalling image to [0, 1]")
        image = (image - image.min()) / (image.max() - image.min())

    if return_only:
        return image

    plt.imshow(image)
    plt.show(bbox_inches="tight")

    if log:
        print("Finished")
    return image


# %%
def plot_grid(images, rows, columns, res):
    if not isinstance(images, list):
        images = list(images)

    plt.figure(figsize=(rows , columns ))
    for i in range(min(rows * columns, len(images))):
        plt.subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        img = show_image(images[i], return_only=True)
        plt.imshow(img)

    plt.show(bbox_inches="tight", dpi=res)


# %%

if __name__ == '__main__':
    from config import Config
    from datasets_python.debug_mvtec_draem import MVTecDRAEMTrainDataset
    from datasets_python.ds_mvtec_orig import MVTecDataset
    from torch.utils.data import DataLoader

    # %%
    cfg = Config()
    cfg.METHOD = "draem"

    ds_new = MVTecDataset("train", cfg, {"object_class": "capsule"})
    ds_old = MVTecDRAEMTrainDataset("train", cfg, {"object_class": "capsule"})

    dl_new = DataLoader(ds_new, 1, False)
    dl_old = DataLoader(ds_old, 1, False)

    batches_new = list(dl_new)
    batches_old = list(dl_old)

    images_new = list(map(lambda x: x["image"], batches_new))
    images_old = list(map(lambda x: x["image"], batches_old))

    augs_new = list(map(lambda x: x["augmented_image"], batches_new))
    augs_old = list(map(lambda x: x["augmented_image"], batches_old))

    # plot_grid(images_new, 5, 5, 200)
    # plot_grid(images_old, 5, 5, 200)

    plot_grid(augs_new, 5, 5, 200)
    plot_grid(augs_old, 5, 5, 200)


    plot_grid(list(map(lambda x: x["mask"], batches_new)), 5, 5, 100)
