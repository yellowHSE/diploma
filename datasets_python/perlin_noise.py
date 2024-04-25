import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def tile_grads(slice1, slice2, gradients, d):
    return np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)


def dot(grad, shift, shape, grid):
    grad1 = grad
    g1 = np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]), axis=-1)
    g2 = grad1[:shape[0], :shape[1]]
    return (g1 * g2).sum(axis=-1)


def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1], gradients, d), [0, 0], shape, grid)
    n10 = dot(tile_grads([1, None], [0, -1], gradients, d), [-1, 0], shape, grid)
    n01 = dot(tile_grads([0, -1], [1, None], gradients, d), [0, -1], shape, grid)
    n11 = dot(tile_grads([1, None], [1, None], gradients, d), [-1, -1], shape, grid)

    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def generate_anomalies_alpha(image, source_image, perc_defective_pixels, alpha, seed=1337):
    np.random.seed(seed)

    if image.shape[0] % 32 != 0 or image.shape[1] % 32 != 0:
        raise Exception("Shape must be divisible by 32")

    source_image = cv2.resize(source_image, (image.shape[1], image.shape[0]))

    # scale struktur v perlin noisu glede na x in y os
    perlin_scalex = 2 ** np.random.randint(0, 6)
    perlin_scaley = 2 ** np.random.randint(0, 6)

    perlin_noise = rand_perlin_2d_np((image.shape[0], image.shape[1]), (perlin_scalex, perlin_scaley))

    threshold = np.sort(np.reshape(perlin_noise, -1))[int((perc_defective_pixels) * perlin_noise.size / 100)]

    perlin_thr = np.where(perlin_noise < threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr = np.expand_dims(perlin_thr, axis=2)

    augmented_image = image * (1 - perlin_thr) + (source_image * (perlin_thr) * alpha + image * (perlin_thr) * (1 - alpha))

    # return augmented_image, image, source_image, perlin_noise, perlin_thr
    return augmented_image.astype(np.uint8)


# def sh(im):
#     plt.imshow(im)
#     plt.show()
#
#
def show_plt(aug, orig, src, p_tr, p_n):
    plt.subplot(1, 5, 1)
    plt.imshow(aug)
    plt.subplot(1, 5, 2)
    plt.imshow(orig)
    plt.subplot(1, 5, 3)
    plt.imshow(src)
    plt.subplot(1, 5, 4)
    plt.imshow(p_tr)
    plt.subplot(1, 5, 5)
    plt.imshow(p_n)

    plt.show(dpi=500, bbox_inches="tight")


# %%

if __name__ == '__main__':

    import glob

    images = sorted(glob.glob("/storage/datasets/GOSTOP/MVTec/mvtec/capsule/train/good/*.png"))
    src_images = sorted(glob.glob("/home/jakob/projects/RIAD/datasets_data/describable/dtd/images/blotchy/*.jpg"))

    # src_image_path = "/home/jakob/projects/RIAD/datasets_data/describable/dtd/images/blotchy/blotchy_0032.jpg"
    src_image_path = "/storage/datasets/GOSTOP/DAGM/Class1/Train/0576.PNG"
    src_image = cv2.imread(src_image_path)
    image = np.ones_like(src_image, dtype=np.uint8) * 255
    for i in range(1):
        image_path = images[np.random.randint(len(images))]
        # src_image_path = src_images[np.random.randint(len(src_images))]
        augmented_image, original_image, source_image, perlin_noise, perlin_thr = generate_anomalies_alpha(image, src_image, perc_defective_pixels=20, alpha=0.1, seed=3)
        show_plt(augmented_image, original_image, source_image, perlin_noise, perlin_thr)
