import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def show_example_images(images, labels, title, show_count=10):

    images = images.tolist()
    dim = int(len(images[0]) ** (1 / 2))
    example_dict = dict()
    for idx, label in enumerate(labels):
        if label in example_dict.keys() and len(example_dict[label]) < show_count:
            img = np.array(images[idx]).reshape((dim, dim))
            example_dict[label].append(img)
        if label not in example_dict.keys():
            img = np.array(images[idx]).reshape((dim, dim))
            example_dict[label] = [img]

    sorted_labels = sorted(example_dict.keys(), key=lambda x: x.lower())
    example_rows = []
    for label in sorted_labels:
        example_rows.append(np.concatenate(example_dict[label], axis=1))  # concat columns e.g. same category
    example_img = np.concatenate(example_rows, axis=0)  # concat all rows

    visualize_single(example_img, title, reshape_to_2d=False)


def visualize_points(data, labels, title, subset_ratio=1.0):

    plt.figure()
    np.random.seed(10)
    subset_idx = np.random.randint(0, data.shape[0], size=int(data.shape[0]*subset_ratio))

    unique_labels = sorted([label for label in set(labels)])
    labels_arr = np.array(labels)[subset_idx]
    data = data[subset_idx, :]
    for label in unique_labels:
        lidx = np.where(labels_arr == label)
        plt.scatter(data[lidx, 0], data[lidx, 1], s=1, label=label)
    for idx, label in enumerate(labels_arr.tolist()):
        plt.annotate(label, data[idx, :], fontsize=6)
    plt.legend()
    plt.title(title)


def visualize_tsne(data, labels, title, subset_ratio=1.0):

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000, n_jobs=-1)
    tsne_data = tsne.fit_transform(data)
    visualize_points(tsne_data, labels, title, subset_ratio=subset_ratio)


def scree_plot(points, title):

    plt.figure()
    plt.title(title)
    plt.plot(range(1, len(points)+1), points, "o-", markersize=2, linewidth=1, color="blue")


def __flat_to_2d(images):
    images = images.tolist()
    dim = int(len(images[0]) ** (1 / 2))
    reshaped = []
    for img in images:
        img = np.array(img).reshape((dim, dim))
        img = (255 * (img - np.min(img)) / (np.max(img) - np.min(img))).astype(np.uint8)
        reshaped.append(img)

    return reshaped, dim


def show_multiple(images, title, input_image=None):

    ncols = math.ceil(len(images) ** (1 / 2))
    nrows = math.ceil(len(images) / ncols)

    reshaped, dim = __flat_to_2d(images)

    rows = []
    for nrow in range(nrows):
        concatted_row = np.full((dim, dim * ncols), 255)
        concatted_row_temp = np.concatenate(reshaped[nrow*ncols:(nrow+1)*ncols], axis=1)
        concatted_row[:, :concatted_row_temp.shape[1]] = concatted_row_temp
        rows.append(concatted_row)

    whole = np.concatenate(rows, axis=0)

    if input_image is not None:
        fig, (axs1, axs2) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})
        visualize_single(input_image, title, axs=axs1, reshape_to_2d=True)
        visualize_single(whole, title, axs=axs2, reshape_to_2d=False)
    else:
        visualize_single(whole, title, reshape_to_2d=False)

    return whole


def visualize_single(img, title, axs=None, reshape_to_2d=False):

    if reshape_to_2d:
        dim = int(img.shape[0] ** (1 / 2))
        img = img.reshape((dim, dim))
    if axs is None:
        _, axs = plt.subplots(1, 1)

    plt.title(title)
    axs.set_xticks([])
    axs.set_yticks([])
    axs.imshow(img, cmap="gray", vmin=0, vmax=255)


def show():

    plt.show()
