import visualizer
import warnings
import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split


def load_mnist(path):
    """
    Loads MNIST set using csv files
    """
    with open(path) as file:
        csvfile = file.readlines()
    images = []
    labels = []
    for line in csvfile[1:]:
        line = line.strip()
        pixel_values = [int(x) for x in line.split(",")[1:]]
        images.append(np.array(pixel_values, dtype=np.ubyte))
        labels.append(line.split(",")[0])  # first is the label

    images = np.array(images)

    return images, labels


def load_house(path, load_size):
    """
    Load toy image other than mnist dataset
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, load_size)
    img = img.flatten()

    return img


def load_lfw_faces(img_size=(40, 40)):
    """
    Loads LFW dataset with sklearn
    """
    faces_dataset = fetch_lfw_people(min_faces_per_person=40, resize=0.5)

    images, labels = [], []
    for idx, face_img in enumerate(faces_dataset.images):
        img = cv2.resize(face_img, img_size).flatten()
        images.append(img)
        labels.append(str(faces_dataset.target[idx]))

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=5)

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    return x_train, y_train, x_test, y_test


def apply_pca(images):
    """
    PCA algortihm implemented from scratch
    """

    warnings.filterwarnings("ignore")  # this is to avoid complex numbers warning

    mean_img = np.mean(images, axis=0, dtype=np.float64)
    centered = images - mean_img
    covariance_mat = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_mat)
    eigenvalues = np.array(eigenvalues, dtype=np.float64)
    eigenvectors = np.array(eigenvectors, dtype=np.float64)  # float is easier for image reconstruction

    # numpy sorted eigenvalues is not guaranteed
    sort_idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    return eigenvalues, eigenvectors, mean_img


def project(eigenvectors, images, dim_size=2):
    """
    Project data to given dimension by using eigenvectors
    """

    if dim_size > 0:
        eigenvectors = eigenvectors[:, :dim_size]
    mean_images = np.mean(images, axis=0)
    centered = images - mean_images
    projected = eigenvectors.T.dot(centered.T).T

    return projected, mean_images


def reconstruct(eigenvectors, projected_data, mean_images, dim_size=2):
    """
    Reconstruct data to given dimension by using eigenvectors
    """

    if dim_size > 0:
        eigenvectors = eigenvectors[:, :dim_size]
    recons = eigenvectors.dot(projected_data.T).T + mean_images

    return recons


def project_and_reconstruct(eigenvectors, images, dim_sizes):
    """
    Project then reconstruct data to given dimension by using eigenvectors
    """

    reconst_results = []
    for dim_size in dim_sizes:
        projected, mean_img = project(eigenvectors, images, dim_size=dim_size)
        reconstructed = reconstruct(eigenvectors, projected, mean_img, dim_size=dim_size)
        reconst_results.append(reconstructed)

    if len(eigenvectors)-1 not in dim_sizes:
        projected, mean_img = project(eigenvectors, images, dim_size=len(eigenvectors)-1)
        reconstructed = reconstruct(eigenvectors, projected, mean_img, dim_size=len(eigenvectors)-1)
        reconst_results.append(reconstructed)

    reconst_results = np.array(reconst_results)
    return reconst_results


def find_elbow_point(eigenvalues, threshold=1e2):
    """
    Finds the elbow point with given threshold
    """
    eigenvalues = eigenvalues.tolist()
    prev_val = eigenvalues[0]
    idx = 0
    for idx in range(1, len(eigenvalues)):
        val = eigenvalues[idx]
        if prev_val - val < threshold:
            break
        prev_val = val

    percentage = compute_variance_percentage(eigenvalues, idx)

    return idx, percentage


def compute_variance_percentage(eigenvalues, idx):
    """
    Computes the explained variance ratio for given number of eigenvectors
    """
    sum_all = np.sum(eigenvalues)
    sum_elbow = np.sum(eigenvalues[:idx+1])
    percentage = sum_elbow / sum_all
    return percentage


if __name__ == '__main__':

    # EXAMPLE PARAMETERS - For MNIST dataset
    dataset = "MNIST"
    train_path = "data/mnist_train.csv"
    test_path = "data/mnist_test.csv"
    additional_image_path = "data/housegray.jpeg"
    projected_dim_size = 2
    pca_tsne_subset_ratio = 0.1
    variance_idx_test1 = 82
    variance_idx_test2 = 492

    """
    # EXAMPLE PARAMETERS - For Faces dataset
    dataset = "Faces"
    train_path = ""
    test_path = ""
    additional_image_path = "data/housegray.jpeg"
    projected_dim_size = 2
    pca_tsne_subset_ratio = 1.0
    variance_idx_test1 = 137
    variance_idx_test2 = 482
    """

    # ALGORITHM
    if dataset == "Faces":
        additional_image_resize = (40, 40)
        train_images, train_labels, test_images, test_labels = load_lfw_faces(additional_image_resize)
        reconst_dim_range = range(2, 40*40, 15)
    else:
        additional_image_resize = (28, 28)
        train_images, train_labels = load_mnist(train_path)
        test_images, test_labels = load_mnist(test_path)
        reconst_dim_range = range(2, 28*28, 10)
    print("Images loaded!")

    eigenvalues_, eigenvectors_, mean_train = apply_pca(train_images)
    print("Computed eigenvectors!")
    elbow_idx, variance_percentage = find_elbow_point(eigenvalues_)
    print("Elbow point: {} Percentage: {:.2f}".format(elbow_idx, variance_percentage))

    # uses all eigenvectors for <=0
    projected_test, mean_test = project(eigenvectors_, test_images, dim_size=projected_dim_size)
    print("Test images projected to 2D!")
    reconstructed_images_mnist = project_and_reconstruct(eigenvectors_, test_images[0],
                                                         dim_sizes=reconst_dim_range)

    house_img = load_house(additional_image_path, additional_image_resize)
    reconstructed_images_house = project_and_reconstruct(eigenvectors_, house_img,
                                                         dim_sizes=reconst_dim_range)
    print("Test images reconstructed!")

    # Visualize all at once
    visualizer.show_example_images(train_images, train_labels,
                                   "Example Images From Dataset", show_count=10)

    visualizer.visualize_single(mean_train, "Mean Image", reshape_to_2d=True)

    visualizer.show_multiple(eigenvectors_[:, :100].T, "Largest Eigenvectors")

    visualizer.scree_plot(eigenvalues_[:50], "Largest Eigenvalues")

    visualizer.visualize_points(projected_test, test_labels,
                                "2D PCA Visualization of the " + dataset + " Test Set",
                                subset_ratio=pca_tsne_subset_ratio)

    visualizer.visualize_tsne(test_images, test_labels,
                              "2D t-SNE Visualization of the " + dataset + " Test Set",
                              subset_ratio=pca_tsne_subset_ratio)

    visualizer.show_multiple(reconstructed_images_mnist, "Reconstruction Example (" + dataset + ")",
                             input_image=test_images[0])

    exp_variance_test1 = compute_variance_percentage(eigenvalues_, idx=variance_idx_test1)
    print("Reconstruction dim: {} Percentage: {:.3f}".format(variance_idx_test1, exp_variance_test1))

    visualizer.show_multiple(reconstructed_images_house, "Reconstruction Example (House)",
                             input_image=house_img)

    exp_variance_test2 = compute_variance_percentage(eigenvalues_, idx=variance_idx_test2)
    print("Reconstruction dim: {} Percentage: {:.3f}".format(variance_idx_test2, exp_variance_test2))

    visualizer.show()
