import pickle
import shutil
from pathlib import Path, PosixPath

import hdbscan
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from kneed import KneeLocator
from models.mobilenetv2_antialias_model import MobileNetV2ModelAntiAlias
from models.mobilenetv2_model import MobileNetV2Model
from models.resnet_antialias_model import ResNetModelAntiAlias
from models.resnet_model import ResNetModel
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture as GMM
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .gradcam import GradCam, GuidedBackpropReLUModel, deprocess_image


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class TranspScale(object):
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be transposed and scaled.
        Returns:
            Numpy array: Transposed (C, H, W) and sacled between 0-255.
        """
        nparray = tensor.numpy().transpose((1, 2, 0))
        nparray = (nparray * 255).astype("uint8")
        return nparray


def load_trained_model(
    trained_model: str = "resnet18_aa",
    input_folder: str = "../training/trained_models_output/",
):
    with open(
        Path(input_folder) / f"school_classifier_labels_{trained_model}.pkl", "rb"
    ) as f:
        labels = pickle.load(f)
    number_classes = len(labels)
    pretrained = True
    fs = 4
    if trained_model == "resnet18_aa":
        model = ResNetModelAntiAlias(
            n_classes=number_classes, pretrained=pretrained, filter_size=fs
        )
    elif trained_model == "mobilenetv2_aa":
        model = MobileNetV2ModelAntiAlias(
            n_classes=number_classes, pretrained=pretrained, filter_size=fs
        )
    elif trained_model == "resnet18":
        model = ResNetModel(n_classes=number_classes, pretrained=pretrained)
    else:
        model = MobileNetV2Model(n_classes=number_classes, pretrained=pretrained)
    model.load_state_dict(
        torch.load(Path(input_folder) / f"school_classifier_model_{trained_model}.pt")
    )
    return model


def create_data_folder_structure(base_path: Path = Path("../datasets")):
    for folder in ["train", "test", "validation"]:
        try:
            (base_path / folder).mkdir(parents=True, exist_ok=False)
            print(f"Created folder: {(base_path / folder)}")
        except FileExistsError:
            print(
                "Data already splitted \nUse the function `delete_data_folder_structure` if you want to create a new data split"
            )
            break


def delete_data_folder_structure(base_path: Path = Path("../datasets")):
    for folder in ["train", "test", "validation"]:
        if (base_path / folder).is_dir():
            shutil.rmtree(base_path / folder)
        print(f"{folder} deleted")


def copy_images(
    image_file: PosixPath,
    target_folder: Path,
):
    assert image_file.is_file()
    shutil.copy(image_file, target_folder)


def summarise_datasets(base_path: Path = Path("../datasets")):
    for d in ["train", "validation", "test"]:
        p = (base_path / d).glob("*/*")
        q = (base_path / d).glob("*")
        files = [x for x in p if x.is_file]
        dirs = [x for x in q if x.is_dir]
        print(f"\nNumber of images in {d}: {len(files)}")
        for d in dirs:
            images = [x for x in d.glob("*") if x.is_file]
            print(f"{d.name} {len(images)}")


def split_folder_of_images(
    base_path: Path = Path("../datasets"),
    train_prop: float = 2 / 3,
    valid_prop: float = 0.2,
    seed: int = 1,
):
    print("Splitting images in /train, /test and /validation folders")
    np.random.seed(seed)
    create_data_folder_structure(base_path)
    p = (base_path / "all").glob("*")
    dirs = [x for x in p if x.is_dir()]

    for d in dirs:
        images = [x for x in d.glob("*") if x.is_file()]
        # Get train / test
        train_images = np.random.choice(
            images, size=round(len(images) * train_prop), replace=False
        ).tolist()
        test_images = list(set(images) - set(train_images))
        # Get train / validation
        valid_images = np.random.choice(
            train_images, size=round(len(train_images) * valid_prop), replace=False
        ).tolist()
        train_images = list(set(train_images) - set(valid_images))
        # Create subfolders by class
        for folder in ["train", "test", "validation"]:
            dest_folder = base_path / folder / d.name
            dest_folder.mkdir(parents=True, exist_ok=True)
        # Copy images to subfolders
        [
            copy_images(train_img, base_path / "train" / d.name)
            for train_img in train_images
        ]
        [copy_images(test_img, base_path / "test" / d.name) for test_img in test_images]
        [
            copy_images(valid_img, base_path / "validation" / d.name)
            for valid_img in valid_images
        ]

    # Print summary
    summarise_datasets(base_path)


def show_image(image, title, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array(means)
    std = np.array(stds)
    image = ((std * image + mean) * 255).astype("uint8")
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.title(title)


def parse_class_report(class_report: dict, key: str):
    s = f"{key} | "
    for key, value in class_report.get(key).items():
        s += f"{key}: {round(value, 3)} | "
    s = s[:-3]
    return s


def images_to_probs(model, images, device: torch.device):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    images = images.to(device)
    with torch.no_grad():
        output = model(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = preds_tensor.detach().cpu().numpy()
    probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
    return preds, probs


def get_gradcam_guidedbackprop(
    trained_model: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    means=[0.485, 0.456, 0.406],
    stds=[0.229, 0.224, 0.225],
):
    unorm = UnNormalize(means, stds)
    transpscale = TranspScale()
    imgs, cams, cam_gbs = [], [], []
    model = load_trained_model(trained_model).to(device).eval()
    preds, probs = images_to_probs(model, images, device)
    labels = labels.tolist()
    n_img = len(images)

    for i in range(0, n_img):
        model = load_trained_model(trained_model).to(device)
        img = images[i]
        gradcam = GradCam(
            model=model.model,
            feature_module=model.model.layer4,
            target_layer_names=[f"{labels[i]}"],
            use_cuda=True,
        )
        cam = gradcam(img.unsqueeze(0).requires_grad_(True), preds[i])
        guidedbackprop = GuidedBackpropReLUModel(model=model.model, use_cuda=True)
        gb = guidedbackprop(img.unsqueeze(0).requires_grad_(True), index=None)
        # Deprocess images
        imgs.append(transpscale(unorm(img)))
        cam_gbs.append(
            deprocess_image(np.array([cam, cam, cam]) * gb).transpose((1, 2, 0))
        )
        cams.append((cam * 255).astype("uint8"))
    return imgs, cams, cam_gbs, preds, probs, labels


def plot_gradcam_guidedbackprop(
    imgs: list,
    cams: list,
    cam_gbs: list,
    preds: list,
    probs: list,
    labels: list,
    classes: list,
    rows: int,
):
    # Plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(20, 20))
    k = 0
    for i in range(0, len(imgs)):
        pred = classes[preds[i]].upper().replace("_", "-")
        obs = classes[labels[i]].upper().replace("_", "-")
        for col in range(0, 3):
            ax = fig.add_subplot(rows, 3, k + 1)
            if col == 0:
                plt.imshow(imgs[i])
            elif col == 1:
                plt.imshow(imgs[i])
                plt.imshow(cams[i], cmap=cm.jet, interpolation="nearest", alpha=0.7)
            elif col == 2:
                plt.imshow(cam_gbs[i], cmap=cm.jet, interpolation="nearest")
            ax.set_title(
                f"PRED: {pred} ({round(probs[i] * 100.0, 2)}%)\nOBS: {obs}",
                color=("green" if preds[i] == labels[i] else "red"),
            )
            k += 1
    fig.tight_layout()
    return fig


def get_features_vector(
    img: torch.Tensor, model, layer, device: torch.device, hidden_size: int = 512
):
    # Create a vector of zeros that will hold our feature vector
    # The "avgpool" layer has an output size of 512 by default
    my_embedding = torch.zeros(hidden_size)
    # Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())

    # Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # Run the model on our transformed image
    img = img.to(device)
    with torch.no_grad():
        model(img)
    # Detach our copy function from the layer
    h.remove()
    # Return the feature vector
    return my_embedding


def extract_feature_vectors_of_imgs(
    input_folder: str = "../datasets_2/all",
    transformation: transforms = None,
    batch_size: int = 30,
    shuffle: bool = False,
    model=None,
    layer_name: str = "avgpool",
    hidden_size: int = 512,
    device: torch.device = None,
):
    tp_img_path, fp_img_path = [], []
    tp_data, fp_data = np.empty([0, hidden_size]), np.empty([0, hidden_size])

    # Use the model object to select the desired layer
    layer = model.model._modules.get(layer_name)

    data = ImageFolder(
        root=Path(input_folder),
        transform=transformation,
    )
    data_loader = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, pin_memory=True
    )
    data_iter = iter(data_loader)

    i = 0
    while True:
        try:
            # Iterate
            data_batch = next(data_iter)
            data_images, data_labels = data_batch

            # Get model predictions
            preds, _ = images_to_probs(model, data_images, device)

            # Filter only "school" predictions
            img_index = np.where(np.array(preds) == 1)[0]

            if len(img_index) > 0:
                img_paths = np.array(
                    data.imgs[i * batch_size : batch_size + batch_size * i]
                )[img_index]
                data_images = data_images[img_index]
                labels = np.array(data_labels)[img_index]

                for img, label, img_path in zip(
                    data_images,
                    labels,
                    img_paths,
                ):
                    # Extract feature layers
                    img_feature_vec = get_features_vector(
                        img=img.unsqueeze(0),
                        model=model,
                        layer=layer,
                        device=device,
                        hidden_size=hidden_size,
                    )
                    # Filter TP and FP
                    if label == 1:
                        tp_data = np.vstack((tp_data, img_feature_vec.numpy()))
                        tp_img_path.append(img_path)
                    else:
                        fp_data = np.vstack((fp_data, img_feature_vec.numpy()))
                        fp_img_path.append(img_path)
        except Exception as e:
            print(e)
            break
        i += 1
    return tp_data, tp_img_path, fp_data, fp_img_path


def get_inertia_and_silhouette_by_nclusters(
    data: np.ndarray,
    max_clusters: int = 11,
):
    inertia, silhouette_coeff = [], []

    for k in range(1, max_clusters):
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            max_iter=500,
            n_init=50,
            random_state=0,
            n_jobs=-1,
        ).fit(data)
        inertia.append(kmeans.inertia_)
        if k > 1:
            score = silhouette_score(data, kmeans.labels_)
            silhouette_coeff.append(score)
    return inertia, silhouette_coeff


def get_silhouette_by_nclusters(
    data: np.ndarray,
    max_clusters: int = 16,
    iterations: int = 3,
    algorithm: str = "gmm",
):
    sil_coeff, sil_err = [], []

    for k in range(2, max_clusters):
        tmp_sil = []
        for _ in range(iterations):
            if algorithm == "spectral":
                model = SpectralClustering(
                    n_clusters=k, n_init=10, affinity="nearest_neighbors", n_jobs=-1
                ).fit(data)
                labels = model.labels_
            elif algorithm == "hdbscan":
                model = hdbscan.HDBSCAN(min_cluster_size=k, gen_min_span_tree=True).fit(
                    data
                )
            else:
                model = GMM(
                    n_components=k, covariance_type="full", max_iter=10, n_init=3
                ).fit(data)
                labels = model.predict(data)

            score = silhouette_score(data, labels)
            tmp_sil.append(score)
        val = np.mean(np.array(tmp_sil))
        err = np.std(tmp_sil)
        sil_coeff.append(val)
        sil_err.append(err)
    return sil_coeff, sil_err


def get_images_samples_by_cluster(
    input_folder: str = "../datasets_2/all",
    group_img_paths: list = [],
    batch_size: int = 30,
    transformation: transforms = None,
):
    group_indexes = []

    dataset = ImageFolder(
        root=Path(input_folder),
        transform=transformation,
    )

    for i in range(0, len(dataset)):
        img_path = np.array(dataset.imgs)[i, 0]
        if img_path in group_img_paths:
            group_indexes.append(i)

    trainset = torch.utils.data.Subset(dataset, group_indexes)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
    )
    return trainloader


def gradcam_plots_by_cluster(
    cluster_labels: np.ndarray = None,
    cluster_group: int = 0,
    imgs_paths: list() = [],
    trained_model: str = "resnet18_aa",
    transformation: transforms = None,
    device: torch.device = None,
    nimgs_by_plot: int = 3,
    input_folder: str = "../datasets_2/all",
    output_folder: str = "/home/guzmanlopez/clustering_output/",
    write=True,
):
    # Turn interactive plotting off
    plt.ioff()

    output_folder = Path(f"{output_folder}/group_{cluster_group + 1}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get image paths from labels
    group_img_paths = np.array(imgs_paths)[cluster_labels == cluster_group, 0]
    # Get trainloader samples
    group_samples = get_images_samples_by_cluster(
        input_folder=input_folder,
        group_img_paths=group_img_paths,
        batch_size=nimgs_by_plot,
        transformation=transformation,
    )
    group_iter = iter(group_samples)
    batch = 1
    while True:
        try:
            group_batch = next(group_iter)
            group_images, group_labels = group_batch
            # Calculate and plot grad-CAM and guided backprop
            imgs, cams, cam_gbs, preds, probs, labels = get_gradcam_guidedbackprop(
                trained_model, group_images, group_labels, device=device
            )
            # Plot
            fig = plot_gradcam_guidedbackprop(
                imgs,
                cams,
                cam_gbs,
                preds,
                probs,
                labels,
                ["not_school", "school"],
                nimgs_by_plot,
            )
            if write:
                fig.savefig(output_folder / f"batch_{batch}.png", facecolor="w")
                plt.close(fig)
            else:
                fig.show()
        except Exception as e:
            print(e)
            break
        batch += 1


def plot_inertia(
    inertia: list = [], max_clusters: int = 11, nc: int = 4, title: str = ""
):
    plt.style.use("ggplot")
    plt.plot(range(1, max_clusters), inertia, marker="o")
    plt.title(title)
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.annotate(
        "selected k",
        (nc, inertia[nc - 1]),
        textcoords="offset points",
        xytext=(0, -10),
        ha="right",
    )
    plt.show()


def plot_pca_explained_variance(data: np.ndarray, title: str = "PCA"):
    pca_expvar_sumsum = (
        100 * PCA(random_state=0).fit(data).explained_variance_ratio_.cumsum()
    )
    kl = KneeLocator(
        range(0, data.shape[1]),
        pca_expvar_sumsum,
        curve="concave",
        direction="increasing",
    )
    plt.style.use("ggplot")
    plt.plot(range(1, data.shape[1] + 1), pca_expvar_sumsum)
    plt.plot(kl.knee + 1, pca_expvar_sumsum[kl.knee], marker="o")
    plt.title(title)
    plt.xlabel("Number of components")
    plt.ylabel("Explained variance (%)")
    plt.annotate(
        f"max. curv.: (x:{kl.knee + 1}, y:{round(pca_expvar_sumsum[kl.knee], 2)}%)",
        (kl.knee + 1, pca_expvar_sumsum[kl.knee]),
        textcoords="offset points",
        xytext=(0, -10),
        ha="left",
    )
    plt.show()


def plot_silhouette(
    silhouette_coeff: list = [],
    max_clusters: int = 11,
    nc: int = 3,
    title: str = "",
    errors: list = None,
):
    plt.style.use("ggplot")
    if errors is not None:
        plt.errorbar(range(2, max_clusters), silhouette_coeff, marker="o", yerr=errors)
    else:
        plt.plot(range(2, max_clusters), silhouette_coeff, marker="o")

    plt.title(title)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette coefficient")
    plt.annotate(
        "selected k",
        (nc, silhouette_coeff[nc - 2]),
        textcoords="offset points",
        xytext=(0, -10),
        ha="right",
    )
    plt.show()


def get_top_score_samples_by_cluster(
    labels: np.ndarray, scores: np.ndarray, top_n: int = 3, number_of_clusters: int = 4
):
    highest_probs_idx_by_clust = {}
    scores_map = {}

    for cluster in range(0, number_of_clusters):
        scores_map[cluster] = {"i": [], "score": []}
        for i, score, label in zip(range(0, len(labels)), scores, labels):
            if cluster == label:
                scores_map[cluster]["i"].append(i)
                scores_map[cluster]["score"].append(score)

    for cluster in range(0, number_of_clusters):
        max_scores_idx = np.flip(
            np.argsort(np.array(scores_map[cluster]["score"]))[-top_n:]
        )
        true_max_scores_idx = np.array(scores_map[cluster]["i"])[max_scores_idx]
        highest_probs_idx_by_clust[cluster] = true_max_scores_idx
    return highest_probs_idx_by_clust


def plot_aic_bic_criterion(
    data: np.ndarray, max_clusters: int = 11, random_state: int = 0, bic: bool = False
):
    cv_types = ["spherical", "tied", "diag", "full"]
    for cv_type in cv_types:
        models = [
            GMM(
                n,
                covariance_type=cv_type,
                max_iter=10,
                n_init=2,
                random_state=random_state,
            ).fit(data)
            for n in np.arange(1, max_clusters)
        ]
        plt.plot(
            np.arange(1, max_clusters),
            [m.aic(data) for m in models],
            label=f"AIC - {cv_type}",
            marker="o",
        )
        if bic:
            plt.plot(
                np.arange(1, max_clusters),
                [m.bic(data) for m in models],
                label=f"BIC - {cv_type}",
                marker="o",
            )
    plt.style.use("ggplot")
    plt.title("GMM")
    plt.legend(loc="upper right")
    plt.xlabel("#clusters")


def get_count_of_imgs_by_group(labels: np.ndarray):
    unique, counts = np.unique(labels, return_counts=True)
    unique = [f"C{c}" for c in unique.astype(str)]
    return dict(zip(unique, counts))


def plot_clusters(
    projection: np.ndarray, labels: np.ndarray, alpha: float = 0.75, title: str = ""
):
    for g in np.unique(labels):
        gsamples = projection.T[:, labels == g]
        plt.scatter(*gsamples, s=50, linewidth=0, alpha=alpha, label=f"C{g}")
    plt.style.use("ggplot")
    plt.legend(loc="lower right")
    plt.title(title)


def get_paths_probs_of_school_model_preds(
    input_folder: str = "../datasets_2/all",
    transformation: transforms = None,
    batch_size: int = 30,
    shuffle: bool = False,
    model=None,
    device: torch.device = None,
):
    tp_prob, fp_prob = [], []
    tp_img_path, fp_img_path = [], []

    data = ImageFolder(
        root=Path(input_folder),
        transform=transformation,
    )
    data_loader = DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, pin_memory=True
    )
    data_iter = iter(data_loader)

    i = 0
    while True:
        try:
            # Iterate
            data_batch = next(data_iter)
            data_images, data_labels = data_batch

            # Get model predictions
            preds, probs = images_to_probs(model, data_images, device)

            # Filter only "school" predictions
            img_index = np.where(np.array(preds) == 1)[0]

            if len(img_index) > 0:
                img_paths = np.array(
                    data.imgs[i * batch_size : batch_size + batch_size * i]
                )[img_index]
                labels = np.array(data_labels)[img_index]
                probs = np.array(probs)[img_index]

                for prob, label, img_path in zip(
                    probs,
                    labels,
                    img_paths,
                ):
                    if label == 1:
                        tp_prob.append(prob)
                        tp_img_path.append(img_path)
                    else:
                        fp_prob.append(prob)
                        fp_img_path.append(img_path)
        except Exception as e:
            print(e)
            break
        i += 1
    return tp_prob, tp_img_path, fp_prob, fp_img_path


def get_predictions_by_cluster(
    cluster_img_label_mapping: dict(),
    cluster_group: int = 0,
    imgs_paths: list() = [],
    model=None,
    transformation: transforms = None,
    device: torch.device = None,
    input_folder: str = "../datasets_2/all",
    batch_size: int = 30,
):
    predictions = []
    test_images = []
    true_labels = []
    used_labels = set()

    root_dir = input_folder.split("/")[-1]
    imgs_root_dir = imgs_paths[0][0].split("/")[-3]

    # Get image paths from cluster groups labels and img ids
    group_img_ids = [
        img_id
        for img_id, group in cluster_img_label_mapping.items()
        if group == cluster_group
    ]

    group_img_paths = []

    for img_id in group_img_ids:
        for img_path, _ in imgs_paths:
            if img_id in img_path:
                # Match paths: "all" folder dataset and "test" folder dataset
                group_img_paths.append(img_path.replace(imgs_root_dir, root_dir))

    # Get loader samples
    group_samples = get_images_samples_by_cluster(
        input_folder=input_folder,
        group_img_paths=group_img_paths,
        batch_size=batch_size,
        transformation=transformation,
    )

    with torch.no_grad():
        for _, data in enumerate(group_samples, 0):
            images, labels = data
            labels = labels.numpy()
            true_labels.extend(labels)
            used_labels.update(labels)
            test_images.extend(images)
            images = images.to(device)
            current_preds = model(images)
            predictions.extend(F.softmax(current_preds.detach().cpu(), dim=1))

    # Get predictions and probabilities
    predictions = torch.stack(predictions)
    probs = torch.max(predictions, dim=1)[0].numpy()
    votes = torch.argmax(predictions, dim=1).numpy()

    return true_labels, votes, probs, used_labels
