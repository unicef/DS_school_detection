# %% [markdown]
# # Covariates correlation
# ## Data from [WorldPop](https://www.worldpop.org)

# %%
import itertools
import pickle

# Modify sys path to import top-level modules
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import transforms

sys.path.append("..")
from utils.geoutils import get_samples_from_raster
from utils.utils import (
    get_paths_probs_of_school_model_preds,
    get_predictions_by_cluster,
    load_trained_model,
)

# Fix random seed for reproducibility
seed = 9102
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Global device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using GPU") if torch.cuda.is_available() else print("Using CPU")

# Image folder path (zoom level 18)
image_input_folder = "../datasets_2/all"

# Pretrained models
trained_model = ["resnet18", "resnet18_aa", "mobilenetv2", "mobilenetv2_aa"][1]

# %% [markdown]
# ## Load model and input data

# %% [markdown]
# ### Data Augmentations
# Resize the image to a smaller one and normalize (substract the mean and divide by the standar deviation)

# %%
image_size = (224, 224)

# Using the mean and std of Imagenet for transformations
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

test_transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ]
)

# %%
# Load trained model
model = load_trained_model(trained_model).to(device)

# Set model to evaluation mode
model.eval()

# %%
# Get image paths and probs of positive predictions ("school")
tp_prob, tp_img_path, fp_prob, fp_img_path = get_paths_probs_of_school_model_preds(
    input_folder=image_input_folder,
    transformation=test_transform,
    batch_size=30,
    model=model,
    device=device,
)

all_prob = tp_prob + fp_prob
all_img_path = tp_img_path + fp_img_path


# %%
# Load cluster k-means label-image mapping
img_cluster_map = pickle.load(open("../clustering/img_cluster_map.pkl", "rb"))

# %%
# Load raster covariate

# ylabel = "Log [Distance to road (Km)]"
# source = "data/col_osm_dst_road_100m_2016.tif"

# ylabel = "Log [Distance to waterway (Km)]"
# source = "data/col_osm_dst_waterway_100m_2016.tif"

# ylabel = "Distance to IUCN areas (Km)"
# source = "data/col_wdpa_dst_cat1_100m_2017.tif"

# ylabel = "Elevation (m)"
# source = "data/col_srtm_topo_100m.tif"

# ylabel = "Slope (degrees)"
# source = "data/col_srtm_slope_100m.tif"

# ylabel = "Female population (5 to 10 years)"
# source = "data/col_f_5_2020.tif"

ylabel = "Male population (5 to 10 years)"
source = "data/col_m_5_2020.tif"

covariate_samples = get_samples_from_raster(source, all_img_path)

# %%
# Mapping clusters with raster samples
covariate_samples_cluster = {}

# TODO: add probabilities
for img_name, samples in covariate_samples.items():
    d = {"cluster": img_cluster_map.get(img_name), "samples": samples}
    covariate_samples_cluster[img_name] = d

# %%
samples_cluster_0 = [
    v.get("samples")
    for _, v in covariate_samples_cluster.items()
    if v.get("cluster") == 0
]

samples_cluster_1 = [
    v.get("samples")
    for _, v in covariate_samples_cluster.items()
    if v.get("cluster") == 1
]

samples_cluster_2 = [
    v.get("samples")
    for _, v in covariate_samples_cluster.items()
    if v.get("cluster") == 2
]

samples_cluster_3 = [
    v.get("samples")
    for _, v in covariate_samples_cluster.items()
    if v.get("cluster") == 3
]

# %%
samples_cluster_0 = np.concatenate(samples_cluster_0)
samples_cluster_1 = np.concatenate(samples_cluster_1)
samples_cluster_2 = np.concatenate(samples_cluster_2)
samples_cluster_3 = np.concatenate(samples_cluster_3)

# For population
samples_cluster_0 = np.array(np.where(samples_cluster_0 == -99999, None, samples_cluster_0), dtype=np.float64)
samples_cluster_1 = np.array(np.where(samples_cluster_1 == -99999, None, samples_cluster_1), dtype=np.float64)  
samples_cluster_2 = np.array(np.where(samples_cluster_2 == -99999, None, samples_cluster_2), dtype=np.float64)  
samples_cluster_3 = np.array(np.where(samples_cluster_3 == -99999, None, samples_cluster_3), dtype=np.float64)  

data = [samples_cluster_0, samples_cluster_1, samples_cluster_2, samples_cluster_3]

print(f"C0 mean: {np.nanmean(samples_cluster_0)}")
print(f"C1 mean: {np.nanmean(samples_cluster_1)}")
print(f"C2 mean: {np.nanmean(samples_cluster_2)}")
print(f"C3 mean: {np.nanmean(samples_cluster_3)}")


# %%
df = pd.DataFrame({"values": [], "cluster": []})
df_log = pd.DataFrame({"values": [], "cluster": []})

for c in range(0, 4):
    df = df.append(pd.DataFrame({"values": data[c], "cluster": f"C{c}"}))
    df_log = df_log.append(
        pd.DataFrame({"values": np.log(data[c] + 1), "cluster": f"C{c}"})
    )

plt.style.use("ggplot")


# %%
log_boxplot = sns.boxplot(
    x="cluster",
    y="values",
    linewidth=0.5,
    flierprops=dict(marker="o", markersize=1),
    data=df,
)
log_boxplot.set(xlabel="Cluster", ylabel=ylabel, ylim=(0, None))

# %%
log_stripplot = sns.stripplot(
    x="cluster", y="values", jitter=0.45, alpha=0.2, data=df
)
log_stripplot.set(xlabel="Cluster", ylabel=ylabel)

# %%
log_violinplot = sns.violinplot(x="cluster", y="values", linewidth=0.5, data=df)
log_violinplot.set(xlabel="Cluster", ylabel=ylabel)


# %% [markdown]
# Accuracies analysis

# %%
def plot_confusion_matrix(
    cm, used_labels, target_names, title="Confusion matrix", cmap="Blues"
):
    accuracy = np.trace(cm) / np.sum(cm).astype("float")
    misclass = 1 - accuracy
    cmap = plt.get_cmap(cmap)

    plt.figure(figsize=(5, 5))
    tick_marks = np.arange(len(used_labels))
    names = [target_names[idx] for idx in used_labels]
    plt.xticks(tick_marks, names, rotation=90)
    plt.yticks(tick_marks, names)
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    # plt.colorbar()

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "{:,}".format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\n\nAccuracy: {:0.2f}%; Misclass: {:0.2f}%".format(
            accuracy * 100, misclass * 100
        )
    )
    plt.show()


# %%
plt.style.use("default")

for dataset in ["train", "test"]:
    for cl in range(0, 4):
        
        # Get model predictions
        true_labels, votes, probs, used_labels = get_predictions_by_cluster(
            cluster_img_label_mapping=img_cluster_map,
            cluster_group=cl,
            imgs_paths=all_img_path,
            model=model,
            transformation=test_transform,
            device=device,
            input_folder=f"../datasets_2/{dataset}",
            batch_size=30,
        )

        # Plot confusion matrix of results
        conf_mat = confusion_matrix(true_labels, votes)
        plot_confusion_matrix(
            cm=conf_mat,
            used_labels=used_labels,
            target_names=np.array(["not_school", "school"]),
            title=f"Dataset: {dataset.title()} - Cluster: {cl}",
        )


# %%
