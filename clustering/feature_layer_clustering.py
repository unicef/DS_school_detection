# %% [markdown]
# # Clustering
# ## Methods based on the last feature vector for the school / not_school UNICEF image classifier

# %%
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from kneed import KneeLocator
from pathlib import Path
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms

# Modify sys path to import top-level modules
import sys

sys.path.append("..")
from utils.utils import (
    load_trained_model,
    extract_feature_vectors_of_imgs,
    get_inertia_and_silhouette_by_nclusters,
    get_silhouette_by_nclusters,
    gradcam_plots_by_cluster,
    plot_clusters,
    plot_inertia,
    plot_silhouette,
    get_top_score_samples_by_cluster,
    plot_aic_bic_criterion,
    get_count_of_imgs_by_group,
    plot_pca_explained_variance,
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

# Maximum number of clusters to evaluate and selected number of clusters
max_clusters, nc = 15, 4


# %% [markdown]
# ## Load model and input data for clustering

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


# %% [markdown]
# ### Extract Feature Layers from images

# %%
# Load trained model
model = load_trained_model(trained_model).to(device)

# Set model to evaluation mode
model.eval()

# %%
# Get all feature layers from all model 'school' label predictions (TP + FP)
tp_data, tp_img_path, fp_data, fp_img_path = extract_feature_vectors_of_imgs(
    input_folder=image_input_folder,
    transformation=test_transform,
    batch_size=30,
    model=model,
    layer_name="avgpool",
    device=device,
)


# %%
all_data = np.vstack((tp_data, fp_data))
all_img_path = tp_img_path + fp_img_path


# %% [markdown]
# ### Scale data

# %%
scaler = MinMaxScaler(feature_range=(0, 1))
all_data_scaled = scaler.fit_transform(all_data)

# %% [markdown]
# ## Dimensionality reduction: PCA

# %%
# Find maximum curvature
plot_pca_explained_variance(all_data_scaled)

# %%
all_data_scaled_pca = PCA(n_components=64, random_state=0).fit_transform(all_data_scaled)
print(
    f"Number of features after PCA: {all_data_scaled_pca.shape[1]} of {all_data_scaled.shape[1]}"
)

# %% [markdown]
# ## Dimensionality reduction for visualization: t-SNE
all_data_projection = TSNE(n_components=2, init="pca", random_state=0).fit_transform(
    all_data_scaled
)
all_data_projection_pca = TSNE(n_components=2, init="pca", random_state=0).fit_transform(
    all_data_scaled_pca
)

# %% [markdown]
# ## k-means

# %% [markdown]
# ### Select k: Elbow method

# %%
kmeans_inertia, kmeans_sil_coeff = get_inertia_and_silhouette_by_nclusters(
    all_data_scaled, max_clusters
)

# %%
# Find best "k" groups based on the point of maximum curvature of inertia vs number of clusters
kl = KneeLocator(
    range(1, max_clusters), kmeans_inertia, curve="convex", direction="decreasing"
)
print(f"Selected k: {kl.elbow}")

# %%
plot_inertia(kmeans_inertia, max_clusters, nc, "k-means: all data + norm\nElbow method")

# %% [markdown]
# ### Silhouette coefficient: measure of cluster cohesion and separation

# %%
plot_silhouette(
    kmeans_sil_coeff,
    max_clusters,
    nc,
    "k-means: all data + norm\nMean Silhouette",
)

# %%
kmeans = KMeans(
    n_clusters=nc,
    init="k-means++",
    max_iter=500,
    n_init=50,
    random_state=0,
    n_jobs=-1,
).fit(all_data_scaled)

print(f"Number of imgs per cluster: {get_count_of_imgs_by_group(kmeans.labels_)}")
plot_clusters(all_data_projection, kmeans.labels_, title="t-SNE for k-means")

# %%
# Get the highest silhouette indexes
kmeans_sample_scores = silhouette_samples(all_data_scaled, kmeans.labels_)
kmeans_top_scores_imgs = get_top_score_samples_by_cluster(
    labels=kmeans.labels_, scores=kmeans_sample_scores, top_n=12, number_of_clusters=nc
)

# %%
# Plot or write mosaics of most represented images by cluster
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=kmeans.labels_[kmeans_top_scores_imgs[cluster]],
        cluster_group=cluster,
        imgs_paths=[all_img_path[i] for i in kmeans_top_scores_imgs[cluster]],
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/kmeans/representative/",
        write=True,
    )

# %%[markdown]
# ### Export mosaic with grad-CAM by cluster groups images

# %%
# Write figures to output folder
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=kmeans.labels_,
        cluster_group=cluster,
        imgs_paths=all_img_path,
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/kmeans/",
        write=True,
    )

# %% [markdown]
# ### Use reduced dimensionality data from PCA

# %%
kmeans_pca = KMeans(
    n_clusters=nc,
    init="k-means++",
    max_iter=500,
    n_init=50,
    random_state=0,
    n_jobs=-1,
).fit(all_data_scaled_pca)

print(f"Number of imgs per cluster: {get_count_of_imgs_by_group(kmeans_pca.labels_)}")
plot_clusters(
    all_data_projection_pca, kmeans_pca.labels_, title="t-SNE for k-means with PCA"
)

# %%
# Save labels and group mapping file
img_cluster_map = dict()

for label, img_path in zip(kmeans_pca.labels_, all_img_path):
    f = Path(img_path[0])
    name = f.name.replace(f.suffix, "")
    img_cluster_map[name] = label

pickle.dump(img_cluster_map, open("img_cluster_map.pkl", "wb"))

# %%
kmeans_inertia_pca, kmeans_sil_coeff_pca = get_inertia_and_silhouette_by_nclusters(
    all_data_scaled_pca, max_clusters
)

# %%
plot_inertia(
    kmeans_inertia_pca, max_clusters, nc, "k-means: all data + norm + PCA\nElbow method"
)
plot_silhouette(
    kmeans_sil_coeff_pca,
    max_clusters,
    nc,
    "k-means: all data + norm + PCA\nMean Silhouette",
)

# %%[markdown]
# ### Export mosaic with grad-CAM by cluster groups images

# %%
# Get the highest silhouette indexes
kmeans_sample_scores_pca = silhouette_samples(all_data_scaled_pca, kmeans_pca.labels_)
kmeans_top_scores_imgs_pca = get_top_score_samples_by_cluster(
    labels=kmeans_pca.labels_,
    scores=kmeans_sample_scores_pca,
    top_n=12,
    number_of_clusters=nc,
)

# Plot or write mosaics of most represented images by cluster
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=kmeans_pca.labels_[kmeans_top_scores_imgs_pca[cluster]],
        cluster_group=cluster,
        imgs_paths=[all_img_path[i] for i in kmeans_top_scores_imgs_pca[cluster]],
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/kmeans_pca/representative/",
        write=True,
    )

# %%
# Write figures to output folder
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=kmeans_pca.labels_,
        cluster_group=cluster,
        imgs_paths=all_img_path,
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/kmeans_pca/",
        write=True,
    )

# %% [markdown]
# ## Gaussian Mixture Models

# %% [markdown]
# ### Find number of clusters using Akaike criterion

# %%
plot_aic_bic_criterion(data=all_data_scaled, max_clusters=max_clusters, bic=False)

# %% [markdown]
# ### Find number of clusters using Silhouette

# %%
gmm_sil_coeff, gmm_sil_err = get_silhouette_by_nclusters(
    all_data_scaled, max_clusters, iterations=3
)

plot_silhouette(
    gmm_sil_coeff,
    max_clusters,
    nc,
    "GMM: all data + norm\nMean Silhouette",
    gmm_sil_err,
)

# %% [markdown]
# ### Fit GMM to data

# %%
gmm = GMM(
    n_components=nc,
    covariance_type="full",
    max_iter=500,
    n_init=50,
    random_state=0,
    verbose=0,
).fit(all_data_scaled)

# Get labels and probabilities
gmm_labels = gmm.predict(all_data_scaled)
gmm_probs = gmm.predict_proba(all_data_scaled)
gmm_scores = gmm.score_samples(all_data_scaled)

print(f"Number of imgs per cluster: {get_count_of_imgs_by_group(gmm_labels)}")
plot_clusters(all_data_projection, gmm_labels, title="t-SNE for GMM")

# %%
gmm_top_scores_imgs = get_top_score_samples_by_cluster(
    labels=gmm_labels, scores=gmm_scores, top_n=12, number_of_clusters=nc
)

# Plot or write mosaics of most represented images by cluster
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=gmm_labels[gmm_top_scores_imgs[cluster]],
        cluster_group=cluster,
        imgs_paths=[all_img_path[i] for i in gmm_top_scores_imgs[cluster]],
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/gmm/representative/",
        write=True,
    )


# %%
# Write figures to output folder
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=gmm_labels,
        cluster_group=cluster,
        imgs_paths=all_img_path,
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/gmm/",
        write=True,
    )

# %% [markdown]
# ### Use reduced dimensionality data from PCA

# %% [markdown]
# ### Find number of clusters using Akaike criterion

# %%
plot_aic_bic_criterion(data=all_data_scaled_pca, max_clusters=max_clusters, bic=False)


# %% [markdown]
# ### Find number of clusters using Silhouette

# %%
gmm_sil_coeff_pca, gmm_sil_err_pca = get_silhouette_by_nclusters(
    all_data_scaled_pca, max_clusters, iterations=3
)

plot_silhouette(
    gmm_sil_coeff_pca,
    max_clusters,
    nc,
    "GMM: all data + norm + PCA\nMean Silhouette",
    gmm_sil_err_pca,
)


# %%
gmm_pca = GMM(
    n_components=nc,
    covariance_type="full",
    max_iter=500,
    n_init=50,
    random_state=0,
    verbose=0,
).fit(all_data_scaled_pca)

# Get labels and probabilities
gmm_labels_pca = gmm_pca.predict(all_data_scaled_pca)
gmm_probs_pca = gmm_pca.predict_proba(all_data_scaled_pca)
gmm_scores_pca = gmm_pca.score_samples(all_data_scaled_pca)

print(f"Number of imgs per cluster: {get_count_of_imgs_by_group(gmm_labels_pca)}")
plot_clusters(all_data_projection_pca, gmm_labels_pca, title="t-SNE for GMM with PCA")

# %%
gmm_top_scores_imgs_pca = get_top_score_samples_by_cluster(
    labels=gmm_labels_pca, scores=gmm_scores_pca, top_n=12, number_of_clusters=nc
)

# Plot or write mosaics of most represented images by cluster
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=gmm_labels_pca[gmm_top_scores_imgs_pca[cluster]],
        cluster_group=cluster,
        imgs_paths=[all_img_path[i] for i in gmm_top_scores_imgs_pca[cluster]],
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/gmm_pca/representative/",
        write=True,
    )


# %%
# Write figures to output folder
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=gmm_labels_pca,
        cluster_group=cluster,
        imgs_paths=all_img_path,
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/gmm_pca/",
        write=True,
    )


# %% [markdown]
# ## Scpectral Clustering

# %%
spec_sil_coeff, spec_sil_err = get_silhouette_by_nclusters(
    all_data_scaled,
    max_clusters,
    iterations=10,
    algorithm="spectral",
)

plot_silhouette(
    spec_sil_coeff,
    max_clusters,
    nc,
    "Spectral Clustering: all data + norm\nMean Silhouette",
    spec_sil_err,
)


# %%
# Building the clustering model
spec_model = SpectralClustering(
    n_clusters=nc,
    affinity="nearest_neighbors",
    n_init=20,
    random_state=0,
    n_jobs=-1,
).fit(all_data_scaled)

spec_labels = spec_model.labels_
spec_scores = silhouette_samples(all_data_scaled, spec_labels)

print(f"Number of imgs per cluster: {get_count_of_imgs_by_group(spec_labels)}")
plot_clusters(all_data_projection, spec_labels, title="t-SNE for Spectral")

# %%
spec_top_scores_imgs = get_top_score_samples_by_cluster(
    labels=spec_labels, scores=spec_scores, top_n=12, number_of_clusters=nc
)

# Plot or write mosaics of most represented images by cluster
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=spec_labels[spec_top_scores_imgs[cluster]],
        cluster_group=cluster,
        imgs_paths=[all_img_path[i] for i in spec_top_scores_imgs[cluster]],
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/spec/representative/",
        write=True,
    )

# %%[markdown]
# ### Export mosaic with grad-CAM by cluster groups images

# %%
# Write figures to output folder
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=spec_labels,
        cluster_group=cluster,
        imgs_paths=all_img_path,
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/spec/",
        write=True,
    )

# %% [markdown]
# ### Use reduced dimensionality data from PCA

# %%
spec_sil_coeff_pca, spec_sil_err_pca = get_silhouette_by_nclusters(
    all_data_scaled_pca,
    max_clusters,
    iterations=10,
    algorithm="spectral",
)

plot_silhouette(
    spec_sil_coeff_pca,
    max_clusters,
    nc,
    "Spectral Clustering: all data + norm + PCA\nMean Silhouette",
    spec_sil_err_pca,
)


# %%
# Building the clustering model
spec_model_pca = SpectralClustering(
    n_clusters=nc,
    affinity="nearest_neighbors",
    n_init=20,
    random_state=0,
    n_jobs=-1,
).fit(all_data_scaled_pca)

spec_labels_pca = spec_model_pca.labels_
spec_scores_pca = silhouette_samples(all_data_scaled_pca, spec_labels_pca)

print(f"Number of imgs per cluster: {get_count_of_imgs_by_group(spec_labels_pca)}")
plot_clusters(
    all_data_projection_pca, spec_labels_pca, title="t-SNE for Spectral with PCA"
)

# %%
spec_top_scores_imgs_pca = get_top_score_samples_by_cluster(
    labels=spec_labels_pca, scores=spec_scores_pca, top_n=12, number_of_clusters=nc
)

# Plot or write mosaics of most represented images by cluster
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=spec_labels_pca[spec_top_scores_imgs_pca[cluster]],
        cluster_group=cluster,
        imgs_paths=[all_img_path[i] for i in spec_top_scores_imgs_pca[cluster]],
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/spec_pca/representative/",
        write=True,
    )

# %%[markdown]
# ### Export mosaic with grad-CAM by cluster groups images

# %%
# Write figures to output folder
for cluster in range(0, nc):
    gradcam_plots_by_cluster(
        cluster_labels=spec_labels_pca,
        cluster_group=cluster,
        imgs_paths=all_img_path,
        trained_model=trained_model,
        transformation=test_transform,
        device=device,
        nimgs_by_plot=3,
        input_folder=image_input_folder,
        output_folder="/home/guzmanlopez/clustering_output/spec_pca/",
        write=True,
    )

# %% [markdown]
# ## Compare silhouette values among different cluster algorithms

# %%
n_clusters = range(2, max_clusters)

plt.style.use("ggplot")
plt.plot(n_clusters, kmeans_sil_coeff, marker=".", label="k-means")
plt.plot(n_clusters, kmeans_sil_coeff_pca, marker=".", label="k-means w/PCA")
plt.plot(n_clusters, gmm_sil_coeff, marker=".", label="GMM")
plt.plot(n_clusters, gmm_sil_coeff_pca, marker=".", label="GMM w/PCA")
plt.plot(n_clusters, spec_sil_coeff, marker=".", label="Spectral")
plt.plot(n_clusters, spec_sil_coeff_pca, marker=".", label="Spectral w/PCA")
plt.title("Silhouette comparision among different cluster algorithms")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette coefficient")
plt.legend(loc="upper right")
plt.show()

# %%