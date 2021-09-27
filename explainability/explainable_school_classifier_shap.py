# %% [markdown]
# # Explainability of the school / not_school classifier

# %%
import numpy as np
import shap
import torch
from pathlib import Path, PosixPath
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Modify sys path to import top-level modules
import sys

sys.path.append("..")
from utils.utils import TranspScale, UnNormalize, load_trained_model, images_to_probs
from utils.geoutils import georef_tile, get_tile_bbox_from_fname, shap_raster_sum_bands

# Fix random seed for reproducibility
seed = 9102
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Global device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using GPU") if torch.cuda.is_available() else print("Using CPU")

# Image folder path (zoom level 18)
base_path = Path("../datasets_2")

# Pretrained models
trained_model = ["resnet18", "resnet18_aa", "mobilenetv2", "mobilenetv2_aa"][0]


# %% [markdown]
# ## Data Augmentations
#
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

unorm = UnNormalize(means, stds)
transpscale = TranspScale()

# %%
# Load images for train
train = ImageFolder(root=Path(base_path / "train"), transform=test_transform)
train_loader = DataLoader(train, batch_size=150, shuffle=True, pin_memory=True)
train_iter = iter(train_loader)

# Iterate
train_batch = next(train_iter)
train_images, train_labels = train_batch

# %%
# Load test
bs = 10

# Urban images
test_urban = ImageFolder(
    root=Path(base_path / "test_subselect" / "urban"), transform=test_transform
)
test_urban_loader = DataLoader(
    test_urban, batch_size=bs, shuffle=False, pin_memory=True
)
test_urban_iter = iter(test_urban_loader)

# Not-urban images
test_not_urban = ImageFolder(
    root=Path(base_path / "test_subselect" / "not_urban"), transform=test_transform
)
test_not_urban_loader = DataLoader(
    test_not_urban, batch_size=bs, shuffle=False, pin_memory=True
)
test_not_urban_iter = iter(test_not_urban_loader)

# %%
# Iterate
test_urban_batch = next(test_urban_iter)
test_urban_images, test_urban_labels = test_urban_batch

test_not_urban_batch = next(test_not_urban_iter)
test_not_urban_images, test_not_urban_labels = test_not_urban_batch

# %% [markdown]
# ## SHAP (SHapley Additive exPlanations)
# [SHAP (SHapley Additive exPlanations)](https://github.com/slundberg/shap) is a game theoretic approach to explain the output
# of any machine learning model. It connects optimal credit allocation with local
# explanations using the classic Shapley values from game theory and their related extensions
#  (see papers for details and citations).

# %%
# Load trained model
model = load_trained_model(trained_model).to(device).eval()


# %% [markdown]
# ### DeepExplainer

# %%
de = shap.DeepExplainer(model, train_images.to(device))

# %%
de_shap_values, de_indexes = de.shap_values(
    X=test_urban_images, ranked_outputs=2, output_rank_order="max"
)

# %%
# Get the names for the classes
de_index_names = np.vectorize(lambda x: ["not-school", "school"][x])(
    de_indexes.detach().cpu()
)

# Get the SHAP values for the classes
de_shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in de_shap_values]

# Get the raw images
de_test_numpy = np.empty((0, 224, 224, 3))

for i, t in enumerate(test_urban_images):
    img = transpscale(unorm(t))
    de_test_numpy = np.insert(de_test_numpy, i, img, axis=0)

# Plot the feature attributions
# shap.image_plot(shap_values=de_shap_numpy, pixel_values=de_test_numpy, labels=de_index_names)

# %%
# Plot images
for i in range(0, len(test_urban_images) - 1):
    shap.image_plot(
        shap_values=[np.array([de_shap_numpy[0][i]]), np.array([de_shap_numpy[1][i]])],
        pixel_values=np.array([de_test_numpy[i]]),
        labels=np.array([de_index_names[i]]),
    )


# %%[markdown]
# ## Georeferencing of SHAP values

# %%
bs = 20
width, height = image_size[0], image_size[1]

for category in ["urban", "not_urban"]:
    test = ImageFolder(
        root=Path(base_path / "test_subselect" / f"{category}"),
        transform=test_transform,
    )
    test_loader = DataLoader(test, batch_size=bs, shuffle=False, pin_memory=True)
    test_iter = iter(test_loader)

    i = 0
    while True:
        try:
            # Iterate
            test_batch = next(test_iter)
            test_images, test_labels = test_batch

            # Get model predictions
            preds, probs = images_to_probs(model, test_images, device)

            # Filter only "school" predictions
            img_index = np.where(np.array(preds) == 1)[0]

            if len(img_index) > 0:
                img_paths = np.array(test.imgs[i * bs : bs + bs * i])[img_index]
                test_images = test_images[img_index]
                labels = np.array(test_labels)[img_index]

                # Calculate SHAP values
                de_shap_values, de_indexes = de.shap_values(
                    X=test_images, ranked_outputs=2, output_rank_order="max"
                )

                # Get the SHAP values for the classes
                de_shap_numpy = [
                    np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in de_shap_values
                ]

                for img_path, img, shap_values, label in zip(
                    img_paths, test_images, de_shap_numpy[0], labels,
                ):
                    img_path = Path(img_path[0])
                    # Get tiles and zoom from filename
                    tile_bbox = get_tile_bbox_from_fname(img_path)

                    output_folder = (
                        Path(
                            str(img_path.parent).replace(
                                "test_subselect", "raster_shap"
                            )
                        )
                        / trained_model
                    )

                    # True positives and false positives discrimination
                    if label == 1:
                        output_folder = output_folder / "tp"
                    elif label == 0:
                        output_folder = output_folder / "fp"

                    # Create required out folders structure
                    output_folder.mkdir(parents=True, exist_ok=True)

                    georef_tile(
                        input_img=shap_values,
                        input_filepath=img_path,
                        width=width,
                        height=height,
                        tile_bbox=tile_bbox,
                        output_folder=output_folder,
                        out_ext="tiff",
                        img_type="shap",
                        dtype="float64",
                    )
        except Exception as e:
            print(e)
            break
        i += 1

# %%
# Sum SHAP values over the three channels (RGB)
shap_raster_sum_bands(
    input_folder=Path("../datasets_2/raster_shap/urban/school/resnet18/tp/")
)
shap_raster_sum_bands(
    input_folder=Path("../datasets_2/raster_shap/urban/not_school/resnet18/fp/")
)
shap_raster_sum_bands(
    input_folder=Path("../datasets_2/raster_shap/not_urban/school/resnet18/tp/")
)
shap_raster_sum_bands(
    input_folder=Path("../datasets_2/raster_shap/not_urban/not_school/resnet18/fp/")
)


# %%
