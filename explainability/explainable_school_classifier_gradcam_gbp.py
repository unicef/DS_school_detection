# %% [markdown]
# # Explainability of the school / not_school classifier

# %%
import numpy as np
import torch
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Modify sys path to import top-level modules
import sys

sys.path.append("..")
from utils.utils import get_gradcam_guidedbackprop, plot_gradcam_guidedbackprop
from utils.geoutils import georef_tile, get_tile_bbox_from_fname

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
trained_model = ["resnet18", "resnet18_aa", "mobilenetv2", "mobilenetv2_aa"][1]


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


# %% [markdown]
# ## Grad-CAM (Gradient Class Activation Maps)
# [Grad-CAMs](https://github.com/jacobgil/pytorch-grad-cam) are a visualization technique for deep learning networks.

# %%
# Load urban images
bs = 3
test_urban = ImageFolder(
    root=Path(base_path / "test_subselect" / "urban"), transform=test_transform
)
test_urban_loader = DataLoader(
    test_urban, batch_size=bs, shuffle=False, pin_memory=True
)
test_urban_iter = iter(test_urban_loader)

# %%
# Iterate
test_urban_batch = next(test_urban_iter)
test_urban_images, test_urban_labels = test_urban_batch

# Calculate and plot grad-CAM and guided backprop
imgs, cams, cam_gbs, preds, probs, labels = get_gradcam_guidedbackprop(
    trained_model, test_urban_images, test_urban_labels, device=device
)
plot_gradcam_guidedbackprop(
    imgs, cams, cam_gbs, preds, probs, labels, test_urban.classes, 3
)

# %%
# Load not-urban images
bs = 3
test_not_urban = ImageFolder(
    root=Path(base_path / "test_subselect" / "not_urban"), transform=test_transform
)
test_not_urban_loader = DataLoader(
    test_not_urban, batch_size=bs, shuffle=False, pin_memory=True
)
test_not_urban_iter = iter(test_not_urban_loader)

# %%
# Iterate
test_not_urban_batch = next(test_not_urban_iter)
test_not_urban_images, test_not_urban_labels = test_not_urban_batch

# Calculate and plot grad-CAM and guided backprop
imgs, cams, cam_gbs, preds, probs, labels = get_gradcam_guidedbackprop(
    trained_model, test_not_urban_images, test_not_urban_labels, device=device
)
plot_gradcam_guidedbackprop(
    imgs, cams, cam_gbs, preds, probs, labels, test_not_urban.classes, 3
)

# %%[markdown]
# ## Georeferencing of tiles

# %%
bs = 30
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

            # Calculate and plot grad-CAM and guided backprop
            imgs, cams, cam_gbs, preds, probs, labels = get_gradcam_guidedbackprop(
                trained_model, test_images, test_labels, device=device
            )

            for img_path, img, cam, cam_gb, pred, label in zip(
                test.imgs[i * bs : bs + bs * i], imgs, cams, cam_gbs, preds, labels
            ):
                if pred == 1:
                    img_path = Path(img_path[0])

                    # Get tiles and zoom from filename
                    tile_bbox = get_tile_bbox_from_fname(img_path)

                    output_folder = (
                        Path(
                            str(img_path.parent).replace(
                                "test_subselect", "raster_gradcam"
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
                        input_img=img,
                        input_filepath=img_path,
                        width=width,
                        height=height,
                        tile_bbox=tile_bbox,
                        output_folder=output_folder,
                        out_ext="tiff",
                        img_type="raw",
                    )

                    georef_tile(
                        input_img=cam.reshape(width, height, 1),
                        input_filepath=img_path,
                        width=width,
                        height=height,
                        tile_bbox=tile_bbox,
                        output_folder=output_folder,
                        out_ext="tiff",
                        img_type="cam",
                    )

                    georef_tile(
                        input_img=cam_gb,
                        input_filepath=img_path,
                        width=width,
                        height=height,
                        tile_bbox=tile_bbox,
                        output_folder=output_folder,
                        out_ext="tiff",
                        img_type="cam_gb",
                    )
        except Exception as e:
            print(e)
            break
        i += 1


# %%
