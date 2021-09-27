# %% [markdown]
# # UNICEF
# ## School detection by high-resolution satellite image classification

# %% [markdown]
# ## Imports and data loading

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
)
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Modify sys path to import top-level modules
import sys
import itertools

sys.path.append("..")
from utils.utils import (
    delete_data_folder_structure,
    show_image,
    parse_class_report,
    split_folder_of_images,
    load_trained_model,
    images_to_probs,
)
from models.mobilenetv2_model import MobileNetV2Model
from models.mobilenetv2_antialias_model import MobileNetV2ModelAntiAlias
from models.resnet_antialias_model import ResNetModelAntiAlias
from models.resnet_model import ResNetModel

# Fix random seed for reproducibility
seed = 9102
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Global device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using GPU for acceleration") if torch.cuda.is_available() else print(
    "Using CPU for computation"
)

# Image folder path (zoom level 18)
base_path = Path("../datasets_2")
shuffle_all_images = False
pretrain_model = ["resnet18", "resnet18_aa", "mobilenetv2", "mobilenetv2_aa"][1]

# Output files
saved_model_filename = f"school_classifier_model_{pretrain_model}.pt"
labels_mapping_filename = f"school_classifier_labels_{pretrain_model}.pkl"

# Data split
train_prop = 0.8  # (taken from ALL the data)
valid_prop = 0.2  # (taken only from TRAIN)

# Hyperparameters
batch_size = 16
lr = 0.0001
epochs = 25

# Tensorboard: writer will output to ./runs/ directory by default
comment = f"_T{round(train_prop*100)}_V{round(valid_prop*100)}_BS{batch_size}_LR{round(lr*10000)}_{pretrain_model}"
writer = SummaryWriter(comment=comment)

# %% [markdown]
# ### Split images in train, test and validation folders
#

# %%
# Split files in folders
if shuffle_all_images:
    delete_data_folder_structure(base_path=base_path)
    split_folder_of_images(
        base_path=base_path, train_prop=train_prop, valid_prop=valid_prop, seed=seed
    )


# %% [markdown]
# ### Data Augmentations
#
# Resize the image to a smaller one and normalize (substract the mean and divide by the standar deviation)
#
# Augmentations:
#
# - Random horizontal flip
# - Random vertical flip
# - Random slight rotation between 1 to 179 degrees
# - Random scale between 1.01 and 1.20
# - Random brightnes between 0.8 and 1.2

# %%
image_size = (224, 224)

# Using the mean and std of Imagenet for transformations
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(1, 179)),
        transforms.RandomAffine(degrees=0, scale=(1.01, 1.20)),
        transforms.ColorJitter(brightness=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds),
    ]
)

# %%
# Check transformations
# from PIL import Image

# img = Image.open(Path(base_path / "all" / "school" / "17-36857-64877.jpg"))
# img = Image.open(Path(base_path / "all" / "school" / "18-73792-129585.jpg"))

# try_transform = transforms.Compose(
#     [
#         transforms.Resize(image_size),
#         transforms.RandomAffine(degrees=0, scale=(1, 1.2)),
#         transforms.ColorJitter(brightness=(0.1, 1.5)),
#         transforms.ToTensor(),
#         transforms.Normalize(means, stds),
#     ]
# )

# show_image(try_transform(img), title="", means=means, stds=stds)


# %% [markdown]
# ### Define train, validation and test sets

# %%
train = ImageFolder(root=Path(base_path / "train"), transform=train_transform)
validation = ImageFolder(root=Path(base_path / "validation"), transform=test_transform)
test = ImageFolder(root=Path(base_path / "test"), transform=test_transform)

print(f"Dataset classes: {train.classes}")
print(
    f"Training images: {len(train)} | Validation images: {len(validation)} | Testing images: {len(test)}"
)

# %% [markdown]
# ### Display some randomly sampled images

# %%
# Get a batch of training data
example_images_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
example_images, classes = next(iter(example_images_loader))
out = torchvision.utils.make_grid(example_images)
show_image(
    out, title="Grid random samples of images", means=means, stds=stds,
)


# %% [markdown]
# ## Training functions

# %%
def plot_classes_preds(model, images, labels, classes, rows=8):
    """
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    """
    preds, probs = images_to_probs(model, images)
    nimg = len(preds)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(20, 20))
    for idx in np.arange(nimg):
        image = images[idx].detach().cpu()
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array(means)
        std = np.array(stds)
        image = ((std * image + mean) * 255).astype("uint8")
        ax = fig.add_subplot(rows, nimg / rows, idx + 1)
        plt.imshow(image)
        pred = classes[preds[idx]].upper().replace("_", "-")
        obs = classes[labels[idx]].upper().replace("_", "-")
        ax.set_title(
            f"PRED: {pred} ({round(probs[idx] * 100.0, 2)}%)\nOBS: {obs}",
            color=("green" if preds[idx] == labels[idx] else "red"),
        )
    fig.tight_layout()
    return fig


def train_epoch(training_model, loader, criterion, optim):
    training_model.train()
    epoch_loss = 0

    all_labels = []
    all_predictions = []
    for images, labels in loader:
        all_labels.extend(labels.numpy())

        optim.zero_grad()

        predictions = training_model(images.to(device))
        all_predictions.extend(torch.argmax(predictions, dim=1).cpu().numpy())

        labels = labels.long()

        loss = criterion(predictions, labels.to(device))
        # loss = criterion(torch.max(F.softmax(predictions, dim=1), dim=1)[0], labels.to(device))

        loss.backward()
        optim.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader), accuracy_score(all_labels, all_predictions) * 100


def validation_epoch(val_model, loader, criterion):
    val_model.eval()
    val_loss = 0

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            predictions = val_model(images.to(device))
            labels = labels.long()
            loss = criterion(predictions, labels.to(device))
            val_loss += loss.item()
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.topk(1, dim=1)[1].cpu().numpy().tolist())

    accuracy = accuracy_score(all_targets, all_predictions)

    return val_loss / len(loader), accuracy * 100


def train_model(
    target_model, number_epochs, criterion, optim, train_iterator, valid_iterator
):
    # Stats trackers
    train_history = []
    valid_history = []
    accuracy_history = []

    begin_time = time.time()
    best_val_acc = 0.0

    for epoch in range(number_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            target_model, train_iterator, criterion, optim
        )
        train_history.append(train_loss)
        print(
            "Training epoch {} | Loss {:.6f} | Accuracy {:.2f}% | Time {:.2f} seconds".format(
                epoch + 1, train_loss, train_acc, time.time() - start_time
            )
        )

        start_time = time.time()
        val_loss, acc = validation_epoch(target_model, valid_iterator, criterion)
        valid_history.append(val_loss)
        accuracy_history.append(acc)
        print(
            "Validation epoch {} | Loss {:.6f} | Accuracy {:.2f}% | Time {:.2f} seconds".format(
                epoch + 1, val_loss, acc, time.time() - start_time
            )
        )

        # Tensorboard
        writer.add_scalars(
            "Learning curves/Loss",
            {"train": train_loss, "validation": val_loss},
            epoch + 1,
        )
        writer.add_scalars(
            "Learning curves/Accuracy",
            {"train": train_acc, "validation": acc},
            epoch + 1,
        )

        # Checkpoint model
        if acc > best_val_acc:
            torch.save(model.state_dict(), f"trained_models_output/{saved_model_filename}")
            best_val_acc = acc
            print("* Found new best accuracy")

    print(
        "Total time for {} epochs: {:.2f} minutes".format(
            number_epochs, (time.time() - begin_time) / 60
        )
    )

    return train_history, valid_history, accuracy_history


def test_model(trained_model, test_iterator):
    test_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_iterator:
            predictions = trained_model(images.to(device))
            test_labels.extend(labels.cpu().numpy())
            all_predictions.extend(
                predictions.topk(1, dim=1)[1].squeeze().cpu().numpy()
            )

    return test_labels, all_predictions


def plot_stats(x_axis, train_loss, valid_loss, valid_acc):
    # Set matplotlib default plot size
    plt.rcParams["figure.figsize"] = [7, 5]

    # Loss
    plt.title("Train and Validation Loss")
    plt.plot(x_axis, train_loss, label="Train Loss")
    plt.plot(x_axis, valid_loss, label="Validation Loss")
    plt.legend()
    plt.show()

    # Accuracy
    plt.title("Validation Accuracy")
    plt.plot(x_axis, valid_acc)
    plt.show()


def plot_confusion_matrix(
    cm, used_labels, target_names, title="Confusion matrix", cmap="Blues"
):
    accuracy = np.trace(cm) / np.sum(cm).astype("float")
    misclass = 1 - accuracy
    cmap = plt.get_cmap(cmap)

    plt.figure(figsize=(15, 10))
    tick_marks = np.arange(len(used_labels))
    names = [target_names[idx] for idx in used_labels]
    plt.xticks(tick_marks, names, rotation=90)
    plt.yticks(tick_marks, names)
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()

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


# %% [markdown]
# ## Create model and loss function

# %%
# Define model hyperparameters
number_classes = len(train.classes)
pretrained = True
fs = 4

if pretrain_model == "resnet18_aa":
    model = ResNetModelAntiAlias(
        n_classes=number_classes, pretrained=pretrained, filter_size=fs
    )
elif pretrain_model == "mobilenetv2_aa":
    model = MobileNetV2ModelAntiAlias(
        n_classes=number_classes, pretrained=pretrained, filter_size=fs
    )
elif pretrain_model == "resnet18":
    model = ResNetModel(n_classes=number_classes, pretrained=pretrained)
else:
    model = MobileNetV2Model(n_classes=number_classes, pretrained=pretrained)

model.to(device)
print(model)
print(f"Number of trainable parameters {model.summary()}")

# Use BinaryCrossEntropy
# loss_function = nn.BCEWithLogitsLoss().to(device)
# loss_function = nn.BCELoss().to(device)
loss_function = nn.CrossEntropyLoss().to(device)

# %% [markdown]
# # Training hyperparameters and main loop

# %%
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(
    validation, batch_size=batch_size, shuffle=False, pin_memory=True
)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)

# %%
# Training
train_loss_history = []
valid_loss_history = []
accuracy_history = []
all_epochs = 0

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train the model and update stats
print(
    f"Start training pretrained model {pretrain_model} with {lr} learning rate, batch size of {batch_size} and {epochs} epochs"
)
train_losses, valid_losses, accuracies = train_model(
    model, epochs, loss_function, optimizer, train_loader, val_loader
)
train_loss_history.extend(train_losses)
valid_loss_history.extend(valid_losses)
accuracy_history.extend(accuracies)
all_epochs += epochs

# Tensorboard
writer.flush()

# %% [markdown]
# ### Learning curves

# %%
plot_stats(range(all_epochs), train_loss_history, valid_loss_history, accuracy_history)

# %% [markdown]
# ## Test set

# %% [markdown]
# ### Load best saved model from file

# %%
model = load_trained_model(pretrain_model)
model.to(device)
model.eval()

# %% [markdown]
# ### Save mapping from network outputs to human readable labels

# %%
import pickle

labels = {idx: value for idx, value in enumerate(train.classes)}
with open(f"trained_models_output/{labels_mapping_filename}", "wb+") as f:
    pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)


# %% [markdown]
# ### Predictions

# %%
all_predictions = []
all_test_images = []
all_true_labels = []
all_used_labels = set()

with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        images, labels = data
        labels = labels.numpy()
        all_true_labels.extend(labels)
        all_used_labels.update(labels)
        all_test_images.extend(images)
        images = images.to(device)
        current_preds = model(images)
        all_predictions.extend(F.softmax(current_preds.detach().cpu(), dim=1))

        # Tensorboard
        try:
            writer.add_figure(
                f"Test/Batch_prediction_{i+1}",
                plot_classes_preds(model, images, labels, test.classes),
            )
        except Exception:
            print(f"Couldn't add figures to Tensorboard from batch {i+1}")

# Tensorboard
writer.flush()

# Get all predictions and probabilities
all_predictions = torch.stack(all_predictions)
probabilities = torch.max(all_predictions, dim=1)[0]
votes = torch.argmax(all_predictions, dim=1).numpy()
labels = np.array(train.classes)


# %%
# Plot confusion matrix of results
conf_mat = confusion_matrix(all_true_labels, votes)
plot_confusion_matrix(
    cm=conf_mat,
    used_labels=all_used_labels,
    target_names=labels,
    title="Confusion matrix",
)
print(classification_report(all_true_labels, votes, target_names=labels, digits=3))

# Tensorboard
classif_report_dict = classification_report(
    all_true_labels, votes, target_names=labels, digits=3, output_dict=True
)
writer.add_text(
    f"Test_classif_report/all", parse_class_report(classif_report_dict, "not_school")
)
writer.add_text(
    f"Test_classif_report/all", parse_class_report(classif_report_dict, "school")
)
writer.add_text(
    f"Test_classif_report/all",
    f"accuracy: {round(classif_report_dict.get('accuracy'), 3)}",
)


# %% [markdown]
# ### ROC curve

# %%
fpr, tpr, thresholds = roc_curve(
    np.array(all_true_labels), probabilities.numpy(), pos_label=0
)
roc_auc = auc(x=fpr, y=tpr)

plt.figure()
lw = 2
plt.plot(
    fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# ### Check subsets of test samples

# %% [markdown]
# #### Urban

# %%
# Load urban images
test_urban = ImageFolder(
    root=Path(base_path / "test_subselect" / "urban"), transform=test_transform
)
test_urban_loader = DataLoader(
    test_urban, batch_size=batch_size, shuffle=False, pin_memory=True
)

example_urban_images, classes = next(iter(test_urban_loader))
out_urban = torchvision.utils.make_grid(example_urban_images)
show_image(
    out_urban, title="Urban samples of images", means=means, stds=stds,
)

# %%
all_predictions_urban = []
all_test_images_urban = []
all_true_labels_urban = []
all_used_labels_urban = set()

with torch.no_grad():
    for i, data in enumerate(test_urban_loader, 0):
        images, labels = data
        labels = labels.numpy()
        all_true_labels_urban.extend(labels)
        all_used_labels_urban.update(labels)
        all_test_images_urban.extend(images)
        images = images.to(device)
        current_preds = model(images)
        all_predictions_urban.extend(F.softmax(current_preds.detach().cpu(), dim=1))

        # Tensorboard
#         try:
#             writer.add_figure(
#                 f"Test_urban/Batch_prediction_{i+1}",
#                 plot_classes_preds(model, images, labels, test.classes, 4),
#             )
#         except Exception:
#             print(f"Couldn't add figures to Tensorboard from batch {i+1}")

# # Tensorboard
# writer.flush()

# Get predictions and probabilities
all_predictions_urban = torch.stack(all_predictions_urban)
probs_urban = torch.max(all_predictions_urban, dim=1)[0]
votes_urban = torch.argmax(all_predictions_urban, dim=1).numpy()
labels = np.array(train.classes)

# %%
# Plot confusion matrix of results
conf_mat = confusion_matrix(all_true_labels_urban, votes_urban)
plot_confusion_matrix(
    cm=conf_mat,
    used_labels=all_used_labels,
    target_names=labels,
    title="Confusion matrix - urban subset",
)
print(
    classification_report(
        all_true_labels_urban, votes_urban, target_names=labels, digits=3
    )
)

# Tensorboard
classif_report_dict = classification_report(
    all_true_labels_urban, votes_urban, target_names=labels, digits=3, output_dict=True
)
writer.add_text(
    f"Test_classif_report/urban", parse_class_report(classif_report_dict, "not_school")
)
writer.add_text(
    f"Test_classif_report/urban", parse_class_report(classif_report_dict, "school")
)
writer.add_text(
    f"Test_classif_report/urban",
    f"accuracy: {round(classif_report_dict.get('accuracy'), 3)}",
)


# %% [markdown]
# #### Not-Urban

# %%
# Load not-urban images
test_not_urban = ImageFolder(
    root=Path(base_path / "test_subselect" / "not_urban"), transform=test_transform
)
test_not_urban_loader = DataLoader(
    test_not_urban, batch_size=batch_size, shuffle=False, pin_memory=True
)

example_not_urban_images, classes = next(iter(test_not_urban_loader))
out_not_urban = torchvision.utils.make_grid(example_not_urban_images)
show_image(
    out_not_urban, title="Not-urban samples of images", means=means, stds=stds,
)

# %%
all_predictions_not_urban = []
all_test_images_not_urban = []
all_true_labels_not_urban = []
all_used_labels_not_urban = set()

with torch.no_grad():
    for i, data in enumerate(test_not_urban_loader, 0):
        images, labels = data
        labels = labels.numpy()
        all_true_labels_not_urban.extend(labels)
        all_used_labels_not_urban.update(labels)
        all_test_images_not_urban.extend(images)
        images = images.to(device)
        current_preds = model(images)
        all_predictions_not_urban.extend(F.softmax(current_preds.detach().cpu(), dim=1))

        # Tensorboard
        try:
            writer.add_figure(
                f"Test_not_urban/Batch_prediction_{i+1}",
                plot_classes_preds(model, images, labels, test.classes, 4),
            )
        except Exception:
            print(f"Couldn't add figures to Tensorboard from batch {i+1}")

# Tensorboard
writer.flush()

# Get predictions and probabilities
all_predictions_not_urban = torch.stack(all_predictions_not_urban)
probs_not_urban = torch.max(all_predictions_not_urban, dim=1)[0]
votes_not_urban = torch.argmax(all_predictions_not_urban, dim=1).numpy()
labels = np.array(train.classes)

# %%
# Plot confusion matrix of results
conf_mat = confusion_matrix(all_true_labels_not_urban, votes_not_urban,)
plot_confusion_matrix(
    cm=conf_mat,
    used_labels=all_used_labels,
    target_names=labels,
    title="Confusion matrix - not-urban subset",
)
print(
    classification_report(
        all_true_labels_not_urban, votes_not_urban, target_names=labels, digits=3
    )
)

# Tensorboard
classif_report_dict = classification_report(
    all_true_labels_not_urban,
    votes_not_urban,
    target_names=labels,
    digits=3,
    output_dict=True,
)
writer.add_text(
    f"Test_classif_report/not_urban",
    parse_class_report(classif_report_dict, "not_school"),
)
writer.add_text(
    f"Test_classif_report/not_urban", parse_class_report(classif_report_dict, "school")
)
writer.add_text(
    f"Test_classif_report/not_urban",
    f"accuracy: {round(classif_report_dict.get('accuracy'), 3)}",
)

# %%
# Tensorboard
writer.close()


# %%
