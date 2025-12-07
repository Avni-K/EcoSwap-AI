"""
Vision Transformer (ViT) Image Classification Script
Uses vit-base-patch16-224-in21k pretrained model for image classification
"""

import json
import random
import warnings
import requests
import numpy as np
from pathlib import Path
from PIL import Image
import torch

# Suppress PIL warnings for corrupt EXIF data and palette images
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification, ViTConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
)
from tqdm import tqdm
import itertools


def send_update_to_discord(message):
    url = "https://discord.com/api/webhooks/1443837171323768873/oM4mmdCxFxX09OAL0Z3Akh3BPumVIixnahGMr7_HYD-WsKxmMaruGwQzk8_M6y3BPM76"
    data = {"content": message}
    requests.post(url, json=data)


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

send_update_to_discord("vit: init done")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
send_update_to_discord(f"ViT: Using device: {device}")


class ImageClassificationDataset(Dataset):
    """Dataset class for image classification based on subfolder structure"""

    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]

            # Open image and handle palette images with transparency
            image = Image.open(img_path)

            # Handle palette images with transparency (convert P -> RGBA -> RGB)
            if image.mode == "P":
                image = image.convert("RGBA").convert("RGB")
            elif image.mode == "RGBA":
                image = image.convert("RGB")
            elif image.mode != "RGB":
                # Convert any other mode to RGB
                image = image.convert("RGB")

            if self.transform:
                image = self.transform(image)

            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Error reading {self.image_paths[idx]}: {e}")
            raise e


class ViTClassificationModel(nn.Module):
    """Vision Transformer classification model"""

    def __init__(
        self,
        num_classes,
        model_name="google/vit-base-patch16-224-in21k",
        dropout_rate=0.1,
    ):
        super(ViTClassificationModel, self).__init__()

        # Load pretrained ViT model
        self.vit = ViTForImageClassification.from_pretrained(model_name)

        # Update dropout in ViT config
        self.vit.config.hidden_dropout_prob = dropout_rate
        self.vit.config.attention_probs_dropout_prob = dropout_rate

        # Replace classification head with custom one for our number of classes
        # Add dropout before classifier
        self.vit.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.vit.config.hidden_size, num_classes),
        )

    def forward(self, pixel_values):
        return self.vit(pixel_values=pixel_values)


def vit_collate_fn(batch):
    """Custom collate function for ViT that handles PIL Images"""
    images, labels = zip(*batch)
    # Keep images as list of PIL Images (processor will handle conversion)
    # Convert labels to tensor
    labels = torch.tensor(labels, dtype=torch.long)
    return list(images), labels


def load_dataset(images_dir="images", test_size=0.2, val_size=0.1):
    """Load dataset from subfolder structure"""
    images_path = Path(images_dir)
    if not images_path.exists():
        raise ValueError(f"Directory '{images_dir}' not found!")

    # Find all image files and their labels (subfolder names)
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".gif",
        ".JPG",
        ".JPEG",
        ".PNG",
    }
    image_paths = []
    labels = []

    # Get all subdirectories (classes)
    class_dirs = [d for d in images_path.iterdir() if d.is_dir()]
    class_names = sorted([d.name for d in class_dirs])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    print(f"Found {len(class_names)} classes: {class_names}")

    # Collect all images with their labels
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_idx = class_to_idx[class_name]

        for ext in image_extensions:
            for img_path in class_dir.glob(f"*{ext}"):
                image_paths.append(img_path)
                labels.append(class_idx)

    print(f"Total images: {len(image_paths)}")
    send_update_to_discord(
        f"ViT: Found {len(class_names)} classes, {len(image_paths)} total images"
    )

    # Split dataset
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=SEED, stratify=labels
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=SEED, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
        class_to_idx,
        idx_to_class,
    )


def train_epoch(model, train_loader, criterion, optimizer, device, processor):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        # Process images with ViT processor (images are PIL Images from dataset)
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)

    return epoch_loss, accuracy


def evaluate(model, data_loader, criterion, device, processor):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            # Process images with ViT processor (images are PIL Images from dataset)
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            loss = criterion(logits, labels)

            running_loss += loss.item()
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, accuracy, recall, f1, all_preds, all_labels


def train_model(
    hyperparams, train_data, val_data, test_data, class_to_idx, idx_to_class
):
    """Train model with given hyperparameters"""
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    num_classes = len(class_to_idx)

    # Load ViT processor
    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)

    # Transform from image_to_tensor.py (for data augmentation during training)
    # Note: ViT processor handles normalization, so we only apply augmentation
    transform_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
        ]
    )

    # Create datasets
    train_dataset = ImageClassificationDataset(
        X_train, y_train, class_to_idx, transform=transform_train
    )
    val_dataset = ImageClassificationDataset(
        X_val, y_val, class_to_idx, transform=transform_val
    )
    test_dataset = ImageClassificationDataset(
        X_test, y_test, class_to_idx, transform=transform_val
    )

    # Create data loaders with custom collate function for PIL Images
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        collate_fn=vit_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=False,
        collate_fn=vit_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=False,
        collate_fn=vit_collate_fn,
    )

    # Initialize model
    dropout_rate = hyperparams.get("dropout_rate", 0.1)
    model = ViTClassificationModel(
        num_classes, model_name, dropout_rate=dropout_rate
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparams["learning_rate"],
        weight_decay=hyperparams.get("weight_decay", 0),
    )

    # Training loop
    best_val_f1 = 0.0
    patience = hyperparams.get("patience", 5)
    patience_counter = 0

    train_history = []
    val_history = []

    num_epochs = hyperparams.get("num_epochs", 10)

    hp_str = f"lr={hyperparams['learning_rate']}, bs={hyperparams['batch_size']}, wd={hyperparams.get('weight_decay', 0)}, dropout={hyperparams.get('dropout_rate', 0.1)}"
    send_update_to_discord(f"ViT: Starting training - {hp_str}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, processor
        )

        # Validate
        val_loss, val_acc, val_recall, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device, processor
        )

        train_history.append({"loss": train_loss, "accuracy": train_acc})
        val_history.append(
            {"loss": val_loss, "accuracy": val_acc, "recall": val_recall, "f1": val_f1}
        )

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}"
        )

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                send_update_to_discord(
                    f"ViT: Early stopping at epoch {epoch + 1} - {hp_str}"
                )
                break

    # Test evaluation
    test_loss, test_acc, test_recall, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, processor
    )

    send_update_to_discord(
        f"ViT: Completed - {hp_str} | Test Acc: {test_acc:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}"
    )

    results = {
        "hyperparameters": hyperparams,
        "best_val_f1": best_val_f1,
        "test_metrics": {
            "loss": test_loss,
            "accuracy": test_acc,
            "recall": test_recall,
            "f1_score": test_f1,
        },
        "train_history": train_history,
        "val_history": val_history,
        "classification_report": classification_report(
            test_labels,
            test_preds,
            target_names=[idx_to_class[i] for i in range(num_classes)],
            output_dict=True,
            zero_division=0,
        ),
    }

    return results, model, processor, test_labels, test_preds

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, 
                annot=False,  # No annotations
                cmap="Blues",  # Blue color scheme
                xticklabels=class_names, 
                yticklabels=class_names,
                vmin=0,  # Set color scale from 0 to 1
                vmax=1,
                cbar_kws={'label': ''})
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Normalized")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    """Train a single ViT model with fixed hyperparameters"""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train ViT classification model")
    parser.add_argument(
        "--images_dir",
        type=str,
        default="images",
        help="Directory containing images (default: images)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vit_single_result.json",
        help="Output JSON file containing training metrics",
    )

    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    train_data, val_data, test_data, class_to_idx, idx_to_class = load_dataset(
        args.images_dir
    )
    with open("classes.json", "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print("Saved class index mapping to classes.json")
    #exit(0)
    # -------- FIXED HYPERPARAMETERS --------
    hyperparams = {
        "learning_rate": 0.00005,
        "batch_size": 64,
        "weight_decay": 0.0,
        "dropout_rate": 0.2,
        "num_epochs": 50,
        "patience": 5,
    }
    # ----------------------------------------

    print("\nTraining model with fixed hyperparameters:")
    print(hyperparams)

    # Train model
    results, trained_model, processor, y_true, y_pred = train_model(
        hyperparams, train_data, val_data, test_data, class_to_idx, idx_to_class
    )

    # Test evaluation is already in results
    test_labels = results['classification_report']
    # But we want y_true and y_pred
    #y_true = results['test_labels']  # update train_model to return this in results
    #y_pred = results['test_preds']

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names=[idx_to_class[i] for i in range(len(idx_to_class))])


    # Save training results to JSON
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {args.output}")

    # ---------------- SAVE TRAINED MODEL ----------------
    print("\nSaving trained model for later inference...")

    save_dir = Path("saved_model")
    save_dir.mkdir(exist_ok=True)

    num_classes = len(class_to_idx)

    trained_model.vit.config.num_labels = len(class_to_idx)
    trained_model.vit.config.label2id = class_to_idx
    trained_model.vit.config.id2label = idx_to_class

    # Save model
    trained_model.vit.save_pretrained(save_dir)
    #model_name = "google/vit-base-patch16-224-in21k"
    #processor = ViTImageProcessor.from_pretrained(model_name)

    # Re-create model with same config
    #num_classes = len(class_to_idx)
    #model = ViTClassificationModel(
    #    num_classes, model_name, dropout_rate=hyperparams["dropout_rate"]
    #)

    # Important: Load trained weights into the model object
    # (train_model returns results, not the model itself)
    # Instead, modify train_model to return the model:
    #   return results, model
    # Then update code here like:
    #
    # results, trained_model = train_model(...)
    #
    # For now, assume you updated train_model to also return `trained_model`

    # Save processor and model
    processor.save_pretrained(save_dir)

    print(f"Model saved to: {save_dir}")
    print("You can now load it using:")
    print('  model = ViTForImageClassification.from_pretrained("saved_model")')
    print('  processor = ViTImageProcessor.from_pretrained("saved_model")')


if __name__ == "__main__":
    main()
