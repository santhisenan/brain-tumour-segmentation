import logging

import torch
import pandas as pd
import plotly.express as px

LOGGER = logging.getLogger(__name__)


def iou_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Intersection over Union for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()

    intersection = (predictions & labels).float().sum((1, 2))
    union = (predictions | labels).float().sum((1, 2))

    iou = (intersection + e) / (union + e)
    return iou


def dice_pytorch(predictions: torch.Tensor, labels: torch.Tensor, e: float = 1e-7):
    """Calculates Dice coefficient for a tensor of predictions"""
    predictions = torch.where(predictions > 0.5, 1, 0)
    labels = labels.byte()

    intersection = (predictions & labels).float().sum((1, 2))
    return ((2 * intersection) + e) / (
        predictions.float().sum((1, 2)) + labels.float().sum((1, 2)) + e
    )


def BCE_dice(output, target, alpha=0.01):
    bce = torch.nn.functional.binary_cross_entropy(output, target)
    soft_dice = 1 - dice_pytorch(output, target).mean()
    return bce + alpha * soft_dice


def plot_losses(history, output_dir):
    df = pd.DataFrame(history).drop("num_epochs_trained", axis=1)
    fig = px.line(df[["train_loss", "val_loss"]])
    fig.write_image(output_dir / "train_val_losses.png")


def plot_metrics(history, output_dir):
    df = pd.DataFrame(history).drop("num_epochs_trained", axis=1)

    fig = px.line(df[["val_IoU"]])
    fig.write_image(output_dir / "val_IoU.png")

    fig = px.line(df[["val_dice"]])
    fig.write_image(output_dir / "val_dice.png")


def save_results(history, output_dir):
    LOGGER.info(f"Saving plots to {output_dir}")
    plot_losses(history, output_dir)
    plot_metrics(history, output_dir)

    df = pd.DataFrame(history).drop("num_epochs_trained", axis=1)
    df.to_csv(output_dir / "losses.csv", index=False)
