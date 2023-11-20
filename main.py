import time
import logging

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchmetrics.classification import Dice

from data import get_dataloaders
from optimization import EarlyStopping
from loss_metrics import BCE_dice, iou_pytorch, dice_pytorch, save_results
from config import (
    device,
    batch_size,
    epochs,
    learning_rate,
    output_path,
    model_name,
    encoder_name,
)
from model import get_model

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

train_loader, valid_loader, test_loader = get_dataloaders(batch_size)
model = get_model(model_name=model_name, encoder_name=encoder_name)


def training_loop(
    epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_IoU": [],
        "val_dice": [],
        "num_epochs_trained": 0,
    }
    early_stopping = EarlyStopping(patience=7)

    with logging_redirect_tqdm():
        for epoch in range(1, epochs + 1):
            running_loss = 0
            model.train()
            for i, data in enumerate(tqdm(train_loader)):
                img, mask = data
                img, mask = img.to(device), mask.to(device)
                # img, mask = img.to(device), mask.int().to(device)

                predictions = model(img)
                predictions = predictions.squeeze(1)
                loss = loss_fn(predictions, mask)
                # loss.requires_grad = True

                running_loss += loss.item() * img.size(0)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            model.eval()
            with torch.no_grad():
                running_IoU = 0
                running_dice = 0
                running_valid_loss = 0
                for i, data in enumerate(valid_loader):
                    img, mask = data
                    img, mask = img.to(device), mask.to(device)
                    # img, mask = img.to(device), mask.int().to(device)

                    predictions = model(img)
                    predictions = predictions.squeeze(1)
                    running_dice += dice_pytorch(predictions, mask).sum().item()
                    running_IoU += iou_pytorch(predictions, mask).sum().item()
                    loss = loss_fn(predictions, mask)
                    running_valid_loss += loss.item() * img.size(0)
            train_loss = running_loss / len(train_loader.dataset)
            val_loss = running_valid_loss / len(valid_loader.dataset)
            val_dice = running_dice / len(valid_loader.dataset)
            val_IoU = running_IoU / len(valid_loader.dataset)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_IoU"].append(val_IoU)
            history["val_dice"].append(val_dice)
            history["num_epochs_trained"] = epoch
            LOGGER.info(
                f"Epoch: {epoch}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss} | Validation Mean IoU: {val_IoU} "
                f"| Validation Dice coefficient: {val_dice}"
            )

            lr_scheduler.step(val_loss)
            if early_stopping(val_loss, model):
                early_stopping.load_weights(model)
                LOGGER.info(f"Stopping after epoch {epoch}")
                break
        model.eval()

    return history


loss_fn = BCE_dice
# loss_fn = Dice().to(device)
# loss_fn.requires_grad = True

optimizer = Adam(model.parameters(), lr=learning_rate)
lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=2, factor=0.2)

history = training_loop(
    epochs, model, train_loader, valid_loader, optimizer, loss_fn, lr_scheduler
)

model_export = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "history": history,
    "hyperparams": {
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
    },
    "loss_function": "BCE Dice",
}

model_output_dir = output_path / f"model-{time.strftime('%Y%m%d-%H%M%S')}"
model_output_dir.mkdir()
torch.save(model_export, model_output_dir / "model.pth")
LOGGER.info(f"Saved checkpoint to {model_output_dir}")

save_results(history, model_output_dir)

LOGGER.info("Finished")
