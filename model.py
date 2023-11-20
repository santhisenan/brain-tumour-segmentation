from config import device
import segmentation_models_pytorch as smp
import logging

LOGGER = logging.getLogger(__name__)


def get_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation="sigmoid",
    )
    # model = smp.UnetPlusPlus(encoder_name="resnet18", activation="sigmoid")

    model.to(device)
    LOGGER.info(f"Loaded model into device {device}")
    return model
