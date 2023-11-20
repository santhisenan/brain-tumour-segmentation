from config import device
import segmentation_models_pytorch as smp
import logging

LOGGER = logging.getLogger(__name__)


def get_model(model_name="unet", encoder_name="resnet18"):
    model = None

    if model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            activation="sigmoid",
        )
    elif model_name == "unet++":
        model = smp.UnetPlusPlus(encoder_name=encoder_name, activation="sigmoid")
    elif model_name == "fpn":
        model = smp.FPN(encoder_name=encoder_name, activation="sigmoid")
    elif model_name == "deeplabv3+":
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, activation="sigmoid")
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    if model:
        model.to(device)
        LOGGER.info(f"Loaded model into device {device}")
        return model
