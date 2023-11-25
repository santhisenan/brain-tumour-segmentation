import torch
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 32
epochs = 50
learning_rate = 0.001
model_name = "unet"
encoder_name = "resnet18"
pretrained = False
loss = "BCE_Dice"
tag = f"{model_name}-{encoder_name}-B{batch_size}-E{epochs}-LR{learning_rate}-PTM{int(pretrained)}-{loss}"

# PATHS
output_path = Path("./outputs/")
