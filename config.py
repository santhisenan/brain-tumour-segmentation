import torch
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
epochs = 50
learning_rate = 0.001

# PATHS
output_path = Path("./outputs/")
