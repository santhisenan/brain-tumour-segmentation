import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

batch_size = 16
epochs = 5
learning_rate = 0.001

# PATHS
model_save_path = ""
