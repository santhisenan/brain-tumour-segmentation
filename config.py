import torch

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

batch_size = 32
epochs = 50
learning_rate = 0.001

# PATHS
model_save_path = "./outputs/model.pth"
