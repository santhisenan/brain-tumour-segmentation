from glob import glob

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
import albumentations as A

import cv2
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from utils import get_file_row


class MriDataset(Dataset):
    def __init__(self, df, transform=None, mean=0.5, std=0.25):
        super(MriDataset, self).__init__()
        self.df = df
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, raw=False):
        row = self.df.iloc[idx]
        img = cv2.imread(row["image_filename"], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(row["mask_filename"], cv2.IMREAD_GRAYSCALE)
        if raw:
            return img, mask

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        img = T.functional.to_tensor(img)
        mask = mask // 255
        mask = torch.Tensor(mask)
        return img, mask


def get_dataloaders(batch_size):
    files_dir = "/data/santhise001/datasets/lgg-segmentation/kaggle_3m"
    file_paths = glob(f"{files_dir}/*/*[0-9].tif")

    csv_path = "/data/santhise001/datasets/lgg-segmentation/kaggle_3m/data.csv"
    df = pd.read_csv(csv_path)

    imputer = SimpleImputer(strategy="most_frequent")

    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    filenames_df = pd.DataFrame(
        (get_file_row(filename) for filename in file_paths),
        columns=["Patient", "image_filename", "mask_filename"],
    )

    df = pd.merge(df, filenames_df, on="Patient")

    train_df, test_df = train_test_split(df, test_size=0.3)
    test_df, valid_df = train_test_split(test_df, test_size=0.1)

    transform = A.Compose(
        [
            A.RandomCrop(width=256, height=256),
            A.RandomBrightnessContrast(p=0.2),
            A.AdvancedBlur(p=0.5),
        ]
    )

    # transform = v2.Compose([])

    train_dataset = MriDataset(train_df, transform)
    valid_dataset = MriDataset(valid_df)
    test_dataset = MriDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader
