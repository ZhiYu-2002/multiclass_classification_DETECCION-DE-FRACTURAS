import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class WristXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.label_cols = ["Fractura"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.image_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        labels = torch.tensor(
            row[self.label_cols].values.astype("float32")
        )

        if self.transform:
            image = self.transform(image)

        return image, labels
