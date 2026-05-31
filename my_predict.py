import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from my_model import build_model

# --------------------
# Config
# --------------------
TEST_CSV = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/test/_classes.csv"      # path to test csv
TEST_IMG_DIR = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/test"               # folder with images
MODEL_PATH = "best_model.pth"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = ["Fractura", "Metal", "Texto"]

# --------------------
# Dataset (same logic as train)
# --------------------
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, row["filename"]

# --------------------
# Transforms
# --------------------
test_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# DataLoader
# --------------------
test_dataset = TestDataset(
    csv_file=TEST_CSV,
    image_dir=TEST_IMG_DIR,
    transform=test_tfms
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)

# --------------------
# Load model
# --------------------
model = build_model(num_labels=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --------------------
# Inference
# --------------------

if __name__ == '__main__':

    results = []

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(DEVICE)

            logits = model(images)
            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).int()

            for i in range(len(filenames)):
                row = {
                    "filename": filenames[i],
                }

                for j, label in enumerate(LABELS):
                    row[f"{label}_prob"] = probs[i, j].item()
                    row[f"{label}_pred"] = preds[i, j].item()

                results.append(row)

    # --------------------
    # Save results
    # --------------------
    df_out = pd.DataFrame(results)
    df_out.to_csv("test_predictions.csv", index=False)

    print("✅ Prediction finished. Saved to test_predictions.csv")
