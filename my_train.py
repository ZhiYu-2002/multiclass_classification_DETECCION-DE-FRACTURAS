# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import transforms

# from my_dataset import WristXrayDataset
# from my_model import build_model

# # --------------------
# # Config
# # --------------------
# CSV_FILE = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/train/_classes.csv"
# IMAGE_DIR = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/train"
# BATCH_SIZE = 16
# EPOCHS = 25
# LR = 1e-4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # --------------------
# # Transforms
# # --------------------
# train_tfms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # --------------------
# # Dataset & Loader
# # --------------------
# dataset = WristXrayDataset(
#     csv_file=CSV_FILE,
#     image_dir=IMAGE_DIR,
#     transform=train_tfms
# )

# loader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=4
# )

# # --------------------
# # Model
# # --------------------
# model = build_model(num_labels=3)
# model.to(DEVICE)

# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)

# # --------------------
# # Training loop
# # --------------------

# if __name__ == '__main__':

#     for epoch in range(EPOCHS):
#         model.train()
#         running_loss = 0

#         for images, labels in loader:
#             images = images.to(DEVICE)
#             labels = labels.to(DEVICE)

#             optimizer.zero_grad()
#             outputs = model(images)

#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         print(f"Epoch [{epoch+1}/{EPOCHS}] "
#             f"Loss: {running_loss:.4f}")

#     torch.save(model.state_dict(), "wrist_multilabel.pth")


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from my_dataset import WristXrayDataset
from my_model import build_model

from torch.utils.tensorboard import SummaryWriter
import time


# --------------------
# Config
# --------------------
TRAIN_CSV = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/train/_classes.csv"
VAL_CSV   = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/valid/_classes.csv"
TRAIN_DIR = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/train"
VAL_DIR   = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/valid"

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# Transforms
# --------------------
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485]*3, [0.229]*3)
])

# --------------------
# Datasets & Loaders
# --------------------
train_ds = WristXrayDataset(TRAIN_CSV, TRAIN_DIR, train_tfms)
val_ds   = WristXrayDataset(VAL_CSV,   VAL_DIR,   val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

# --------------------
# Model
# --------------------
model = build_model(num_labels=3).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")

run_name = time.strftime("wrist_%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=f"runs/{run_name}")

global_step = 0


# --------------------
# Training loop
# --------------------

if __name__ == '__main__':

    for epoch in range(EPOCHS):
        # ---- Train ----
        model.train()
        train_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # log batch loss
            writer.add_scalar(
                "Loss/train_batch",
                loss.item(),
                global_step
            )
            global_step += 1

        train_loss /= len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                loss = criterion(model(imgs), labels)
                val_loss += loss.item()

        val_loss   /= len(val_loader)

        # log epoch losses
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)

        # log learning rate
        writer.add_scalar(
            "LR",
            optimizer.param_groups[0]["lr"],
            epoch
        )

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train={train_loss:.4f} "
            f"Val={val_loss:.4f}")

        # ---- Save best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved")

    writer.close()