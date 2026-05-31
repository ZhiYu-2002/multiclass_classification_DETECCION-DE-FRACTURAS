
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_curve, auc
from my_dataset_one_class import WristXrayDataset
from my_model_one_class import build_model
import numpy as np
import torchvision

from torch.utils.tensorboard import SummaryWriter
import time

import wandb

wandb.init(
    project="wrist-fracture-binary",
    name="resnet18_weighted_bce",
    config={
        "model": "resnet18",
        "epochs": 20,
        "batch_size": 8,
        "learning_rate": 1e-4,
        "loss": "BCEWithLogitsLoss_weighted",
    }
)

config = wandb.config

# --------------------
# Config
# --------------------
TRAIN_CSV = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/train/_classes_oneclass.csv"
VAL_CSV   = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/valid/_classes_oneclass.csv"
TRAIN_DIR = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/train"
VAL_DIR   = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/valid"

BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# pos_weight = 0.4964
# pos_weight = torch.tensor([pos_weight], dtype=torch.float32)

# --------------------
# Unnormalize helper (for WandB image logging)
# --------------------
def unnormalize(img, mean, std):
    img = img.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return img

# --------------------
# Balanced Accuracy
# --------------------

# def logits_to_preds(logits, threshold=0.5):
#     probs = torch.sigmoid(logits)
#     return (probs >= threshold).float()

# def balanced_accuracy(y_true, y_pred):
#     y_true = y_true.view(-1)
#     y_pred = y_pred.view(-1)

#     TP = ((y_pred == 1) & (y_true == 1)).sum().float()
#     TN = ((y_pred == 0) & (y_true == 0)).sum().float()
#     FP = ((y_pred == 1) & (y_true == 0)).sum().float()
#     FN = ((y_pred == 0) & (y_true == 1)).sum().float()

#     recall_pos = TP / (TP + FN + 1e-8)
#     recall_neg = TN / (TN + FP + 1e-8)

#     return 0.5 * (recall_pos + recall_neg)


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
model = build_model(num_labels=1).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

best_val_loss = float("inf")

run_name = time.strftime("wrist_%Y%m%d_%H%M%S")
writer = SummaryWriter(log_dir=f"runs/{run_name}")

global_step = 0


# --------------------
# Validation Code (with balanced accuracy + recall_fracture)
# --------------------

# def validate(model, val_loader, criterion, device):
#     model.eval()
    
#     val_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():

#         for images, labels in val_loader:
#             images = images.to(device)
#             labels = labels.float().unsqueeze(1).to(device)

#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item()

#             probs = torch.sigmoid(outputs)
#             preds = (probs > 0.5).float()
#             # preds = logits_to_preds(outputs)

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())

#     val_loss /= len(val_loader)

#     # Convert to 1D arrays
#     all_labels = [int(x[0]) for x in all_labels]
#     all_preds = [int(x[0]) for x in all_preds]

#     # Balanced Accuracy
#     val_bal_acc = balanced_accuracy_score(all_labels, all_preds)

#     # Recall for fracture (positive class = 1)
#     val_recall_fracture = recall_score(all_labels, all_preds, pos_label=1)

#     wandb.log({
#     "val_loss": val_loss,
#     "val_balanced_accuracy": val_bal_acc,
#     "val_recall_fracture": recall_score(all_labels, all_preds, pos_label=1),
#     "val_specificity": recall_score(all_labels, all_preds, pos_label=0),
#     })


#     wandb.log({
#         "confusion_matrix": wandb.plot.confusion_matrix(
#             probs=None,
#             y_true=all_labels.cpu().numpy(),
#             preds=all_preds.cpu().numpy(),
#             class_names=["Normal", "Fracture"]
#         )
#     })


#     probs = torch.sigmoid(all_preds).cpu().numpy()
#     fpr, tpr, _ = roc_curve(all_labels.cpu().numpy(), probs)
#     roc_auc = auc(fpr, tpr)


#     wandb.log({
#         "roc_curve": wandb.plot.line_series(
#             xs=fpr,
#             ys=[tpr],
#             keys=["ROC"],
#             title="ROC Curve",
#             xname="False Positive Rate"
#         ),
#         "val_auc": roc_auc
#     })


#     images_to_log = images[:8]
#     preds_to_log = preds[:8]
#     labels_to_log = labels[:8]

#     wandb.log({
#         "examples": [
#             wandb.Image(
#                 img,
#                 caption=f"Pred: {p.item()} | True: {l.item()}"
#             )
#             for img, p, l in zip(images_to_log, preds_to_log, labels_to_log)
#         ]
#     })



#     return val_loss, val_bal_acc, val_recall_fracture


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

            wandb.log({"train_loss_batch": loss.item()}, step=global_step)

            global_step += 1

        train_loss /= len(train_loader)

        # ---- Validation ----
        # val_loss, val_bal_acc, val_recall_fracture = validate(
        #     model, val_loader, criterion, DEVICE
        # )

        model.eval()
        
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_logits = []

        with torch.no_grad():

            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.float().to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                # preds = logits_to_preds(outputs)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_logits.extend(outputs.cpu().numpy())

        val_loss /= len(val_loader)

        # Convert to 1D arrays
        all_labels = [int(x[0]) for x in all_labels]
        all_preds = [int(x[0]) for x in all_preds]

        all_logits = np.array(all_logits).reshape(-1)

        # Balanced Accuracy
        val_bal_acc = balanced_accuracy_score(all_labels, all_preds)

        # Recall for fracture (positive class = 1)
        val_recall_fracture = recall_score(all_labels, all_preds, pos_label=1)

        wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_balanced_accuracy": val_bal_acc,
        "val_recall_fracture": recall_score(all_labels, all_preds, pos_label=1),
        "val_specificity": recall_score(all_labels, all_preds, pos_label=0),
        })


        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=["Normal", "Fracture"]
            )
        })


        probs = 1 / (1 + np.exp(-all_logits))  # sigmoid manually
        fpr, tpr, _ = roc_curve(all_labels, probs)
        roc_auc = auc(fpr, tpr)


        wandb.log({
            "roc_curve": wandb.plot.line_series(
                xs=fpr,
                ys=[tpr],
                keys=["ROC"],
                title="ROC Curve",
                xname="False Positive Rate"
            ),
            "val_auc": roc_auc
        })


        # images_to_log = images[:8].cpu()
        # preds_to_log = preds[:8]
        # labels_to_log = labels[:8]

        # wandb.log({
        #     "examples": [
        #         wandb.Image(
        #             img,
        #             caption=f"Pred: {p.item()} | True: {l.item()}"
        #         )
        #         for img, p, l in zip(images_to_log, preds_to_log, labels_to_log)
        #     ]
        # })

        # --------------------
        # Log example predictions (FIXED VERSION)
        # --------------------
        # mean = [0.485, 0.485, 0.485]
        # std  = [0.229, 0.229, 0.229]

        # images_to_log = images[:8].cpu()
        # preds_to_log = preds[:8]
        # labels_to_log = labels[:8]

        # logged_images = []

        # for img, p, l in zip(images_to_log, preds_to_log, labels_to_log):

        #     # Unnormalize
        #     img = unnormalize(img, mean, std)

        #     # Clamp values to valid range
        #     img = img.clamp(0, 1)

        #     logged_images.append(
        #         wandb.Image(
        #             img,
        #             caption=f"Pred: {int(p.item())} | True: {int(l.item())}"
        #         )
        #     )

        # wandb.log({"examples": logged_images})

        # --------------------
        # Log example predictions WITH probability (CORRECT VERSION)
        # --------------------

        mean = [0.485, 0.485, 0.485]
        std  = [0.229, 0.229, 0.229]

        # Use last validation batch safely
        images_to_log = images[:8].detach().cpu()
        labels_to_log = labels[:8].detach().cpu()

        # recompute outputs safely
        outputs_to_log = model(images[:8].to(DEVICE)).detach().cpu()

        probs_to_log = torch.sigmoid(outputs_to_log).squeeze(1)
        preds_to_log = (probs_to_log > 0.5).long()

        logged_images = []

        for img, pred, label, prob in zip(
            images_to_log,
            preds_to_log,
            labels_to_log,
            probs_to_log
        ):

            img = unnormalize(img, mean, std)
            img = img.clamp(0, 1)

            pred_val = int(pred.item())
            label_val = int(label.item())
            prob_val = prob.item()

            # determine prediction type
            if pred_val == 1 and label_val == 1:
                result = "TP"
            elif pred_val == 1 and label_val == 0:
                result = "FP"
            elif pred_val == 0 and label_val == 1:
                result = "FN"
            else:
                result = "TN"

            caption = (
                f"Pred: {pred_val} | "
                f"True: {label_val} | "
                f"Prob: {prob_val:.4f} | "
                f"{result}"
            )

            logged_images.append(
                wandb.Image(img, caption=caption)
            )

        wandb.log({"examples": logged_images})




        # log epoch losses
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)
        writer.add_scalar("Balanced Acc/val_epoch", val_bal_acc, epoch)
        writer.add_scalar("Val Recall (Fracture)/val_epoch", val_recall_fracture, epoch)


        print(f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train={train_loss:.4f} "
            f"Val={val_loss:.4f}"
            f"Val Balanced Acc: {val_bal_acc:.4f}"
            f"Val Recall (Fracture): {val_recall_fracture:.4f}")

        # ---- Save best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_oneclass.pth")
            wandb.save("best_model_oneclass.pth")
            print("✅ Best model saved")

    writer.close()