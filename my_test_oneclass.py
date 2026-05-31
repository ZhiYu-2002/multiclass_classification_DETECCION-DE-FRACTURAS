import torch
import torch.nn as nn
import numpy as np
import wandb
import pandas as pd
from my_dataset_one_class_test import WristXrayDataset

from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import os

import os
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    auc
)

# ======================================================
# GRADCAM OUTPUT
# ======================================================
GRADCAM_DIR = "gradcam_results"
os.makedirs(GRADCAM_DIR, exist_ok=True)

# ======================================================
# ITERGRATED GRADIENTS OUTPUT
# ======================================================
IG_DIR = "integrated_gradients_results"
os.makedirs(IG_DIR, exist_ok=True)


# ======================================================
# CONFIG
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MODEL_PATH = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/wandb/20260217/run-20260216_172150-amw69ey1/files/best_model_oneclass.pth"
TEST_DIR = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/test"
TEST_CSV = "C:/Users/yuzhi/Desktop/month 3/DETECCION DE FRACTURAS.v4i.multiclass/test/_classes_oneclass.csv"

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

# ======================================================
# INIT WANDB
# ======================================================
wandb.init(
    project="wrist-fracture-classification",
    name="test-evaluation",
)

# ======================================================
# TRANSFORMS (MUST MATCH TRAINING)
# ======================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================
# LOAD TEST DATA
# ======================================================
# test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

test_ds = WristXrayDataset(TEST_CSV, TEST_DIR, transform)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ======================================================
# LOAD MODEL
# ======================================================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 1)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

criterion = nn.BCEWithLogitsLoss()

ig = IntegratedGradients(model)

# ======================================================
# INIT GRADCAM
# ======================================================
target_layer = model.layer4[-1]

cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)

# ======================================================
# TEST LOOP
# ======================================================
test_loss = 0.0
all_labels = []
all_preds = []
all_probs = []
all_paths = []

example_images = None
example_probs = None
example_preds = None
example_labels = None

# gradcam needs gradients
# with torch.no_grad():
with torch.set_grad_enabled(True):
    
    for batch_idx, (images, labels, paths) in enumerate(test_loader):

        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        all_labels.extend(labels.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        # Save file paths
        all_paths.extend(paths)
        # start_idx = batch_idx * BATCH_SIZE
        # end_idx = start_idx + images.size(0)
        # all_paths.extend(
        #     test_ds.samples[start_idx:end_idx]
        # )

        # Save first batch for visualization
        if batch_idx == 0:
            example_images = images[:8].cpu()
            example_probs = probs[:8].cpu()
            example_preds = preds[:8].cpu()
            example_labels = labels[:8].cpu()

        for i in range(images.size(0)):

            input_tensor = images[i].unsqueeze(0)

            target = BinaryClassifierOutputTarget(int(preds[i].item()))

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[target]
            )[0]

            # Convert image for visualization
            img = input_tensor.squeeze().cpu().numpy().transpose(1,2,0)

            img = img * np.array(std) + np.array(mean)
            img = np.clip(img, 0, 1)

            visualization = show_cam_on_image(
                img.astype(np.float32),
                grayscale_cam,
                use_rgb=True
            )

            prob = probs[i].item()
            pred = int(preds[i].item())
            label = int(labels[i].item())

            if pred == 1 and label == 1:
                case = "TP"
            elif pred == 1 and label == 0:
                case = "FP"
            elif pred == 0 and label == 0:
                case = "TN"
            else:
                case = "FN"

            filename = f"{case}_prob{prob:.3f}_{os.path.basename(paths[i])}"

            cv2.imwrite(
                os.path.join(GRADCAM_DIR, filename),
                cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            )

        # ======================================================
        # INTEGRATED GRADIENTS (BATCH VERSION)
        # ======================================================

        # Create baseline for whole batch
        baseline = torch.zeros_like(images).to(DEVICE)

        # Compute IG ONCE for entire batch
        attributions = ig.attribute(
            images,
            baselines=baseline,
            target=0,
            n_steps=25   # faster than 50, still good quality
        )
            
        # Move to CPU
        attributions = attributions.detach().cpu().numpy()


        # ======================================================
        # SAVE IG RESULTS (loop only for visualization)
        # ======================================================
        for i in range(images.size(0)):

            # safer conversion
            prob = probs[i].item()
            pred = int(preds[i].item())
            label = int(labels[i].item() > 0.5)

            # OPTIONAL: only save mistakes (recommended)
            if pred == label:
                continue

            # if preds[i].item() == labels[i].item():
            #     continue

            # FIX: flip for normal class
            attr = attributions[i]
            
            if pred == 0:
                attr = -attr

            # Attribution map
            attribution = attributions[i].transpose(1,2,0)
            attribution = np.mean(np.abs(attribution), axis=2)
            attribution = attribution / (attribution.max() + 1e-8)

            # Original image (unnormalize)
            img = images[i].cpu().numpy().transpose(1,2,0)
            img = img * np.array(std) + np.array(mean)
            img = np.clip(img, 0, 1)

            plt.figure(figsize=(4,4))
            plt.imshow(img)
            plt.imshow(attribution, cmap="jet", alpha=0.5)
            plt.axis("off")

            plt.title(f"P:{prob:.2f} Pred:{pred} True:{label}")

            filename = os.path.basename(paths[i])
            save_name = f"IG_{prob:.3f}_pred{pred}_true{label}_{filename}"

            save_path = os.path.join(IG_DIR, save_name)

            plt.savefig(save_path, bbox_inches="tight")
            plt.close()

test_loss /= len(test_loader)

# ======================================================
# Convert arrays
# ======================================================
all_labels = np.array(all_labels).reshape(-1)
all_preds  = np.array(all_preds).reshape(-1)
all_probs  = np.array(all_probs).reshape(-1)

# ======================================================
# METRICS
# ======================================================
accuracy = accuracy_score(all_labels, all_preds)
balanced_acc = balanced_accuracy_score(all_labels, all_preds)
recall_fracture = recall_score(all_labels, all_preds, pos_label=1)
specificity = recall_score(all_labels, all_preds, pos_label=0)
precision = precision_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# ======================================================
# LOG METRICS TO WANDB
# ======================================================
wandb.log({
    "test_loss": test_loss,
    "test_accuracy": accuracy,
    "test_balanced_accuracy": balanced_acc,
    "test_recall_fracture": recall_fracture,
    "test_specificity": specificity,
    "test_precision": precision,
    "test_f1": f1,
    "test_auc": roc_auc,
})

# ======================================================
# CONFUSION MATRIX
# ======================================================
wandb.log({
    "test_confusion_matrix":
        wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            class_names=["Normal", "Fracture"]
        )
})

# ======================================================
# ROC CURVE
# ======================================================
wandb.log({
    "test_roc_curve":
        wandb.plot.line_series(
            xs=fpr,
            ys=[tpr],
            keys=["ROC"],
            title="Test ROC Curve",
            xname="False Positive Rate"
        )
})

# ======================================================
# LOG EXAMPLE IMAGES WITH PROBABILITY
# ======================================================
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

def unnormalize(img, mean, std):
    mean = torch.tensor(mean).view(3,1,1)
    std = torch.tensor(std).view(3,1,1)
    return img * std + mean

example_probs = example_probs.squeeze(1)
example_preds = example_preds.squeeze(1)
example_labels = example_labels.squeeze(1)

logged_images = []

for img, prob, pred, label in zip(
    example_images,
    example_probs,
    example_preds,
    example_labels
):

    img = unnormalize(img, mean, std)
    img = img.clamp(0,1)

    logged_images.append(
        wandb.Image(
            img,
            caption=(
                f"Prob: {prob.item():.3f} | "
                f"Pred: {int(pred.item())} | "
                f"True: {int(label.item())}"
            )
        )
    )

wandb.log({"test_examples": logged_images})


# ======================================================
# SAVE GRADCAM
# ======================================================

wandb.log({"gradcam_example": wandb.Image(visualization)})

# ======================================================
# SAVE PREDICTIONS TO CSV
# ======================================================
results_df = pd.DataFrame({
    "label": all_labels,
    "prediction": all_preds,
    "probability": all_probs,
})

results_df.to_csv("test_predictions.csv", index=False)
wandb.save("test_predictions.csv")

print("\n===== TEST COMPLETE =====")
print(f"AUC: {roc_auc:.4f}")
print("Results logged to W&B.")