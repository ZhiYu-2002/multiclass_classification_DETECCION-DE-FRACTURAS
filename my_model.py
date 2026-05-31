import torch.nn as nn
from torchvision import models

def build_model(num_labels=3):
    model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_labels
    )

    return model
