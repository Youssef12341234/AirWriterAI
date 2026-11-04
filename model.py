import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, nb_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, nb_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = CNN()
    x = torch.randn(8, 1, 28, 28)
    out = model(x)
    print("Entrée :", tuple(x.shape))
    print("Sortie :", tuple(out.shape))
    print("Paramètres :", sum(p.numel() for p in model.parameters()))