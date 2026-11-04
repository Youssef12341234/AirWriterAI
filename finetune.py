import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from torchvision import datasets, transforms
from model import CNN
from traces import lettre_vers_28x28

# ===== Réglages =====
DOSSIER = "dataset"
EPOCHS = 10          # nombre de tours
LR = 0.0001          # force des corrections (petite : on ajuste, on ne refait pas tout)
REPETITION = 5       # chaque exemple perso est répété 5 fois
NB_EMNIST = 3000     # exemples EMNIST mélangés pour ne pas oublier
BATCH = 64
MOYENNE, ECART = 0.1307, 0.3081

random.seed(0)
torch.manual_seed(0)

# ===== Les caractères et la répartition apprentissage / test =====
with open("classes_emnist.json") as f:
    classes = json.load(f)
with open("split.json") as f:
    split = json.load(f)


def charger_exemples(noms):
    # lit les exemples, les convertit en images 28x28 normalisées, et donne leurs étiquettes
    images, labels = [], []
    for nom in noms:
        with open(os.path.join(DOSSIER, nom)) as f:
            exemple = json.load(f)
        traits = [t for t in exemple["traits"] if len(t) > 0]
        images.append(lettre_vers_28x28(traits))
        labels.append(classes.index(exemple["label"]))
    x = torch.from_numpy(np.array(images)).float() / 255.0
    x = ((x - MOYENNE) / ECART).unsqueeze(1)          # forme (N, 1, 28, 28)
    return x, torch.tensor(labels)


x_app, y_app = charger_exemples(split["apprentissage"])
x_test, y_test = charger_exemples(split["test"])


class Perso(Dataset):
    # tes exemples, au même format que ceux d'EMNIST
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])


# ===== EMNIST : un échantillon à mélanger, et un échantillon pour contrôler l'oubli =====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.transpose(1, 2)),   # remet les images à l'endroit
    transforms.Normalize((MOYENNE,), (ECART,)),
])
emnist_train = datasets.EMNIST(root="data", split="balanced", train=True,
                               download=False, transform=transform)
emnist_test = datasets.EMNIST(root="data", split="balanced", train=False,
                              download=False, transform=transform)

indices_train = random.sample(range(len(emnist_train)), NB_EMNIST)
indices_test = random.sample(range(len(emnist_test)), 5000)

# le mélange : tes exemples (x5) + 3000 exemples EMNIST
donnees = ConcatDataset([Perso(x_app, y_app)] * REPETITION + [Subset(emnist_train, indices_train)])
loader = DataLoader(donnees, batch_size=BATCH, shuffle=True)
loader_emnist_test = DataLoader(Subset(emnist_test, indices_test), batch_size=1000)

# ===== Le modèle de départ =====
model = CNN(nb_classes=47)
model.load_state_dict(torch.load("modele_emnist.pt"))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def score_perso(avec_mode):
    # précision sur tes 72 exemples de test
    model.eval()
    with torch.no_grad():
        probas = torch.softmax(model(x_test), dim=1)
        if avec_mode:                                  # on autorise seulement la bonne famille
            masque = torch.zeros_like(probas)
            masque[y_test < 10, :10] = 1               # le vrai caractère est un chiffre : chiffres seulement
            masque[y_test >= 10, 10:36] = 1            # c'est une lettre : majuscules A-Z seulement
            probas = probas * masque
        predictions = probas.argmax(dim=1)
    return 100 * (predictions == y_test).float().mean().item()


def score_emnist():
    # contrôle d'oubli : précision sur un échantillon d'EMNIST
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader_emnist_test:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return 100 * correct / total


print("Epoch | perte  | test perso (bon mode) | test perso (mixte) | EMNIST")
print(f"  0   |   -    | {score_perso(True):.1f} %                | {score_perso(False):.1f} %             | {score_emnist():.1f} %   <- AVANT")

# ===== Fine-tuning =====
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_perte = 0
    for images, labels in loader:
        optimizer.zero_grad()
        perte = loss_fn(model(images), labels)
        perte.backward()
        optimizer.step()
        total_perte += perte.item()
    print(f" {epoch:2d}   | {total_perte / len(loader):.4f} | {score_perso(True):.1f} %                | "
          f"{score_perso(False):.1f} %             | {score_emnist():.1f} %")

# on sauvegarde la DERNIÈRE version (pas la meilleure sur le test)
torch.save(model.state_dict(), "modele_perso.pt")
print("Modèle sauvegardé : modele_perso.pt")