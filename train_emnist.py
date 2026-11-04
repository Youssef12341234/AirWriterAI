import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN          # même modèle que pour MNIST

# ----- Réglages -----
EPOCHS = 5      # nombre de tours complets sur les 112 800 images
BATCH = 128     # nombre d'images montrées en une fois
LR = 0.001      # force des corrections

# ----- Préparer les images -----
transform = transforms.Compose([
    transforms.ToTensor(),                              # image -> nombres
    transforms.Lambda(lambda x: x.transpose(1, 2)),     # remet l'image à l'endroit
    transforms.Normalize((0.1307,), (0.3081,)),         # même normalisation qu'avant
])

train_data = datasets.EMNIST(root="data", split="balanced", train=True,
                             download=False, transform=transform)
test_data = datasets.EMNIST(root="data", split="balanced", train=False,
                            download=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# ----- Sauvegarder la liste des caractères (0..9, A..Z, a, b, d...) -----
with open("classes_emnist.json", "w") as f:
    json.dump(train_data.classes, f)

# ----- Préparer l'élève, la note d'erreur et le prof -----
model = CNN(nb_classes=47)                      # 47 réponses possibles
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def evaluer():
    # note à l'examen : pourcentage de bonnes réponses sur les 18 800 images de test
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
    return 100 * correct / len(test_data)


# ----- Entraînement -----
meilleure = 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    debut = time.time()
    total_perte = 0

    for images, labels in train_loader:
        optimizer.zero_grad()                   # 1. effacer les anciennes corrections
        perte = loss_fn(model(images), labels)  # 2-3. le modèle répond, on mesure l'erreur
        perte.backward()                        # 4. chercher d'où vient l'erreur
        optimizer.step()                        # 5. corriger le modèle
        total_perte += perte.item()

    precision = evaluer()
    print(f"Epoch {epoch}/{EPOCHS} | perte : {total_perte / len(train_loader):.4f} | "
          f"précision test : {precision:.2f}% | {time.time() - debut:.0f}s")

    if precision > meilleure:                   # on garde la meilleure version
        meilleure = precision
        torch.save(model.state_dict(), "modele_emnist.pt")
        print("   -> meilleur modèle sauvegardé")

print(f"Terminé. Meilleure précision : {meilleure:.2f}%")