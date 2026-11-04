import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN

# Même préparation des images que pour l'entraînement
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
test_data = datasets.MNIST(root="data", train=False, download=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000)

# Charger le modèle et ce qu'il a appris
model = CNN()
model.load_state_dict(torch.load("modele_mnist.pt"))
model.eval()    # mode examen

# Tableau 10x10 : lignes = vrai chiffre, colonnes = réponse du modèle
matrice = np.zeros((10, 10), dtype=int)
with torch.no_grad():
    for images, labels in test_loader:
        predictions = model(images).argmax(dim=1)
        for vrai, pred in zip(labels.tolist(), predictions.tolist()):
            matrice[vrai][pred] += 1

# Score pour chaque chiffre
print("Précision par chiffre :")
for c in range(10):
    total = matrice[c].sum()
    print(f"  {c} : {100 * matrice[c][c] / total:.2f}%  ({matrice[c][c]}/{total})")
print(f"Précision globale : {100 * np.trace(matrice) / matrice.sum():.2f}%")

# Les 3 erreurs les plus fréquentes
erreurs = [(matrice[v][p], v, p) for v in range(10) for p in range(10) if v != p]
erreurs.sort(reverse=True)
print("Confusions les plus fréquentes :")
for n, v, p in erreurs[:3]:
    print(f"  {v} pris pour {p} : {n} fois")

# Affichage de la matrice
plt.imshow(matrice, cmap="Blues")
plt.title("Matrice de confusion")
plt.xlabel("Réponse du modèle")
plt.ylabel("Vrai chiffre")
plt.xticks(range(10))
plt.yticks(range(10))
for v in range(10):
    for p in range(10):
        plt.text(p, v, matrice[v][p], ha="center", va="center", fontsize=7,
                 color="white" if matrice[v][p] > matrice.max() / 2 else "black")
plt.colorbar()
plt.show()