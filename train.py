import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN          # on récupère notre modèle de model.py

# ----- Réglages -----
EPOCHS = 3      # nombre de tours complets sur les 60 000 images
BATCH = 64      # nombre d'images montrées en une fois
LR = 0.001      # force des corrections (learning rate)

# ----- Préparer les images -----
# ToTensor : transforme l'image en nombres
# Normalize : met ces nombres à une petite échelle (plus facile à apprendre)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Les images pour apprendre (train) et celles pour l'examen (test)
train_data = datasets.MNIST(root="data", train=True, download=False, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=False, transform=transform)

# Le DataLoader distribue les images par tas de 64
# shuffle=True : on mélange à chaque tour pour qu'il n'apprenne pas l'ordre par cœur
train_loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# ----- Préparer l'élève, la note d'erreur et le prof -----
model = CNN()                                   # le cerveau neuf
loss_fn = nn.CrossEntropyLoss()                 # calcule la note d'erreur
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # le prof qui corrige


# ----- Fonction d'examen -----
def evaluer():
    model.eval()                 # mode examen : on n'apprend plus
    correct = 0
    with torch.no_grad():        # pas de calcul d'apprentissage pendant l'examen
        for images, labels in test_loader:
            # argmax = le chiffre qui a le score le plus haut = sa réponse
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()   # on compte les bonnes réponses
    return 100 * correct / len(test_data)    # pourcentage de réussite


# ----- L'entraînement -----
for epoch in range(1, EPOCHS + 1):          # on répète 3 tours
    model.train()                           # mode apprentissage
    debut = time.time()
    total_perte = 0

    for images, labels in train_loader:     # on prend un tas de 64 images à la fois
        optimizer.zero_grad()               # 1. on efface les corrections du tas précédent
        reponses = model(images)            # 2. le modèle répond
        perte = loss_fn(reponses, labels)   # 3. on calcule son erreur
        perte.backward()                    # 4. on cherche d'où vient l'erreur
        optimizer.step()                    # 5. on corrige le modèle
        total_perte += perte.item()         # on garde la note pour l'affichage

    perte_moyenne = total_perte / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS} | perte : {perte_moyenne:.4f} | "
          f"précision test : {evaluer():.2f}% | {time.time() - debut:.0f}s")

# ----- Sauvegarder ce qu'il a appris -----
torch.save(model.state_dict(), "modele_mnist.pt")
print("Modèle sauvegardé : modele_mnist.pt")