import os
import glob
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from model import CNN
from traces import lettre_vers_28x28

# ===== Réglages =====
DOSSIER_TEST = "dataset_test"                      # le test frais
CARACTERES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MOYENNE, ECART = 0.1307, 0.3081

with open("classes_emnist.json") as f:
    classes = json.load(f)                          # les 47 caractères du modèle


def charger_modele(fichier):
    modele = CNN(nb_classes=47)
    modele.load_state_dict(torch.load(fichier))
    modele.eval()
    return modele


modeles = {
    "AVANT (modele_emnist)": charger_modele("modele_emnist.pt"),
    "APRES (modele_perso)": charger_modele("modele_perso.pt"),
}

# ===== Charger le test frais =====
noms, images, etiquettes = [], [], []
for chemin in sorted(glob.glob(os.path.join(DOSSIER_TEST, "*.json"))):
    with open(chemin) as f:
        exemple = json.load(f)
    traits = [t for t in exemple["traits"] if len(t) > 0]
    noms.append(os.path.basename(chemin))
    images.append(lettre_vers_28x28(traits))
    etiquettes.append(classes.index(exemple["label"]))

x = torch.from_numpy(np.array(images)).float() / 255.0
x = ((x - MOYENNE) / ECART).unsqueeze(1)            # forme (N, 1, 28, 28)
y = torch.tensor(etiquettes)
print(f"Test frais : {len(y)} exemples\n")


def predictions(modele, avec_mode):
    # avec_mode = True : on n'autorise que la bonne famille (chiffres ou majuscules A-Z)
    with torch.no_grad():
        probas = torch.softmax(modele(x), dim=1)
        if avec_mode:
            masque = torch.zeros_like(probas)
            masque[y < 10, :10] = 1
            masque[y >= 10, 10:36] = 1
            probas = probas * masque
        return probas.argmax(dim=1)


def precision(pred):
    return 100 * (pred == y).float().mean().item()


# ===== Tableau des scores =====
print(f"{'':24s} | bon mode | mixte")
for nom, modele in modeles.items():
    print(f"{nom:24s} | {precision(predictions(modele, True)):6.1f} %  | "
          f"{precision(predictions(modele, False)):5.1f} %")

# ===== Liste des erreurs =====
for nom, modele in modeles.items():
    for libelle, avec_mode in (("bon mode", True), ("mixte", False)):
        pred = predictions(modele, avec_mode).tolist()
        erreurs = [f"{noms[i]} ({classes[y[i]]}->{classes[pred[i]]})"
                   for i in range(len(pred)) if pred[i] != y[i].item()]
        print(f"\nErreurs {nom}, {libelle} : {len(erreurs)}")
        for e in erreurs:
            print("   ", e)

# ===== Matrices de confusion (mode mixte, le plus exigeant) =====
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
for ax, (nom, modele) in zip(axes, modeles.items()):
    pred = predictions(modele, False).tolist()
    matrice = np.zeros((36, 47), dtype=int)         # lignes = vrai, colonnes = réponse
    for vrai, lu in zip(y.tolist(), pred):
        matrice[vrai][lu] += 1
    ax.imshow(matrice, cmap="Blues")
    ax.set_title(f"{nom} : {precision(torch.tensor(pred)):.1f} % (mode mixte)")
    ax.set_xticks(range(47))
    ax.set_xticklabels(classes, fontsize=6)
    ax.set_yticks(range(36))
    ax.set_yticklabels(list(CARACTERES), fontsize=6)
    ax.set_xlabel("Réponse du modèle")
    ax.set_ylabel("Vrai caractère")
    for v in range(36):                             # les erreurs en rouge
        for p in range(47):
            if matrice[v][p] > 0 and v != p:
                ax.text(p, v, matrice[v][p], ha="center", va="center", color="red", fontsize=8)
plt.tight_layout()
plt.savefig("docs/matrices_avant_apres.png", dpi=150)    # enregistre l'image pour le README
plt.show()