import os
import glob
import json
import torch
from model import CNN
from traces import lettre_vers_28x28

# ===== Réglages =====
DOSSIER = "dataset"
CARACTERES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NB_TEST = 2        # nombre d'exemples de test par caractère

# ===== 1. Séparer apprentissage et test =====
apprentissage, test = [], []
for c in CARACTERES:
    fichiers = sorted(glob.glob(os.path.join(DOSSIER, f"{c}_*.json")))
    noms = [os.path.basename(f) for f in fichiers]
    test += noms[-NB_TEST:]               # les derniers écrits = test
    apprentissage += noms[:-NB_TEST]      # les autres = apprentissage

with open("split.json", "w") as f:
    json.dump({"apprentissage": apprentissage, "test": test}, f)
print(f"Apprentissage : {len(apprentissage)} exemples | Test : {len(test)} exemples")
print("Répartition enregistrée dans split.json")

# ===== 2. Mesurer le modèle actuel sur le test (le score « avant ») =====
with open("classes_emnist.json") as f:
    classes = json.load(f)
model = CNN(nb_classes=47)
model.load_state_dict(torch.load("modele_emnist.pt"))
model.eval()
MOYENNE, ECART = 0.1307, 0.3081


def predire(img, est_chiffre):
    # chiffres autorisés (0-9) ou majuscules autorisées (A-Z), selon le caractère attendu
    x = torch.from_numpy(img).float() / 255.0
    x = (x - MOYENNE) / ECART
    x = x.reshape(1, 1, 28, 28)
    with torch.no_grad():
        probas = torch.softmax(model(x), dim=1)[0]
    masque = torch.zeros(47)
    if est_chiffre:
        masque[:10] = 1
    else:
        masque[10:36] = 1
    probas = probas * masque
    probas = probas / probas.sum()
    indice = int(probas.argmax())
    return classes[indice], float(probas[indice])


bons = 0
erreurs = []
for nom in test:
    with open(os.path.join(DOSSIER, nom)) as f:
        exemple = json.load(f)
    traits = [t for t in exemple["traits"] if len(t) > 0]
    img = lettre_vers_28x28(traits)
    lu, confiance = predire(img, exemple["label"].isdigit())
    if lu == exemple["label"]:
        bons += 1
    else:
        erreurs.append((nom, exemple["label"], lu, confiance))

print(f"\nScore AVANT fine-tuning sur le test : {100 * bons / len(test):.1f} % ({bons}/{len(test)})")
print("Erreurs (écrit -> lu) :")
for nom, vrai, lu, confiance in erreurs:
    print(f"  {nom} : écrit {vrai}, lu {lu} ({100 * confiance:.0f} %)")