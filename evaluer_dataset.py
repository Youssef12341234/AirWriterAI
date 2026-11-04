import os
import glob
import json
import cv2
import numpy as np
import torch
from model import CNN
from traces import lettre_vers_28x28

DOSSIER = "dataset"

# ----- Le modèle EMNIST et la liste de ses 47 caractères -----
with open("classes_emnist.json") as f:
    classes = json.load(f)
model = CNN(nb_classes=47)
model.load_state_dict(torch.load("modele_emnist.pt"))
model.eval()
MOYENNE, ECART = 0.1307, 0.3081


def predire(img, est_chiffre):
    # est_chiffre = True : on n'autorise que les chiffres ; False : que les lettres
    x = torch.from_numpy(img).float() / 255.0
    x = (x - MOYENNE) / ECART
    x = x.reshape(1, 1, 28, 28)
    with torch.no_grad():
        probas = torch.softmax(model(x), dim=1)[0]
    masque = torch.zeros(47)
    if est_chiffre:
        masque[:10] = 1          # numéros 0 à 9 = chiffres
    else:
            masque[10:36] = 1        # numéros 10 à 35 = majuscules A à Z seulement        # numéros 10 à 46 = lettres
    probas = probas * masque
    probas = probas / probas.sum()
    indice = int(probas.argmax())
    return classes[indice], float(probas[indice])


# ----- Lire tous les exemples et interroger le modèle -----
resultats = []     # (vrai caractère, lu, confiance, image, nom du fichier)
for chemin in sorted(glob.glob(os.path.join(DOSSIER, "*.json"))):
    with open(chemin) as f:
        exemple = json.load(f)
    traits = [t for t in exemple["traits"] if len(t) > 0]
    img = lettre_vers_28x28(traits)          # tout l'exemple = UNE seule lettre
    lu, confiance = predire(img, exemple["label"].isdigit())
    resultats.append((exemple["label"], lu, confiance, img, os.path.basename(chemin)))

# ----- Précision globale -----
bons = sum(1 for vrai, lu, *_ in resultats if vrai == lu)
print(f"Précision actuelle sur tes exemples : {100 * bons / len(resultats):.1f} %"
      f" ({bons}/{len(resultats)})")

# ----- Liste des erreurs -----
print("Erreurs (écrit -> lu) :")
for vrai, lu, confiance, _, nom in resultats:
    if vrai != lu:
        print(f"  {nom} : écrit {vrai}, lu {lu} ({100 * confiance:.0f} %)")

# ----- Affichage des images mal lues (24 au maximum) -----
erreurs = [r for r in resultats if r[0] != r[1]][:24]
if erreurs:
    vignettes = []
    for vrai, lu, confiance, img, nom in erreurs:
        g = cv2.resize(img, (112, 112), interpolation=cv2.INTER_NEAREST)
        g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.putText(g, f"{vrai}->{lu}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        vignettes.append(cv2.copyMakeBorder(g, 0, 2, 0, 2, cv2.BORDER_CONSTANT, value=(80, 80, 80)))
    while len(vignettes) % 8 != 0:                      # on complète la dernière ligne
        vignettes.append(np.zeros_like(vignettes[0]))
    lignes = [np.hstack(vignettes[i:i + 8]) for i in range(0, len(vignettes), 8)]
    cv2.imshow("Erreurs du modele (touche pour fermer)", np.vstack(lignes))
    cv2.waitKey(0)
    cv2.destroyAllWindows()