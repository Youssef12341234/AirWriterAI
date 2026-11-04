import json
import cv2
import numpy as np
import torch
from model import CNN
from traces import separer_lettres, lettre_vers_28x28

# ----- NOUVEAU : mode de lecture -----
# "texte"   = lettres seulement
# "chiffres" = chiffres seulement
# "mixte"   = tout
MODE = "texte"

# ----- La liste des 47 caractères (0..9, A..Z, a, b, d...) -----
with open("classes_emnist.json") as f:
    classes = json.load(f)

# ----- NOUVEAU : quelles réponses sont autorisées selon le mode -----
# les numéros 0 à 9 sont les chiffres, 10 à 46 sont les lettres
if MODE == "texte":
    autorises = list(range(10, 47))
elif MODE == "chiffres":
    autorises = list(range(0, 10))
else:
    autorises = list(range(47))
masque = torch.zeros(47)        # 0 = interdit, 1 = autorisé
masque[autorises] = 1

# ----- Charger le modèle (47 classes) -----
model = CNN(nb_classes=47)
model.load_state_dict(torch.load("modele_emnist.pt"))
model.eval()

MOYENNE, ECART = 0.1307, 0.3081


def predire(img):
    # img : image 28x28 avec des pixels de 0 à 255
    x = torch.from_numpy(img).float() / 255.0
    x = (x - MOYENNE) / ECART
    x = x.reshape(1, 1, 28, 28)
    with torch.no_grad():
        probas = torch.softmax(model(x), dim=1)[0]
    probas = probas * masque                 # NOUVEAU : on met à 0 les réponses interdites
    probas = probas / probas.sum()           # NOUVEAU : on refait un total de 100 %
    top = torch.topk(probas, 2)
    return top.indices.tolist(), top.values.tolist()


def afficher(img, texte):
    grand = cv2.resize(img, (168, 168), interpolation=cv2.INTER_NEAREST)
    grand = cv2.cvtColor(grand, cv2.COLOR_GRAY2BGR)
    cv2.putText(grand, texte, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return cv2.copyMakeBorder(grand, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(80, 80, 80))


# ----- Programme -----
with open("dernier_trace.json") as f:
    donnees = json.load(f)

print(f"Mode de lecture : {MODE}")
lettres = separer_lettres(donnees["traits"])
images = []
mot = ""
for i, lettre in enumerate(lettres, start=1):
    img = lettre_vers_28x28(lettre)
    ids, confs = predire(img)
    mot += classes[ids[0]]
    print(f"Lettre {i} : 1er choix {classes[ids[0]]} ({100 * confs[0]:.0f} %) | "
          f"2e choix {classes[ids[1]]} ({100 * confs[1]:.0f} %)")
    images.append(afficher(img, f"{classes[ids[0]]} ({100 * confs[0]:.0f}%)"))

print("Mot lu :", mot)

cv2.imshow("Predictions EMNIST (touche pour fermer)", np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()