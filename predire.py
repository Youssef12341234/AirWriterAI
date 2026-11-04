import json
import cv2
import numpy as np
import torch
from model import CNN
from traces import separer_lettres, lettre_vers_28x28

# ----- Charger le modèle entraîné -----
model = CNN()
model.load_state_dict(torch.load("modele_mnist.pt"))
model.eval()                       # mode examen

# mêmes valeurs de normalisation que dans train.py
MOYENNE, ECART = 0.1307, 0.3081


def predire(img):
    # img : image 28x28 avec des pixels de 0 à 255
    x = torch.from_numpy(img).float() / 255.0     # pixels entre 0 et 1
    x = (x - MOYENNE) / ECART                     # même normalisation qu'à l'entraînement
    x = x.reshape(1, 1, 28, 28)                   # 1 image, 1 canal, 28x28
    with torch.no_grad():                         # pas d'apprentissage ici
        scores = model(x)
        probas = torch.softmax(scores, dim=1)[0]  # scores transformés en pourcentages
    top = torch.topk(probas, 2)                   # les 2 meilleurs choix
    return top.indices.tolist(), top.values.tolist()


def afficher(img, texte):
    # agrandit l'image pour la voir et écrit la prédiction dessus
    grand = cv2.resize(img, (168, 168), interpolation=cv2.INTER_NEAREST)
    grand = cv2.cvtColor(grand, cv2.COLOR_GRAY2BGR)
    cv2.putText(grand, texte, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return cv2.copyMakeBorder(grand, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(80, 80, 80))


# ----- Programme -----
with open("dernier_trace.json") as f:
    donnees = json.load(f)

lettres = separer_lettres(donnees["traits"])
images = []
for i, lettre in enumerate(lettres, start=1):
    img = lettre_vers_28x28(lettre)
    classes, confiances = predire(img)
    print(f"Lettre {i} : 1er choix {classes[0]} ({100 * confiances[0]:.0f} %) | "
          f"2e choix {classes[1]} ({100 * confiances[1]:.0f} %)")
    images.append(afficher(img, f"{classes[0]} ({100 * confiances[0]:.0f}%)"))

cv2.imshow("Predictions du modele (touche pour fermer)", np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()