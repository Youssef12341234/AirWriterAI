import json
import torch
from model import CNN
from traces import separer_lettres, lettre_vers_28x28

MOYENNE, ECART = 0.1307, 0.3081   # mêmes valeurs qu'à l'entraînement

# la liste des 47 caractères
with open("classes_emnist.json") as f:
    classes = json.load(f)

# le modèle entraîné
model = CNN(nb_classes=47)
model.load_state_dict(torch.load("modele_perso.pt"))
model.eval()


def masque_du_mode(mode):
    # 1 = réponse autorisée, 0 = interdite (numéros 0-9 = chiffres, 10-46 = lettres)
    masque = torch.zeros(47)
    if mode == "texte":
        masque[10:36] = 1
    elif mode == "chiffres":
        masque[:10] = 1
    else:
        masque[:] = 1
    return masque


def predire(img, mode):
    # img : image 28x28 (pixels de 0 à 255). Rend (caractère, confiance)
    x = torch.from_numpy(img).float() / 255.0
    x = (x - MOYENNE) / ECART
    x = x.reshape(1, 1, 28, 28)
    with torch.no_grad():
        probas = torch.softmax(model(x), dim=1)[0]
    probas = probas * masque_du_mode(mode)     # on interdit les mauvaises réponses
    probas = probas / probas.sum()             # on refait un total de 100 %
    indice = int(probas.argmax())
    return classes[indice], float(probas[indice])


def lire_mot(traits, mode):
    # traits -> lettres séparées -> images 28x28 -> caractères -> mot
    resultats = []
    for lettre in separer_lettres(traits):
        img = lettre_vers_28x28(lettre)
        resultats.append(predire(img, mode))
    mot = "".join(caractere for caractere, _ in resultats)
    return mot, resultats