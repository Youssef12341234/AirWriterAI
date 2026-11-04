import os
import glob
import json
import shutil
import cv2
import numpy as np
from traces import lettre_vers_28x28

# ===== Réglages =====
DOSSIER = "dataset"                # les exemples
REJETES = "dataset_rejetes"        # les exemples écartés (déplacés, pas supprimés)
CARACTERES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VIGNETTE = 140                     # taille d'une image à l'écran (pixels)
PAR_LIGNE = 6                      # nombre d'images par ligne


def charger_classe(c):
    # lit tous les exemples d'un caractère et les convertit en images 28x28
    exemples = []
    for chemin in sorted(glob.glob(os.path.join(DOSSIER, f"{c}_*.json"))):
        with open(chemin) as f:
            exemple = json.load(f)
        traits = [t for t in exemple["traits"] if len(t) > 0]
        exemples.append((os.path.basename(chemin), lettre_vers_28x28(traits)))
    return exemples


def dessiner_ecran(c, exemples, marques):
    # fabrique l'image de la fenêtre : une vignette par exemple
    lignes = max(1, -(-len(exemples) // PAR_LIGNE))          # nombre de lignes (arrondi au-dessus)
    largeur = PAR_LIGNE * (VIGNETTE + 4)
    hauteur = 60 + lignes * (VIGNETTE + 4)
    ecran = np.zeros((hauteur, largeur, 3), dtype=np.uint8)

    cv2.putText(ecran, f"Caractere {c} : {len(exemples)} exemples | clic = marquer / demarquer",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(ecran, "Entree: ecarter les marques | n: suivant | p: precedent | q: quitter",
                (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    zones = []                                                # pour retrouver quel exemple on clique
    for i, (nom, img) in enumerate(exemples):
        x = (i % PAR_LIGNE) * (VIGNETTE + 4)
        y = 60 + (i // PAR_LIGNE) * (VIGNETTE + 4)
        vignette = cv2.resize(img, (VIGNETTE, VIGNETTE), interpolation=cv2.INTER_NEAREST)
        vignette = cv2.cvtColor(vignette, cv2.COLOR_GRAY2BGR)
        marque = nom in marques
        cv2.rectangle(vignette, (0, 0), (VIGNETTE - 1, VIGNETTE - 1),
                      (0, 0, 255) if marque else (80, 80, 80), 5 if marque else 1)
        cv2.putText(vignette, nom.replace(".json", ""), (6, VIGNETTE - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if marque:
            cv2.putText(vignette, "X", (VIGNETTE - 35, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3)
        ecran[y:y + VIGNETTE, x:x + VIGNETTE] = vignette
        zones.append((x, y, x + VIGNETTE, y + VIGNETTE, nom))
    return ecran, zones


def clic(evenement, x, y, flags, param):
    # un clic sur une vignette la marque (ou la démarque)
    if evenement == cv2.EVENT_LBUTTONDOWN:
        for x0, y0, x1, y1, nom in zones:
            if x0 <= x < x1 and y0 <= y < y1:
                if nom in marques:
                    marques.remove(nom)
                else:
                    marques.add(nom)


# ===== Programme =====
os.makedirs(REJETES, exist_ok=True)
idx = 0                                    # position dans la liste des caractères
marques = set()                            # les exemples marqués "à écarter"
exemples = charger_classe(CARACTERES[idx])
zones = []

cv2.namedWindow("Nettoyage")
cv2.setMouseCallback("Nettoyage", clic)

while True:
    ecran, zones = dessiner_ecran(CARACTERES[idx], exemples, marques)
    cv2.imshow("Nettoyage", ecran)
    key = cv2.waitKey(30) & 0xFF

    if key == ord("q"):
        break
    elif key in (ord("n"), ord("p")):                       # changer de caractère
        idx = (idx + (1 if key == ord("n") else -1)) % len(CARACTERES)
        marques = set()
        exemples = charger_classe(CARACTERES[idx])
    elif key == 13:                                         # Entrée : écarter les marqués
        for nom in marques:
            shutil.move(os.path.join(DOSSIER, nom), os.path.join(REJETES, nom))
        print(f"{len(marques)} exemple(s) écarté(s) pour le caractère {CARACTERES[idx]}")
        marques = set()
        exemples = charger_classe(CARACTERES[idx])

cv2.destroyAllWindows()