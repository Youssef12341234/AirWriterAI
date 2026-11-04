import os
import json
import cv2
import numpy as np
from torchvision import datasets

# ===== Réglages =====
ESPACE_MIN = 2           # vide minimum entre deux lettres (comme en S7)
TAILLE = 28              # taille finale de l'image (comme MNIST)
ECHELLE = 10             # on dessine 10 fois plus grand, puis on réduit
GRAND = TAILLE * ECHELLE # 280 : taille du grand carré de travail
ZONE = 20 * ECHELLE      # 200 : zone où tient la lettre (20x20 dans MNIST)
EPAISSEUR = 25           # épaisseur du trait en grand (environ 2,5 pixels en 28x28)


# ===== Fonctions de la S7 (séparer les lettres) =====
def etendue(trait):
    # zone horizontale occupée par un trait : (x gauche, x droite)
    xs = [p[0] for p in trait]
    return min(xs), max(xs)


def separer_lettres(traits):
    # range les traits de gauche à droite, puis les regroupe en lettres
    traits = [t for t in traits if len(t) > 0]
    traits.sort(key=lambda t: etendue(t)[0])
    lettres = []
    x_fin_lettre = None
    for trait in traits:
        x_min, x_max = etendue(trait)
        if x_fin_lettre is not None and x_min - x_fin_lettre <= ESPACE_MIN:
            lettres[-1].append(trait)               # même lettre
            x_fin_lettre = max(x_fin_lettre, x_max)
        else:
            lettres.append([trait])                 # nouvelle lettre
            x_fin_lettre = x_max
    return lettres


def boite(traits):
    # rectangle qui entoure une lettre : (x min, y min, x max, y max)
    points = [p for trait in traits for p in trait]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# ===== NOUVEAU : transformer une lettre en image 28x28 =====
def lettre_vers_28x28(lettre):
    x0, y0, x1, y1 = boite(lettre)
    largeur = max(x1 - x0, 1)
    hauteur = max(y1 - y0, 1)

    # un SEUL facteur pour x et y : la lettre est agrandie sans être déformée.
    # Le côté le plus long de la lettre occupe exactement la zone de 20x20.
    echelle = ZONE / max(largeur, hauteur)

    # décalage pour centrer la lettre dans le grand carré (= marge autour)
    decalage_x = (GRAND - largeur * echelle) / 2
    decalage_y = (GRAND - hauteur * echelle) / 2

    # grand carré noir de travail
    grande = np.zeros((GRAND, GRAND), dtype=np.uint8)

    for trait in lettre:
        # on convertit chaque point : on enlève le coin de la boîte, on agrandit, on décale
        points = [(int((x - x0) * echelle + decalage_x),
                   int((y - y0) * echelle + decalage_y)) for x, y in trait]
        if len(points) == 1:                         # un seul point : un rond
            cv2.circle(grande, points[0], EPAISSEUR // 2, 255, -1)
        else:                                        # sinon, on relie les points
            tableau = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(grande, [tableau], False, 255, EPAISSEUR, cv2.LINE_AA)

    # on réduit en 28x28 : INTER_AREA fait une moyenne, d'où les bords doux en gris
    return cv2.resize(grande, (TAILLE, TAILLE), interpolation=cv2.INTER_AREA)


# ===== Affichage =====
def agrandir(img, titre):
    # agrandit l'image 28x28 (x6) pour qu'on la voie, et ajoute un titre
    grand = cv2.resize(img, (168, 168), interpolation=cv2.INTER_NEAREST)
    grand = cv2.cvtColor(grand, cv2.COLOR_GRAY2BGR)
    cv2.putText(grand, titre, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return cv2.copyMakeBorder(grand, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(80, 80, 80))


# ===== Programme =====
with open("dernier_trace.json") as f:
    donnees = json.load(f)

lettres = separer_lettres(donnees["traits"])
os.makedirs("sorties", exist_ok=True)       # dossier où on range les images

images = []
for i, lettre in enumerate(lettres, start=1):
    img = lettre_vers_28x28(lettre)
    images.append(agrandir(img, f"Lettre {i}"))
    cv2.imwrite(f"sorties/lettre_{i}.png", img)
    print(f"Lettre {i} : taille {img.shape}, pixels de {img.min()} à {img.max()}, "
          f"moyenne {img.mean():.1f}")

# une vraie image MNIST pour comparer
mnist = datasets.MNIST(root="data", train=True, download=False)
ref, label = mnist[0]
ref = np.array(ref)
images.append(agrandir(ref, f"MNIST : {label}"))
print(f"MNIST (chiffre {label}) : taille {ref.shape}, pixels de {ref.min()} à {ref.max()}, "
      f"moyenne {ref.mean():.1f}")

cv2.imshow("Tes lettres en 28x28 + une image MNIST (touche pour fermer)", np.hstack(images))
cv2.waitKey(0)
cv2.destroyAllWindows()