import cv2
import numpy as np

# ===== Réglages =====
ESPACE_MIN = 2           # vide minimum entre deux lettres
TAILLE = 28              # taille finale de l'image (comme MNIST)
ECHELLE = 10             # on dessine 10 fois plus grand, puis on réduit
GRAND = TAILLE * ECHELLE # 280 : taille du grand carré de travail
ZONE = 20 * ECHELLE      # 200 : zone où tient la lettre (20x20 dans MNIST)
EPAISSEUR = 25           # épaisseur du trait en grand


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


def lettre_vers_28x28(lettre):
    # transforme une lettre (liste de traits) en image 28x28 style MNIST
    x0, y0, x1, y1 = boite(lettre)
    largeur = max(x1 - x0, 1)
    hauteur = max(y1 - y0, 1)

    echelle = ZONE / max(largeur, hauteur)              # un seul facteur : pas de déformation
    decalage_x = (GRAND - largeur * echelle) / 2        # pour centrer la lettre
    decalage_y = (GRAND - hauteur * echelle) / 2

    grande = np.zeros((GRAND, GRAND), dtype=np.uint8)   # grand carré noir
    for trait in lettre:
        points = [(int((x - x0) * echelle + decalage_x),
                   int((y - y0) * echelle + decalage_y)) for x, y in trait]
        if len(points) == 1:
            cv2.circle(grande, points[0], EPAISSEUR // 2, 255, -1)
        else:
            tableau = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(grande, [tableau], False, 255, EPAISSEUR, cv2.LINE_AA)

    return cv2.resize(grande, (TAILLE, TAILLE), interpolation=cv2.INTER_AREA)