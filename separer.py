import json
import cv2
import numpy as np

ESPACE_MIN = 2 # un vide plus grand que ça (en pixels) = nouvelle lettre


def etendue(trait):
    # donne la zone horizontale d'un trait : (x le plus à gauche, x le plus à droite)
    xs = [p[0] for p in trait]
    return min(xs), max(xs)


def separer_lettres(traits):
    # 1. on enlève les traits vides, puis on les range de gauche à droite
    traits = [t for t in traits if len(t) > 0]
    traits.sort(key=lambda t: etendue(t)[0])

    lettres = []             # la liste des lettres (chaque lettre = une liste de traits)
    x_fin_lettre = None      # jusqu'où va la lettre en cours (côté droit)

    for trait in traits:
        x_min, x_max = etendue(trait)
        if x_fin_lettre is not None and x_min - x_fin_lettre <= ESPACE_MIN:
            # pas de grand vide : ce trait fait partie de la même lettre
            lettres[-1].append(trait)
            x_fin_lettre = max(x_fin_lettre, x_max)
        else:
            # grand vide (ou premier trait) : c'est une nouvelle lettre
            lettres.append([trait])
            x_fin_lettre = x_max
    return lettres


def boite(traits):
    # le rectangle qui entoure une lettre : (x min, y min, x max, y max)
    points = [p for trait in traits for p in trait]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


# ----- Programme -----
with open("dernier_trace.json") as f:
    donnees = json.load(f)

lettres = separer_lettres(donnees["traits"])
print(f"{len(lettres)} lettre(s) trouvée(s)")
for i, lettre in enumerate(lettres, start=1):
    print(f"  Lettre {i} : {len(lettre)} trait(s), boîte {boite(lettre)}")

# ----- Affichage : une couleur par lettre -----
tous = [p for lettre in lettres for trait in lettre for p in trait]
largeur = max(p[0] for p in tous) + 40
hauteur = max(p[1] for p in tous) + 40
image = np.zeros((hauteur, largeur, 3), dtype=np.uint8)

couleurs = [(0, 255, 0), (0, 165, 255), (255, 128, 0), (255, 0, 255), (0, 255, 255)]

for i, lettre in enumerate(lettres):
    couleur = couleurs[i % len(couleurs)]
    for trait in lettre:
        points = np.array(trait, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(image, [points], False, couleur, 5, cv2.LINE_AA)
    x0, y0, x1, y1 = boite(lettre)
    cv2.rectangle(image, (x0 - 8, y0 - 8), (x1 + 8, y1 + 8), couleur, 1)
    cv2.putText(image, str(i + 1), (x0, y0 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, couleur, 2)

cv2.imshow("Lettres separees (appuie sur une touche pour fermer)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()