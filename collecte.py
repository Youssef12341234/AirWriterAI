import os
import time
import math
import json
import copy
import random
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ===== Réglages de la collecte (NOUVEAU) =====
CARACTERES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # les 36 caractères à collecter
OBJECTIF = 2          # nombre d'exemples voulus par caractère
DOSSIER = "dataset_test"    # dossier où les exemples sont rangés
POINTS_MIN = 15        # un exemple avec moins de points est refusé
TAILLE_MIN = 40        # un exemple plus petit (en pixels) est refusé

# ===== Réglages du suivi (les mêmes que draw.py) =====
RAYON_GOMME = 40
EPAISSEUR = 6
DELAI_DEBUT = 0.25
DELAI_FIN = 0.15
DELAI_GOMME = 0.20
LONGUEUR_MIN = 25
SAUT_MAX = 150
MIN_CUTOFF = 1.5
BETA = 0.02
SEUIL_HAUT = 0.15
SEUIL_BAS = 0.05
PAIRES = [(8, 6), (12, 10), (16, 14), (20, 18)]

# ===== Détecteur de main =====
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(options)


# ===== Fonctions déjà vues dans draw.py =====
class OneEuro:
    """Filtre qui lisse beaucoup quand tu vas lentement et peu quand tu vas vite."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def hysteresis(etat, valeur):
    if valeur > SEUIL_HAUT:
        return True
    if valeur < SEUIL_BAS:
        return False
    return etat


def sauvegarder(traits, historique):
    historique.append(copy.deepcopy(traits))
    if len(historique) > 30:
        historique.pop(0)


def dessiner(traits, forme):
    canvas = np.zeros(forme, dtype=np.uint8)
    for trait in traits:
        if len(trait) == 1:
            cv2.circle(canvas, trait[0], EPAISSEUR // 2, (255, 255, 255), -1)
        elif len(trait) >= 2:
            points = np.array(trait, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], False, (255, 255, 255), EPAISSEUR, cv2.LINE_AA)
    return canvas


def gommer(traits, centre, rayon):
    resultat = []
    for trait in traits:
        morceau = []
        for p in trait:
            if math.hypot(p[0] - centre[0], p[1] - centre[1]) > rayon:
                morceau.append(p)
            else:
                if morceau:
                    resultat.append(morceau)
                morceau = []
        if morceau:
            resultat.append(morceau)
    return resultat


def boite(traits):
    points = [p for trait in traits for p in trait]
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


# ===== NOUVELLES fonctions de collecte =====
def compter_exemples():
    # compte les exemples déjà enregistrés pour chaque caractère (d'après le nom des fichiers)
    os.makedirs(DOSSIER, exist_ok=True)
    compte = {c: 0 for c in CARACTERES}
    for nom in os.listdir(DOSSIER):
        if nom.endswith(".json"):
            c = nom.split("_")[0]          # "B_0007.json" -> "B"
            if c in compte:
                compte[c] += 1
    return compte


def choisir_cible(compte, exclure=None):
    # choisit au hasard parmi les caractères qui ont le MOINS d'exemples
    possibles = [c for c in CARACTERES if c != exclure]
    minimum = min(compte[c] for c in possibles)
    return random.choice([c for c in possibles if compte[c] == minimum])


def enregistrer_exemple(traits, cible, compte):
    # écrit un fichier "B_0001.json" avec l'étiquette, les traits et la boîte
    numero = compte[cible] + 1
    while os.path.exists(os.path.join(DOSSIER, f"{cible}_{numero:04d}.json")):
        numero += 1                         # on ne remplace jamais un fichier existant
    donnees = {"label": cible, "traits": traits, "boite": boite(traits)}
    with open(os.path.join(DOSSIER, f"{cible}_{numero:04d}.json"), "w") as f:
        json.dump(donnees, f)


# ===== Variables du programme =====
cap = cv2.VideoCapture(0)

compte = compter_exemples()
cible = choisir_cible(compte)           # NOUVEAU : le caractère à écrire
message = ""                            # NOUVEAU : petit message temporaire à l'écran
message_fin = 0

traits = []
trait_courant = None
historique = []
n_hist_debut = 0
fx = OneEuro(MIN_CUTOFF, BETA)
fy = OneEuro(MIN_CUTOFF, BETA)
doigts = [False, False, False, False]
stylo = False
ecriture_active = False
ecriture_avant = False
gomme_avant = False
longueur_trait = 0.0
t_debut = None
t_fin = None
t_gomme = None

# ===== Boucle principale =====
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    maintenant = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(image, int(maintenant * 1000))

    h, w, _ = frame.shape
    point = None
    centre = None

    # --- 1. Lire la main ---
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        taille = max(distance(hand[0], hand[9]), 1e-6)
        for i, (bout, art) in enumerate(PAIRES):
            ecart = (hand[art].y - hand[bout].y) / taille
            doigts[i] = hysteresis(doigts[i], ecart)
        x = fx(hand[8].x * w, maintenant)
        y = fy(hand[8].y * h, maintenant)
        point = (int(x), int(y))
        centre = (int(hand[9].x * w), int(hand[9].y * h))
    else:
        doigts = [False, False, False, False]
        fx.reset()
        fy.reset()

    index, majeur, annulaire, auriculaire = doigts
    main_ouverte = all(doigts)
    index_pret = index and not majeur

    # --- 2. Décider si on écrit ---
    if stylo and index_pret:
        if t_debut is None:
            t_debut = maintenant
        t_fin = None
        if maintenant - t_debut >= DELAI_DEBUT:
            ecriture_active = True
    else:
        t_debut = None
        if t_fin is None:
            t_fin = maintenant
        if maintenant - t_fin >= DELAI_FIN:
            ecriture_active = False

    # --- 3. Décider si on gomme ---
    if main_ouverte:
        if t_gomme is None:
            t_gomme = maintenant
    else:
        t_gomme = None
    gomme_active = t_gomme is not None and maintenant - t_gomme >= DELAI_GOMME

    # --- 4. Début et fin d'un trait ---
    if ecriture_active and not ecriture_avant:
        sauvegarder(traits, historique)
        n_hist_debut = len(historique)
        trait_courant = []
        traits.append(trait_courant)
        longueur_trait = 0.0
    if not ecriture_active and ecriture_avant:
        if longueur_trait < LONGUEUR_MIN and len(historique) == n_hist_debut:
            traits = historique.pop()
        trait_courant = None
    ecriture_avant = ecriture_active

    if gomme_active and not gomme_avant:
        sauvegarder(traits, historique)
    gomme_avant = gomme_active

    # --- 5. Mode affiché ---
    if gomme_active:
        mode = "GOMME"
    elif ecriture_active:
        mode = "ECRITURE"
    elif stylo and index_pret:
        mode = "PREPARATION"
    elif index_pret:
        mode = "POINTEUR"
    else:
        mode = "PAUSE"

    # --- 6. Écrire ---
    if ecriture_active and not gomme_active and point is not None:
        if trait_courant is None:
            trait_courant = []
            traits.append(trait_courant)
        if trait_courant:
            dernier = trait_courant[-1]
            d = math.hypot(point[0] - dernier[0], point[1] - dernier[1])
            if d > SAUT_MAX:
                trait_courant = [point]
                traits.append(trait_courant)
            elif d >= 2:
                trait_courant.append(point)
                longueur_trait += d
        else:
            trait_courant.append(point)

    # --- 7. Gommer ---
    if gomme_active and centre is not None:
        traits = gommer(traits, centre, RAYON_GOMME)
        trait_courant = None
        cv2.circle(frame, centre, RAYON_GOMME, (255, 255, 255), 2)

    # --- 8. Affichage ---
    if point is not None:
        if mode == "ECRITURE":
            couleur = (0, 255, 0)
        elif mode == "PREPARATION":
            couleur = (0, 255, 255)
            progres = min((maintenant - t_debut) / DELAI_DEBUT, 1) if t_debut else 0
            cv2.ellipse(frame, point, (20, 20), -90, 0, int(360 * progres), couleur, 3)
        else:
            couleur = (0, 0, 255)
        cv2.circle(frame, point, 8, couleur, -1)

    b = boite(traits)
    if b is not None:
        cv2.rectangle(frame, (b[0] - 10, b[1] - 10), (b[2] + 10, b[3] + 10), (255, 128, 0), 1)

    frame = cv2.add(frame, dessiner(traits, frame.shape))

    nb_points = sum(len(t) for t in traits)
    cv2.putText(frame, mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    etat = "STYLO : ON" if stylo else "STYLO : OFF"
    couleur_etat = (0, 255, 0) if stylo else (0, 0, 255)
    cv2.putText(frame, etat, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur_etat, 2)

    # NOUVEAU : le caractère à écrire, en grand, en haut à droite
    cv2.putText(frame, "Ecris :", (w - 190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, cible, (w - 130, 125), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 6)
    cv2.putText(frame, f"{cible} : {compte[cible]}/{OBJECTIF}", (w - 190, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # NOUVEAU : message temporaire (exemple enregistré, refusé...)
    if maintenant < message_fin:
        cv2.putText(frame, message, (w // 2 - 150, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # NOUVEAU : bandeau du bas avec la progression et l'aide
    fait = sum(min(compte[c], OBJECTIF) for c in CARACTERES)
    total = OBJECTIF * len(CARACTERES)
    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, h - 62), (w - 10, h - 48), (80, 80, 80), -1)
    cv2.rectangle(frame, (10, h - 62), (10 + int((w - 20) * fait / total), h - 48), (0, 200, 0), -1)
    cv2.putText(frame, f"Progression : {fait}/{total} | Enregistres : {sum(compte.values())}",
                (10, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Espace: stylo | Entree: valider | s: passer | u: annuler | c: vider | q: quitter",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    cv2.imshow("AirWriter - Collecte", frame)

    # --- 9. Touches ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord(" "):                # armer / désarmer le stylo
        stylo = not stylo
    elif key == ord("u"):                # annuler
        if historique:
            traits = historique.pop()
            trait_courant = None
    elif key == ord("c"):                # vider l'écran
        sauvegarder(traits, historique)
        traits = []
        trait_courant = None
    elif key == ord("s"):                # NOUVEAU : passer ce caractère
        cible = choisir_cible(compte, exclure=cible)
    elif key == 13:                      # NOUVEAU : Entrée = enregistrer l'exemple
        propres = [t for t in traits if len(t) > 0]
        b = boite(propres)
        if b is None or nb_points < POINTS_MIN or max(b[2] - b[0], b[3] - b[1]) < TAILLE_MIN:
            message = "Trop petit : recommence"
            message_fin = maintenant + 2
        else:
            enregistrer_exemple(propres, cible, compte)
            compte[cible] += 1
            message = f"{cible} enregistre !"
            message_fin = maintenant + 1.5
            traits = []
            trait_courant = None
            historique = []
            cible = choisir_cible(compte, exclure=cible)

cap.release()
cv2.destroyAllWindows()
print("Exemples enregistrés par caractère :", compte)