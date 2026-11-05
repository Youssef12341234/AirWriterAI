# AirWriterAI : écrire dans l'air avec la main

Programme personnel qui lit ce que j'écris **dans l'air**, avec l'index devant la webcam. La main est détectée avec MediaPipe, le tracé est converti en image 28×28, puis lu par un petit réseau de neurones (CNN, PyTorch) entraîné sur EMNIST et affiné sur **mon** écriture.

```
webcam -> main (MediaPipe) -> tracé lissé -> lettres séparées -> image 28x28 -> CNN -> texte
```

## Lancer

Python 3.12 (utilisé ici). Dans le dossier du projet :

```
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python draw.py
```

## Utilisation

- **Espace** : active ou désactive le stylo. Index levé (majeur replié) : écriture, avec un court délai.
- **Main ouverte** : gomme locale.
- **Entrée** : lit ce qui est écrit et l'ajoute au texte.
- **m** : change le mode de lecture (texte A-Z, chiffres, mixte).
- **n** : espace. **Retour arrière** : supprime une lettre. **u** : annule. **c** : vide l'écran. **q** : quitte.
- Laisser un petit vide entre les lettres : elles sont séparées d'après l'espace horizontal.

## Résultats

Test indépendant : 73 exemples écrits dans une séance différente de l'entraînement.

| Modèle | Bon mode (chiffre ou lettre connu) | Mode mixte (sans aide) |
|---|---|---|
| Départ (EMNIST) | 93,2 % | 80,8 % |
| Personnalisé | **98,6 %** | **98,6 %** |

Contrôle d'oubli sur EMNIST : 87,5 % avant, 87,6 % après. Matrices de confusion : `docs/matrices_avant_apres.png`.

## Limites

- Données d'une seule personne (moi), avec une seule webcam : le modèle est personnalisé.
- 73 exemples de test : une erreur vaut environ 1,4 point.
- Majuscules A-Z et chiffres seulement ; je teste surtout des caractères isolés.

## Fichiers

| Fichier | Rôle |
|---|---|
| `draw.py` | Application principale |
| `lecteur.py`, `traces.py`, `model.py` | Lecture, séparation des lettres, réseau |
| `modele_perso.pt` | Modèle final |
| `modele_emnist.pt`, `modele_mnist.pt` | Modèles intermédiaires |
| `dataset/`, `dataset_test/`, `split.json` | Mes exemples, mon test, la répartition |
| `collecte.py`, `nettoyer.py` | Collecter et nettoyer mes exemples |
| `finetune.py`, `comparer.py` | Affinage et comparaison avant/après |
| `webcam.py`, `hand.py`, `*_mnist.py`, `*_emnist.py`, `separer.py`, `convertir.py`, `predire*.py` | Tests et étapes intermédiaires |

## Refaire l'entraînement

1. `python explore_emnist.py` : télécharge EMNIST (environ 500 Mo, dans `data/`).
2. `python train_emnist.py` : entraîne le modèle de départ.
3. `python collecte.py`, puis `python nettoyer.py` : mes exemples.
4. `python preparer_donnees.py` : séparation apprentissage / test.
5. `python finetune.py` : affinage, crée `modele_perso.pt`.
6. `python comparer.py` : comparaison sur le test indépendant (`dataset_test/`).