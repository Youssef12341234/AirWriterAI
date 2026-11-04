import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

# on charge EMNIST (déjà téléchargé, donc download=False)
train = datasets.EMNIST(root="data", split="balanced", train=True, download=False)

# 1. Vérifier que chaque classe a le même nombre d'images
compte = np.bincount(train.targets.numpy())
print("Images par classe : minimum", compte.min(), "| maximum", compte.max())

# 2. Afficher 10 images : en haut les brutes, en bas les corrigées
fig, axes = plt.subplots(2, 10, figsize=(14, 3.8))
for i in range(10):
    img, label = train[i]
    brute = np.array(img)        # image telle qu'elle est enregistrée
    corrigee = brute.T           # .T = transposée : on échange lignes et colonnes

    axes[0][i].imshow(brute, cmap="gray")
    axes[0][i].set_title(train.classes[label])    # le caractère attendu
    axes[0][i].axis("off")

    axes[1][i].imshow(corrigee, cmap="gray")
    axes[1][i].axis("off")

fig.suptitle("Ligne du haut : images brutes (tournées)  |  Ligne du bas : images corrigées")
plt.tight_layout()
plt.show()