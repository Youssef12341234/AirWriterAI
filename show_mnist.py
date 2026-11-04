import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import datasets

train = datasets.MNIST(root="data", train=True, download=False)

# 1. Afficher 10 images avec leur label
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    img, label = train[i]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Label : {label}")
    ax.axis("off")
plt.tight_layout()
plt.show()

# 2. Valeurs des pixels
arr = np.array(train[0][0])
print("Pixels : min =", arr.min(), "| max =", arr.max())

# 3. Nombre d'images par classe
compte = Counter(train.targets.tolist())
print("Images par classe :")
for c in sorted(compte):
    print(c, ":", compte[c])