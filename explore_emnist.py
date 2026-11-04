from torchvision import datasets

# download=True : télécharge le jeu de données s'il n'est pas déjà sur le PC
# split="balanced" : la version 47 classes (chiffres + lettres)
train = datasets.EMNIST(root="data", split="balanced", train=True, download=True)
test = datasets.EMNIST(root="data", split="balanced", train=False, download=True)

print("Train :", len(train), "images")
print("Test  :", len(test), "images")
print("Nombre de classes :", len(train.classes))
print("Classes :", train.classes)