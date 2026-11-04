from torchvision import datasets

train = datasets.MNIST(root="data", train=True, download=True)
test = datasets.MNIST(root="data", train=False, download=True)

print("Train :", len(train), "images")
print("Test  :", len(test), "images")

img, label = train[0]
print("Taille image :", img.size, "| label :", label)