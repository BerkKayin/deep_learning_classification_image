## Importation bibliothèques
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Transformation des données avec augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Chargement des données d'entraînement et de test
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

print("Nombre d'échantillons d'entraînement : " + str(len(train_dataset)))
print("Nombre d'échantillons de test : " + str(len(test_dataset)))

# Charger les échantillons d'entraînement et de test dans le DataLoader
taille_lot = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=taille_lot, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=taille_lot, shuffle=False)

# Définir le modèle CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instancier le modèle
model = CNN()

# Définir l'optimiseur et la perte
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Entraîner le modèle
epochs = 50
loss_list = []
acc_list = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss_list.append(running_loss / len(train_loader))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    acc_list.append(accuracy)

    print(f'Époque : {epoch+1}/{epochs}, Perte : {running_loss/len(train_loader):.4f}, Précision : {accuracy:.2f}%')

# Afficher la courbe de perte
plt.plot(loss_list)
plt.xlabel("Nombre d'époques")
plt.ylabel("Perte totale")
plt.title("Perte")
plt.show()

# Afficher la courbe de précision
plt.plot(acc_list)
plt.xlabel("Nombre d'époques")
plt.ylabel("Précision totale")
plt.title("Précision")
plt.show()
