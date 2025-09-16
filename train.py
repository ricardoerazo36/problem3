import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Preprocesamiento
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizar a [-1, 1]
])

# 2. Descargar MNIST automáticamente en ./data
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. Definir un modelo generador simple
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        return out.view(-1, 1, 28, 28)

# 4. Instanciar modelo
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Generator(latent_dim).to(device)

# 5. Configurar entrenamiento
optimizer = optim.Adam(model.parameters(), lr=0.0002)
criterion = nn.MSELoss()

# 6. Entrenar (ejemplo rápido con 5 epochs)
epochs = 5
for epoch in range(epochs):
    for imgs, _ in train_loader:
        imgs = imgs.to(device)

        # vector de ruido
        z = torch.randn(imgs.size(0), latent_dim).to(device)

        # salida generada
        gen_imgs = model(z)

        # pérdida ficticia = comparar con imágenes reales (muy simplificado)
        loss = criterion(gen_imgs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

# 7. Guardar modelo
torch.save(model.state_dict(), "mnist_generator.pth")
print("✅ Modelo guardado en mnist_generator.pth")