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

# 3. Definir Generator (condicional - recibe el dígito como input)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Embedding para las clases (dígitos 0-9)
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),  # latent + label embedding
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # Combinar ruido con embedding de la etiqueta
        label_embed = self.label_embedding(labels)
        combined = torch.cat([z, label_embed], dim=1)
        out = self.fc(combined)
        return out.view(-1, 1, 28, 28)

# 4. Definir Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Embedding para las clases
        self.label_embedding = nn.Embedding(num_classes, 28*28)
        
        self.fc = nn.Sequential(
            nn.Linear(28*28 * 2, 512),  # imagen + label embedding
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # Aplanar imagen
        x = x.view(x.size(0), -1)
        # Embedding de etiqueta
        label_embed = self.label_embedding(labels)
        # Combinar
        combined = torch.cat([x, label_embed], dim=1)
        return self.fc(combined)

# 5. Configuración
latent_dim = 100
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

# 6. Optimizadores y función de pérdida
lr = 0.0002
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
criterion = nn.BCELoss()

# 7. Entrenamiento
epochs = 10
for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        batch_size = imgs.size(0)
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Etiquetas para real/fake
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # ============ Entrenar Discriminator ============
        optimizer_D.zero_grad()
        
        # Pérdida con imágenes reales
        real_outputs = discriminator(imgs, labels)
        real_loss = criterion(real_outputs, real_labels)
        
        # Generar imágenes falsas
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = generator(z, labels)
        
        # Pérdida con imágenes falsas
        fake_outputs = discriminator(fake_imgs.detach(), labels)
        fake_loss = criterion(fake_outputs, fake_labels)
        
        # Pérdida total del discriminador
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # ============ Entrenar Generator ============
        optimizer_G.zero_grad()
        
        # El generador quiere que el discriminador clasifique sus imágenes como reales
        outputs = discriminator(fake_imgs, labels)
        g_loss = criterion(outputs, real_labels)
        
        g_loss.backward()
        optimizer_G.step()
        
        if i % 200 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(train_loader)}] '
                  f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')

# 8. Guardar modelo
torch.save(generator.state_dict(), "mnist_generator.pth")
print("✅ Modelo guardado en mnist_generator.pth")