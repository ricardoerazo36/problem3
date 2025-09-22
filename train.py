import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Configuración mejorada
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# 1. Preprocesamiento optimizado
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalizar a [-1, 1]
])

# 2. Dataset con batch size más eficiente
batch_size = 128  # Aumentado para mejor estabilidad
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# Sin num_workers para evitar problemas en Windows
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 3. Generator mejorado con mejor arquitectura
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding más pequeño pero efectivo
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Arquitectura más profunda pero eficiente
        self.fc = nn.Sequential(
            # Input: latent_dim + 50 (embedding)
            nn.Linear(latent_dim + 50, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
        
        # Inicialización de pesos
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, noise, labels):
        # Embedding de etiquetas
        label_embed = self.label_embedding(labels)
        # Concatenar ruido con embedding
        input_tensor = torch.cat([noise, label_embed], dim=1)
        output = self.fc(input_tensor)
        return output.view(-1, 1, 28, 28)

# 4. Discriminator mejorado
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
        # Embedding para etiquetas
        self.label_embedding = nn.Embedding(num_classes, 50)
        
        # Arquitectura más robusta
        self.fc = nn.Sequential(
            # Input: 28*28 + 50 (embedding)
            nn.Linear(28*28 + 50, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Inicialización de pesos
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, images, labels):
        # Aplanar imágenes
        images_flat = images.view(images.size(0), -1)
        # Embedding de etiquetas
        label_embed = self.label_embedding(labels)
        # Concatenar
        input_tensor = torch.cat([images_flat, label_embed], dim=1)
        return self.fc(input_tensor)

# 5. Función para generar y mostrar imágenes de muestra
def generate_sample_images(generator, epoch, device, fixed_noise, fixed_labels):
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise, fixed_labels)
        fake_images = (fake_images + 1) / 2  # Desnormalizar
        
        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i in range(10):
            row = i // 5
            col = i % 5
            axes[row, col].imshow(fake_images[i, 0].cpu().numpy(), cmap='gray')
            axes[row, col].set_title(f'Digit {fixed_labels[i].item()}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'samples_epoch_{epoch}.png', dpi=100, bbox_inches='tight')
        plt.close()
    generator.train()

# 6. Configuración del entrenamiento
latent_dim = 100
num_classes = 10
learning_rate = 0.0002
beta1 = 0.5  # Para Adam optimizer

# Crear modelos
generator = Generator(latent_dim, num_classes).to(DEVICE)
discriminator = Discriminator(num_classes).to(DEVICE)

print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

# Optimizadores con mejores hiperparámetros
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Función de pérdida
criterion = nn.BCELoss()

# Ruido fijo para generar imágenes de muestra
fixed_noise = torch.randn(10, latent_dim).to(DEVICE)
fixed_labels = torch.arange(10).to(DEVICE)

# Crear directorio para samples si no existe
os.makedirs('samples', exist_ok=True)

# 7. Bucle de entrenamiento optimizado
if __name__ == '__main__':
    epochs = 20  
    print(f"Iniciando entrenamiento por {epochs} épocas...")

    G_losses = []
    D_losses = []

    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        for i, (real_images, real_labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(DEVICE)
            real_labels = real_labels.to(DEVICE)
            
            # Etiquetas para pérdida
            real_target = torch.ones(batch_size, 1).to(DEVICE)
            fake_target = torch.zeros(batch_size, 1).to(DEVICE)
            
            # ============ Entrenar Discriminator ============
            discriminator.zero_grad()
            
            # Pérdida con imágenes reales
            real_output = discriminator(real_images, real_labels)
            d_loss_real = criterion(real_output, real_target)
            
            # Generar imágenes falsas
            noise = torch.randn(batch_size, latent_dim).to(DEVICE)
            fake_images = generator(noise, real_labels)
            
            # Pérdida con imágenes falsas
            fake_output = discriminator(fake_images.detach(), real_labels)
            d_loss_fake = criterion(fake_output, fake_target)
            
            # Pérdida total del discriminador
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()
            
            # ============ Entrenar Generator ============
            generator.zero_grad()
            
            # El generador quiere engañar al discriminador
            fake_output = discriminator(fake_images, real_labels)
            g_loss = criterion(fake_output, real_target)
            
            g_loss.backward()
            optimizer_G.step()
            
            # Acumular pérdidas
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            
            # Imprimir progreso cada 100 batches
            if i % 100 == 0:
                print(f'Época [{epoch+1}/{epochs}] Batch [{i}/{len(train_loader)}] '
                      f'D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}')
        
        # Promediar pérdidas de la época
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)
        
        print(f'Época [{epoch+1}/{epochs}] completada - Avg D_loss: {avg_d_loss:.4f} Avg G_loss: {avg_g_loss:.4f}')
        
        # Generar imágenes de muestra cada 5 épocas
        if (epoch + 1) % 5 == 0:
            generate_sample_images(generator, epoch + 1, DEVICE, fixed_noise, fixed_labels)
            print(f'Imágenes de muestra guardadas para época {epoch + 1}')

    # 8. Guardar modelo final
    torch.save(generator.state_dict(), "mnist_generator.pth")
    print("✅ Modelo guardado en mnist_generator.pth")

    # 9. Mostrar gráfico de pérdidas
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Pérdidas durante el entrenamiento')
    plt.grid(True)
    plt.savefig('training_losses.png')
    plt.show()

    print("🎉 Entrenamiento completado!")
    print(f"Modelo final guardado como: mnist_generator.pth")
    print(f"Gráfico de pérdidas guardado como: training_losses.png")