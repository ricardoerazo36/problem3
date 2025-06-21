import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
latent_dim = 100
num_classes = 10
batch_size = 128
epochs = 10

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        one_hot = F.one_hot(labels, num_classes).float()
        x = torch.cat([z, one_hot], dim=1)
        out = self.net(x)
        return out.view(-1, 1, 28, 28)

# Prepare MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize generator
G = Generator()
optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(epochs):
    for images, labels in loader:
        z = torch.randn(images.size(0), latent_dim)
        fake_imgs = G(z, labels)
        target = images.view(-1, 1, 28, 28)
        loss = loss_fn(fake_imgs, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save generator
torch.save(G.state_dict(), "mnist_generator.pth")
print("Model saved as mnist_generator.pth")