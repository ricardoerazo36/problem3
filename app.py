import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Definir clase Generator igual que en train.py
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

# Cargar modelo entrenado
latent_dim = 100
model = Generator(latent_dim)
model.load_state_dict(torch.load("mnist_generator.pth", map_location="cpu"))
model.eval()

# Interfaz Streamlit
st.title("Generador de Dígitos Manuscritos (MNIST)")
st.write("Selecciona un dígito (0–9) y genera 5 imágenes (aleatorias).")

digit = st.number_input("Elige un dígito:", min_value=0, max_value=9, step=1)

if st.button("Generar"):
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        z = torch.randn(1, latent_dim)  # vector aleatorio
        gen_img = model(z).detach().numpy()[0, 0]
        axes[i].imshow(gen_img, cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)