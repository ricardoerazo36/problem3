import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import streamlit as st

# Definir la misma clase Generator que usaste en train.py
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