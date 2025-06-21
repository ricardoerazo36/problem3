import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Generator model (same as training)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, z, labels):
        one_hot = F.one_hot(labels, 10).float()
        x = torch.cat([z, one_hot], dim=1)
        out = self.net(x)
        return out.view(-1, 1, 28, 28)

# Load model
model = Generator()
model.load_state_dict(torch.load("mnist_generator.pth", map_location="cpu"))
model.eval()

st.title("🧠 MNIST Handwritten Digit Generator")
digit = st.selectbox("Select a digit (0-9)", list(range(10)))
generate = st.button("Generate Images")

if generate:
    z = torch.randn(5, 100)
    labels = torch.tensor([digit]*5)
    with torch.no_grad():
        images = model(z, labels)

    st.write(f"Generated 5 samples of digit `{digit}`:")
    cols = st.columns(5)
    for i in range(5):
        fig, ax = plt.subplots()
        ax.imshow(images[i][0], cmap="gray")
        ax.axis("off")
        cols[i].pyplot(fig)
