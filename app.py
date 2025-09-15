import streamlit as st
import torch
import matplotlib.pyplot as plt

# Cargar tu modelo entrenado 
# model = torch.load("mnist_generator.pth")
# model.eval()

st.title("Generador de Dígitos Manuscritos (MNIST)")
st.write("Selecciona un dígito (0–9) y genera imágenes.")

# Input: dígito a generar
digit = st.number_input("Elige un dígito:", min_value=0, max_value=9, step=1)

if st.button("Generar"):
    # Aquí generaremos 5 imágenes con el modelo
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        # Por ahora, placeholder: ruido aleatorio en vez de modelo
        img = torch.rand(28, 28).numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
