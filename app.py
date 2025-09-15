import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Generador de Dígitos Manuscritos (MNIST)")
st.write("Selecciona un dígito (0–9) y genera imágenes (placeholder con ruido).")

digit = st.number_input("Elige un dígito:", min_value=0, max_value=9, step=1)

if st.button("Generar"):
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    for i in range(5):
        # Solo ruido aleatorio para probar
        img = np.random.rand(28, 28)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
