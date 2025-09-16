import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Definir clase Generator igual que en train.py (condicional)
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

# Cargar modelo entrenado
latent_dim = 100
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = Generator(latent_dim, num_classes).to(device)
    model.load_state_dict(torch.load("mnist_generator.pth", map_location="cpu"))
    model.eval()
    model_loaded = True
except FileNotFoundError:
    st.error("❌ No se encontró el archivo 'mnist_generator.pth'. Entrena el modelo primero ejecutando train.py")
    model_loaded = False

# Interfaz Streamlit
st.title("🔢 Generador de Dígitos Manuscritos (MNIST)")
st.write("Este generador usa una GAN condicional para crear dígitos manuscritos específicos.")

if model_loaded:
    st.success("✅ Modelo cargado correctamente")
    
    # Selector de dígito
    digit = st.selectbox("Elige un dígito:", options=list(range(10)))
    
    # Número de imágenes a generar
    num_images = st.slider("Número de imágenes a generar:", min_value=1, max_value=10, value=5)
    
    if st.button("🎲 Generar Imágenes"):
        with st.spinner("Generando imágenes..."):
            # Configurar la figura
            cols = min(num_images, 5)
            rows = (num_images + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
            
            # Asegurar que axes sea siempre 2D para indexar fácilmente
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = np.array([axes])
            elif cols == 1:
                axes = np.array([[ax] for ax in axes])
            
            for i in range(num_images):
                row = i // cols
                col = i % cols
                
                # Generar imagen
                z = torch.randn(1, latent_dim).to(device)
                labels = torch.tensor([digit]).to(device)
                
                with torch.no_grad():
                    gen_img = model(z, labels).cpu().numpy()[0, 0]
                    gen_img = (gen_img + 1) / 2

                # Mostrar imagen
                ax = axes[row, col]
                ax.imshow(gen_img, cmap="gray")
                ax.set_title(f"Dígito {digit} - #{i+1}")
                ax.axis("off")
            
            # Ocultar ejes sobrantes si los hay
            total_subplots = rows * cols
            for i in range(num_images, total_subplots):
                row = i // cols
                col = i % cols
                if rows == 1:
                    ax = axes[col] if cols > 1 else axes[0]
                else:
                    ax = axes[row, col]
                ax.axis("off")
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Información adicional
    st.markdown("---")
    st.info("""
    **ℹ️ Cómo funciona:**
    - Este generador usa una GAN (Red Generativa Adversarial) condicional
    - Puede generar dígitos específicos (0-9) combinando ruido aleatorio con la etiqueta del dígito
    - Cada generación produce imágenes únicas del mismo dígito
    """)
    
else:
    st.markdown("""
    ## 📋 Para usar esta aplicación:
    
    1. Ejecuta `train.py` para entrenar el modelo
    2. Asegúrate de que se genere el archivo `mnist_generator.pth`
    3. Recarga esta aplicación
    
    ```bash
    python train.py
    streamlit run app.py
    ```
    """)

# Mostrar información del sistema
with st.sidebar:
    st.markdown("## ⚙️ Información del Sistema")
    st.text(f"Device: {device}")
    st.text(f"PyTorch: {torch.__version__}")
    if model_loaded:
        st.text(f"Modelo: Cargado ✅")
        st.text(f"Parámetros del generador: {sum(p.numel() for p in model.parameters()):,}")
    else:
        st.text(f"Modelo: No encontrado ❌")