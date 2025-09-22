import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configurar página
st.set_page_config(
    page_title="Generador MNIST",
    page_icon="🔢",
    layout="centered"
)

# Definir clase Generator (debe coincidir exactamente con train.py)
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

@st.cache_resource
def load_model():
    """Cargar modelo con cache para evitar recargas innecesarias"""
    latent_dim = 100
    num_classes = 10
    device = torch.device("cpu")  # Usar CPU para inferencia en Streamlit
    
    try:
        model = Generator(latent_dim, num_classes).to(device)
        model.load_state_dict(torch.load("mnist_generator.pth", map_location="cpu"))
        model.eval()
        return model, device, True
    except FileNotFoundError:
        return None, device, False

def generate_images(model, device, digit, num_images=5, seed=None):
    """Generar imágenes con seed opcional para reproducibilidad"""
    if seed is not None:
        torch.manual_seed(seed)
    
    latent_dim = 100
    
    # Generar ruido y etiquetas
    noise = torch.randn(num_images, latent_dim).to(device)
    labels = torch.tensor([digit] * num_images).to(device)
    
    # Generar imágenes
    with torch.no_grad():
        generated_images = model(noise, labels)
        # Desnormalizar de [-1, 1] a [0, 1]
        generated_images = (generated_images + 1) / 2
        
    return generated_images.cpu().numpy()

# Interfaz principal
st.title("🔢 Generador de Dígitos Manuscritos MNIST")
st.markdown("### Generador GAN Condicional entrenado desde cero")

# Cargar modelo
model, device, model_loaded = load_model()

if model_loaded:
    st.success("✅ Modelo cargado correctamente")
    
    # Controles en la barra lateral
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Selector de dígito
        digit = st.selectbox(
            "🎯 Elige un dígito:",
            options=list(range(10)),
            index=0
        )
        
        # Número de imágenes
        num_images = st.slider(
            "🖼️ Número de imágenes:",
            min_value=1,
            max_value=10,
            value=5
        )
        
        # Seed para reproducibilidad
        use_seed = st.checkbox("🎲 Usar seed fijo")
        seed = None
        if use_seed:
            seed = st.number_input("Seed:", min_value=0, value=42)
        
        # Información del sistema
        st.markdown("---")
        st.markdown("**💻 Sistema:**")
        st.text(f"Device: {device}")
        st.text(f"PyTorch: {torch.__version__}")
        
        if model_loaded:
            total_params = sum(p.numel() for p in model.parameters())
            st.text(f"Parámetros: {total_params:,}")
    
    # Área principal
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button(
            "🎨 Generar Imágenes", 
            type="primary", 
            use_container_width=True
        )
    
    if generate_button:
        with st.spinner(f"🎨 Generando {num_images} imágenes del dígito {digit}..."):
            try:
                # Generar imágenes
                images = generate_images(model, device, digit, num_images, seed)
                
                # Mostrar imágenes
                st.markdown(f"### 🎯 Dígito generado: **{digit}**")
                
                # Configurar layout según número de imágenes
                if num_images <= 5:
                    cols = st.columns(num_images)
                    for i in range(num_images):
                        with cols[i]:
                            fig, ax = plt.subplots(figsize=(3, 3))
                            ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
                            ax.set_title(f"#{i+1}", fontsize=14, fontweight='bold')
                            ax.axis("off")
                            st.pyplot(fig, use_container_width=True)
                            plt.close()
                else:
                    # Para más de 5 imágenes, usar una cuadrícula
                    cols_per_row = 5
                    rows = (num_images + cols_per_row - 1) // cols_per_row
                    
                    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 3*rows))
                    
                    if rows == 1:
                        axes = [axes]
                    
                    for i in range(num_images):
                        row = i // cols_per_row
                        col = i % cols_per_row
                        
                        if rows == 1:
                            ax = axes[col] if cols_per_row > 1 else axes
                        else:
                            ax = axes[row][col]
                        
                        ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
                        ax.set_title(f"#{i+1}", fontsize=12, fontweight='bold')
                        ax.axis("off")
                    
                    # Ocultar ejes vacíos
                    total_plots = rows * cols_per_row
                    for i in range(num_images, total_plots):
                        row = i // cols_per_row
                        col = i % cols_per_row
                        if rows == 1:
                            ax = axes[col] if cols_per_row > 1 else axes
                        else:
                            ax = axes[row][col]
                        ax.axis("off")
                    
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                
                # Mostrar estadísticas
                avg_pixel_value = np.mean(images)
                st.info(f"📊 Valor promedio de píxeles: {avg_pixel_value:.3f}")
                
            except Exception as e:
                st.error(f"❌ Error al generar imágenes: {str(e)}")
    
    # Información adicional
    with st.expander("ℹ️ Información sobre el modelo", expanded=False):
        st.markdown("""
        **🧠 Arquitectura:**
        - **Tipo:** GAN Condicional (Conditional GAN)
        - **Dataset:** MNIST (dígitos manuscritos 0-9)
        - **Resolución:** 28×28 píxeles, escala de grises
        - **Condicionamiento:** Embedding de etiquetas de clase
        
        **⚡ Características:**
        - Entrenado completamente desde cero (sin pesos preentrenados)
        - Arquitectura optimizada para generar dígitos específicos
        - Batch normalization para estabilidad del entrenamiento
        - Inicialización de pesos mejorada
        
        **🎨 Uso:**
        - Selecciona el dígito que quieres generar (0-9)
        - Ajusta el número de imágenes a generar
        - Cada generación produce variaciones únicas del mismo dígito
        - Usa un seed fijo para reproducir resultados exactos
        """)
    
    # Sección de pruebas rápidas
    st.markdown("---")
    st.markdown("### 🚀 Prueba Rápida")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎲 Generar todos los dígitos (0-9)", use_container_width=True):
            with st.spinner("Generando todos los dígitos..."):
                fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                for digit_test in range(10):
                    test_images = generate_images(model, device, digit_test, 1, 42)
                    row = digit_test // 5
                    col = digit_test % 5
                    axes[row, col].imshow(test_images[0, 0], cmap="gray", vmin=0, vmax=1)
                    axes[row, col].set_title(f"Dígito {digit_test}", fontweight='bold')
                    axes[row, col].axis("off")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    
    with col2:
        if st.button("📊 Análisis de variabilidad", use_container_width=True):
            with st.spinner("Analizando variabilidad..."):
                # Generar múltiples versiones del mismo dígito
                test_digit = digit
                multiple_images = generate_images(model, device, test_digit, 6)
                
                fig, axes = plt.subplots(2, 3, figsize=(9, 6))
                for i in range(6):
                    row = i // 3
                    col = i % 3
                    axes[row, col].imshow(multiple_images[i, 0], cmap="gray", vmin=0, vmax=1)
                    axes[row, col].set_title(f"{test_digit} - Var #{i+1}")
                    axes[row, col].axis("off")
                
                plt.suptitle(f"Variabilidad del dígito {test_digit}", fontsize=16, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

else:
    # Instrucciones cuando el modelo no está disponible
    st.error("❌ No se encontró el archivo 'mnist_generator.pth'")
    
    st.markdown("""
    ## 🔧 Para usar esta aplicación:
    
    ### Paso 1: Entrenar el modelo
    ```bash
    python train.py
    ```
    
    ### Paso 2: Ejecutar la aplicación
    ```bash
    streamlit run app.py
    ```
    
    ### 📋 Requisitos:
    - Los archivos `train.py` y `app.py` deben estar en la misma carpeta
    - Se generará automáticamente el dataset MNIST en la carpeta `./data/`
    - El entrenamiento puede tomar algunos minutos dependiendo de tu hardware
    
    ### 🎯 Características del modelo:
    - **Entrenamiento desde cero** (sin pesos preentrenados)
    - **GAN Condicional** para generar dígitos específicos
    - **Optimizado** para mejor convergencia y calidad
    """)
    
    # Mostrar información de archivos requeridos
    current_dir = Path(".")
    
    st.markdown("### 📁 Estado de archivos:")
    
    files_status = [
        ("train.py", "Archivo de entrenamiento"),
        ("app.py", "Esta aplicación"),
        ("mnist_generator.pth", "Modelo entrenado"),
        ("requirements.txt", "Dependencias")
    ]
    
    for filename, description in files_status:
        file_path = current_dir / filename
        if file_path.exists():
            st.success(f"✅ {filename} - {description}")
        else:
            st.warning(f"⚠️ {filename} - {description} (No encontrado)")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🤖 Generador MNIST con GAN Condicional | Entrenado desde cero con PyTorch"
    "</div>", 
    unsafe_allow_html=True
)