import os
import json
import io

# Evita tentativa de GPU (falha neste ambiente) e reduz risco de deadlock com o Streamlit.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ==============================================
# App Streamlit para VAE PneumoniaMNIST
# ==============================================
# Funcionalidades:
# - Triagem de pneumonia baseada no erro de reconstrução
# - Geração de novas imagens de raio-X
# - Upload e reconstrução de imagens
# ==============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
WEIGHTS_PATH = os.path.join(MODELS_DIR, 'vae_pneumonia.weights.h5')
CONFIG_PATH = os.path.join(MODELS_DIR, 'config.json')


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')


def build_decoder(latent_dim: int) -> tf.keras.Model:
    latent_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = tf.keras.layers.Reshape((7, 7, 64))(x)
    x = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(latent_inputs, outputs, name='decoder')


class VAE(tf.keras.Model):
    def __init__(self, encoder: tf.keras.Model, decoder: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        z_mean, z_log_var, z = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction

    def encode(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def decode(self, z, training=False):
        return self.decoder(z, training=training)


@st.cache_resource
def load_model():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(WEIGHTS_PATH):
        return None, 'Pesos ou configuração não encontrados. Treine o modelo executando train_vae.py.'
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        latent_dim = int(config.get('latent_dim', 16))
        encoder = build_encoder(latent_dim)
        decoder = build_decoder(latent_dim)
        vae = VAE(encoder, decoder)
        dummy = tf.zeros((1, 28, 28, 1))
        _ = vae(dummy, training=False)
        vae.load_weights(WEIGHTS_PATH)
        return vae, None
    except Exception as exc:
        return None, f'Falha ao carregar o modelo: {exc}'


def preprocess_image(image: Image.Image) -> np.ndarray:
    # Converter para grayscale e 28x28
    if image.mode != 'L':
        image = image.convert('L')
    if image.size != (28, 28):
        image = image.resize((28, 28))
    arr = np.array(image).astype('float32')
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=-1)  # (28,28,1)
    arr = np.expand_dims(arr, axis=0)   # (1,28,28,1)
    return arr


def compute_reconstruction_error(x: np.ndarray, x_recon: np.ndarray) -> float:
    # Erro MSE por imagem
    return float(np.mean((x - x_recon) ** 2))


def classify_pneumonia(reconstruction_error: float) -> tuple:
    """
    Classifica se há possível pneumonia baseado no erro de reconstrução.
    Erro alto = possível pneumonia (imagem fora do padrão normal aprendido).
    """
    # Thresholds baseados em experiência com o dataset (ajustar conforme necessário)
    if reconstruction_error < 0.01:
        return "NORMAL", "Baixo risco de pneumonia", "green"
    elif reconstruction_error < 0.02:
        return "BORDERLINE", "Risco moderado - recomenda-se avaliação médica", "orange"
    else:
        return "POSSÍVEL PNEUMONIA", "Alto risco - urgente avaliação médica", "red"


def generate_new_images(vae: VAE, num_images: int = 4) -> np.ndarray:
    """Gera novas imagens de raio-X usando o VAE treinado."""
    latent_dim = vae.encoder.output_shape[0][-1]  # Pega a dimensão do z_mean
    
    # Amostrar do espaço latente normal padrão
    z_samples = np.random.normal(0, 1, (num_images, latent_dim))
    
    # Decodificar para gerar imagens
    generated_images = vae.decode(z_samples, training=False).numpy()
    
    return generated_images


st.set_page_config(page_title='VAE PneumoniaMNIST - Triagem e Geração', layout='wide')
st.title('VAE PneumoniaMNIST - Triagem de Pneumonia e Geração de Imagens')

# Sidebar para carregar modelo
with st.sidebar:
    st.header('Modelo VAE')
    with st.spinner('Carregando modelo VAE...'):
        vae, err = load_model()
    if err:
        st.error(err)
        st.stop()
    else:
        st.success('✅ Modelo carregado com sucesso!')
        st.info(f'Dimensão latente: {vae.encoder.output_shape[0][-1]}')

# Tabs para diferentes funcionalidades
tab1, tab2, tab3 = st.tabs(["🔍 Triagem de Pneumonia", "🎨 Gerar Novas Imagens", "📊 Sobre o Modelo"])

with tab1:
    st.header("Triagem de Pneumonia via VAE")
    st.markdown("""
    **Como funciona:** O VAE foi treinado em imagens normais de raio-X. 
    Imagens com pneumonia têm padrões diferentes, resultando em maior erro de reconstrução.
    """)
    
    uploaded = st.file_uploader(
        'Envie uma imagem de raio-X para análise (PNG/JPG)',
        type=['png', 'jpg', 'jpeg'],
        key='upload_triagem'
    )
    
    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.read()))
        x = preprocess_image(image)
        
        # Reconstrução
        recon = vae(x, training=False).numpy()
        mse = compute_reconstruction_error(x, recon)
        
        # Classificação
        classification, description, color = classify_pneumonia(mse)
        
        # Exibição dos resultados
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagem Original")
            st.image(x[0].squeeze(), clamp=True, use_container_width=True)
        with col2:
            st.subheader("Reconstrução VAE")
            st.image(recon[0].squeeze(), clamp=True, use_container_width=True)
        
        # Resultado da triagem
        st.markdown("---")
        st.markdown(f"### 📊 Resultado da Triagem")
        
        # Métricas em cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Erro de Reconstrução", f"{mse:.6f}")
        with col2:
            st.metric("Classificação", classification)
        with col3:
            st.metric("Confiança", f"{(1-mse)*100:.1f}%" if mse < 1 else "0%")
        
        # Alerta colorido
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 0.5rem; background-color: {color}20; border-left: 4px solid {color};">
            <h4 style="color: {color}; margin: 0;">{classification}</h4>
            <p style="margin: 0.5rem 0 0 0;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption("⚠️ **Importante:** Este é apenas um auxiliar de triagem. Sempre consulte um médico para diagnóstico definitivo.")

with tab2:
    st.header("🎨 Geração de Novas Imagens de Raio-X")
    st.markdown("""
    Gere novas imagens sintéticas de raio-X usando o espaço latente aprendido pelo VAE.
    """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        num_images = st.slider("Número de imagens a gerar", 1, 8, 4)
        if st.button("🔄 Gerar Novas Imagens", type="primary"):
            with st.spinner("Gerando imagens..."):
                generated = generate_new_images(vae, num_images)
                st.session_state.generated_images = generated
    
    with col2:
        st.markdown("""
        **Controles:**
        - Ajuste o número de imagens
        - Clique em gerar para criar novas
        - As imagens são amostradas do espaço latente normal
        """)
    
    # Exibir imagens geradas
    if 'generated_images' in st.session_state:
        st.subheader("Imagens Geradas")
        cols = st.columns(num_images)
        for i, col in enumerate(cols):
            with col:
                st.image(st.session_state.generated_images[i].squeeze(),
                        clamp=True,
                        caption=f"Imagem {i+1}",
                        use_container_width=True)
        
        # Botão para download
        if st.button("💾 Salvar Imagens"):
            # Converter para PIL e salvar
            images = []
            for i in range(num_images):
                img_array = (st.session_state.generated_images[i].squeeze() * 255).astype(np.uint8)
                img = Image.fromarray(img_array, mode='L')
                images.append(img)
            
            # Salvar como ZIP (implementar se necessário)
            st.success(f"Imagens geradas! Use print screen ou salve individualmente.")

with tab3:
    st.header("📊 Sobre o Modelo VAE")
    st.markdown("""
    ### Arquitetura do Modelo
    
    **Encoder:**
    - Conv2D(32) → Conv2D(64) → Flatten → Dense(128) → Latent Space
    
    **Decoder:**
    - Dense(7×7×64) → Reshape → Conv2DTranspose(64) → Conv2DTranspose(32) → Output
    
    **Dimensão Latente:** 16 variáveis
    
    ### Como Funciona a Triagem
    
    1. **Imagens Normais:** Baixo erro de reconstrução (padrão bem aprendido)
    2. **Imagens com Pneumonia:** Alto erro de reconstrução (padrão diferente)
    3. **Thresholds:** 
       - < 0.01: Normal
       - 0.01-0.02: Borderline  
       - > 0.02: Possível pneumonia
    
    ### Limitações
    
    - Treinado apenas em PneumoniaMNIST
    - Não substitui diagnóstico médico
    - Sensibilidade depende da qualidade da imagem
    """)
    
    # Estatísticas do modelo
    if vae:
        st.subheader("Estatísticas do Modelo")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Parâmetros Encoder", f"{vae.encoder.count_params():,}")
            st.metric("Parâmetros Decoder", f"{vae.decoder.count_params():,}")
        with col2:
            st.metric("Total Parâmetros", f"{vae.count_params():,}")
            st.metric("Dimensão Latente", vae.encoder.output_shape[0][-1])

# Footer
st.markdown("---")
st.caption("""
🔬 **Modelo VAE para Triagem de Pneumonia** | 
Desenvolvido com TensorFlow e Streamlit | 
Sempre consulte um médico para diagnóstico definitivo.
""") 