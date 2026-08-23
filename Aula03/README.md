# 🔬 VAE PneumoniaMNIST - Triagem de Pneumonia e Geração de Imagens

Um projeto completo de **Variational Autoencoder (VAE)** treinado no dataset PneumoniaMNIST para triagem de pneumonia e geração de imagens sintéticas de raio-X.

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Funcionalidades](#funcionalidades)
- [Arquitetura do Modelo](#arquitetura-do-modelo)
- [Instalação](#instalação)
- [Uso](#uso)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Dataset](#dataset)
- [Como Funciona a Triagem](#como-funciona-a-triagem)
- [Limitações](#limitações)
- [Contribuição](#contribuição)
- [Licença](#licença)

## 🎯 Visão Geral

Este projeto implementa um **Variational Autoencoder (VAE)** usando TensorFlow/Keras para:

1. **Triagem Automática de Pneumonia**: Analisa imagens de raio-X e classifica o risco de pneumonia baseado no erro de reconstrução
2. **Geração de Imagens Sintéticas**: Cria novas imagens de raio-X usando o espaço latente aprendido
3. **Interface Web Interativa**: App Streamlit para upload, análise e geração de imagens

O VAE aprende representações latentes de imagens normais de raio-X e usa o erro de reconstrução como indicador de anomalia (possível pneumonia).

## ✨ Funcionalidades

### 🔍 Triagem de Pneumonia
- Upload de imagens de raio-X (PNG/JPG)
- Conversão automática para 28x28 grayscale
- Classificação automática:
  - **NORMAL** (verde): Baixo risco
  - **BORDERLINE** (laranja): Risco moderado
  - **POSSÍVEL PNEUMONIA** (vermelho): Alto risco
- Visualização lado a lado: original vs reconstrução
- Métricas de confiança e erro de reconstrução

### 🎨 Geração de Imagens
- Geração de 1-8 imagens sintéticas
- Amostragem do espaço latente normal
- Interface interativa com slider
- Opção de salvar imagens geradas

### 📊 Informações do Modelo
- Estatísticas detalhadas (parâmetros, arquitetura)
- Explicação do funcionamento
- Limitações e avisos importantes

## 🏗️ Arquitetura do Modelo

### Encoder
```
Input (28, 28, 1) → Conv2D(32, 3x3, stride=2) → Conv2D(64, 3x3, stride=2) 
→ Flatten → Dense(128) → Dense(latent_dim) [z_mean, z_log_var]
```

### Sampling Layer
```
z = z_mean + exp(0.5 * z_log_var) * ε, onde ε ~ N(0,1)
```

### Decoder
```
Input (latent_dim) → Dense(7×7×64) → Reshape(7,7,64) 
→ Conv2DTranspose(64, 3x3, stride=2) → Conv2DTranspose(32, 3x3, stride=2) 
→ Conv2DTranspose(1, 3x3, activation='sigmoid') → Output (28, 28, 1)
```

### Hiperparâmetros
- **Dimensão Latente**: 16
- **Batch Size**: 128
- **Épocas**: 20
- **Learning Rate**: 1e-3
- **Total de Parâmetros**: ~100K

## 🚀 Instalação

### Pré-requisitos
- Python 3.11
- pip
- Ambiente virtual (recomendado)

### 1. Clone o Repositório
```bash
git clone <repository-url>
cd Aula03
```

### 2. Crie e Ative o Ambiente Virtual
```bash
# Criar ambiente virtual com Python 3.11
python3.11 -m venv .venv

# Ativar (Linux/Mac)
source .venv/bin/activate

# Ativar (Windows)
.venv\Scripts\activate
```

### 3. Instale as Dependências
```bash
# Atualizar pip
python -m pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt
```

## 📖 Uso

### 1. Treinar o Modelo VAE

```bash
# Treinar o VAE no dataset PneumoniaMNIST
python train_vae.py
```

**O que acontece:**
- Download automático do dataset PneumoniaMNIST
- Treinamento por 20 épocas
- Validação no conjunto de teste
- Salvamento dos pesos em `models/vae_pneumonia.weights.h5`
- Geração de figura de reconstruções em `outputs/reconstructions.png`

**Saída esperada:**
```
Carregando PneumoniaMNIST...
Treino: (4708, 28, 28, 1), Validação: (524, 28, 28, 1)
Iniciando treinamento...
Epoch 1/20
...
Salvando pesos em: models/vae_pneumonia.weights.h5
Gerando figura de reconstruções em: outputs/reconstructions.png
Concluído.
```

### 2. Executar o App Streamlit

```bash
# Iniciar interface web
streamlit run app.py
```

**Acesso:**
- URL local: `http://localhost:8501`
- Interface organizada em 3 abas principais

## 📁 Estrutura do Projeto

```
Aula03/
├── train_vae.py          # Script de treinamento do VAE
├── app.py               # App Streamlit para interface web
├── requirements.txt     # Dependências Python
├── README.md           # Este arquivo
├── models/             # Modelos treinados (criado após treino)
│   ├── vae_pneumonia.weights.h5
│   └── config.json
└── outputs/            # Saídas do treinamento (criado após treino)
    └── reconstructions.png
```

## 🗃️ Dataset

### PneumoniaMNIST
- **Fonte**: [MedMNIST](https://medmnist.com/)
- **Tamanho**: 5.232 imagens (4.708 treino + 524 validação)
- **Resolução**: 28×28 pixels
- **Canais**: 1 (grayscale)
- **Classes**: Normal vs Pneumonia
- **Download**: Automático via `medmnist` package

### Pré-processamento
- Conversão para float32
- Normalização para [0, 1]
- Garantia de shape (28, 28, 1)
- Data augmentation via shuffling

## 🔬 Como Funciona a Triagem

### Princípio do VAE para Detecção de Anomalias

1. **Treinamento**: O VAE aprende a representar imagens normais de raio-X no espaço latente
2. **Reconstrução**: Imagens de entrada são codificadas e decodificadas
3. **Erro de Reconstrução**: Imagens normais têm baixo erro, anormais têm alto erro
4. **Classificação**: Thresholds baseados no erro determinam o risco

### Thresholds de Classificação
```python
if reconstruction_error < 0.01:
    return "NORMAL"           # Baixo risco
elif reconstruction_error < 0.02:
    return "BORDERLINE"       # Risco moderado
else:
    return "POSSÍVEL PNEUMONIA"  # Alto risco
```

### Métricas de Performance
- **Erro de Reconstrução (MSE)**: Indicador principal de anomalia
- **Confiança**: `(1 - MSE) × 100%` (quando MSE < 1)
- **Classificação Automática**: Baseada nos thresholds

## ⚠️ Limitações

### Técnicas
- **Dataset Limitado**: Treinado apenas em PneumoniaMNIST
- **Resolução Baixa**: Imagens 28×28 podem perder detalhes importantes
- **Generalização**: Performance pode variar em outros datasets de raio-X
- **Thresholds Fixos**: Valores baseados em experiência, não otimizados

### Médicas
- **Não é Diagnóstico**: Apenas auxiliar de triagem
- **Falsos Positivos/Negativos**: Possíveis erros de classificação
- **Qualidade da Imagem**: Performance depende da qualidade do upload
- **Sempre Consultar Médico**: Para diagnóstico definitivo

## 🤝 Contribuição

### Como Contribuir
1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

### Áreas de Melhoria
- [ ] Otimização automática dos thresholds de classificação
- [ ] Suporte a diferentes resoluções de imagem
- [ ] Métricas de performance mais robustas
- [ ] Interface para ajuste de hiperparâmetros
- [ ] Exportação de relatórios em PDF
- [ ] Integração com outros datasets médicos

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 🙏 Agradecimentos

- **MedMNIST**: Dataset PneumoniaMNIST
- **TensorFlow/Keras**: Framework de deep learning
- **Streamlit**: Interface web interativa
- **FIAP**: Disciplina de Generative AI Advanced Networks

## 📞 Suporte

Para dúvidas, problemas ou sugestões:
- Abra uma [Issue](../../issues) no GitHub
- Entre em contato com os desenvolvedores
- Consulte a documentação do TensorFlow e Streamlit

---

**⚠️ Aviso Médico**: Este projeto é apenas para fins educacionais e de pesquisa. **NUNCA** use para diagnóstico médico real. Sempre consulte um profissional de saúde qualificado.

**🔬 Desenvolvido para**: FIAP - Disciplina de Generative AI Advanced Networks - Aula 03 