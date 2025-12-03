# 🫀 Heart Disease ML Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

## 👥 Equipe de Desenvolvimento

- **André Luiz G. C. da Fonseca** - algcf@cesar.school
- **Gabriel C. G. P. Farias** - gcgpf@cesar.school
- **João Vitor M. Fittipaldi** - jvmf@cesar.school
- **Maria Júlia O. T. Menezes** - mjotm@cesar.school
- **Maria Luísa C. Lima** - mlcl@cesar.school

---

## 📄 Abstract

This work presents the development of a complete architecture for predicting ischemic cardiovascular diseases, built from the reproduction and expansion of the study *"Enhancing Prognosis Accuracy for Ischemic Cardiovascular Disease Using K Nearest Neighbor Algorithm: A Robust Approach"*. The solution was structured as a fully integrated and containerized data pipeline, involving an ingestion API developed with FastAPI, distributed storage using MinIO, a modeling environment in JupyterLab, and experiment tracking with MLFlow. Several algorithms were analyzed, including KNN, Random Forest, Gradient Boosting, among others, with the objective of comparing performance and validating the original methodology. The experimental analysis demonstrated that the optimized Gradient Boosting model offered the best balance between accuracy, stability, and predictive capability, standing out as the most suitable approach for the clinical scenario addressed.

**Artigo Original:** [IEEE Xplore](https://ieeexplore.ieee.org/document/10239171)

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    HEART DISEASE ML PIPELINE                 │
└─────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │   Raw Data   │
    │  (heart.csv) │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   FastAPI    │◀──── HTTP Requests
    │  (Ingestão)  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────────────┐
    │         MinIO (S3-like)          │
    │  • Raw data                      │
    │  • Processed data                │
    │  • Trained models (.pkl)         │
    │  • Model metadata                │
    └──────┬───────────────────────────┘
           │
           ├──────────────────┐
           │                  │
           ▼                  ▼
    ┌─────────────┐    ┌─────────────┐
    │ PostgreSQL  │    │  JupyterLab │
    │ (Metadados) │    │ (Notebooks) │
    └──────┬──────┘    └──────┬──────┘
           │                  │
           │                  │ • 01_exploratory_analysis.ipynb
           │                  │ • 02_preprocessing.ipynb
           │                  │ • 03_model_training.ipynb
           │                  │ • 04_model_evaluation.ipynb
           │                  │ • 05_predictions.ipynb
           │                  │
           │                  ▼
           │           ┌─────────────┐
           │           │   MLflow    │
           └──────────▶│ (Tracking)  │
                       └──────┬──────┘
                              │
                              │ Model Registry
                              │ Experiment Logs
                              │
                              ▼
                       ┌─────────────┐
                       │  Production │
                       │    Model    │
                       └──────┬──────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │    Streaming    │
                       │   Simulator     │
                       └──────┬──────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  ThingsBoard    │
                       │   (Dashboard)   │
                       └─────────────────┘
```

---

## 🚀 Início Rápido

### 📋 Pré-requisitos

- **Docker** 20.10+ e **Docker Compose** 2.0+
- **Python** 3.11+
- **Git**
- 8GB RAM mínimo
- 15GB espaço em disco

### 🐧 Instalação - Linux

#### 1. Clone o Repositório

```bash
git clone https://github.com/mjuliamenezes/heart-disease-ml-pipeline.git
cd heart-disease-ml-pipeline
```

#### 2. Configure Variáveis de Ambiente

```bash
# Criar arquivo .env na raiz do projeto
cat > .env << 'EOF'
# DATABASE CREDENTIALS
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mlflow_db

# MINIO (S3-compatible) - Object Storage Local
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_BUCKET=ml-bucket-heart

# MLFLOW
MLFLOW_TRACKING_URI=http://mlflow:5000

# FASTAPI
API_HOST=0.0.0.0
API_PORT=8000

# THINGSBOARD
TB_USERNAME=tenant@thingsboard.org
TB_PASSWORD=tenant
THINGSBOARD_TOKEN=ozqbzirn1y9q3j197m6i

# STREAMING
STREAM_INTERVAL_SECONDS=5
EOF
```

#### 3. Build e Inicialização dos Containers

```bash
# Build das imagens
docker compose build

# Subir todos os serviços
docker compose up -d

# Verificar status (aguardar todos ficarem "healthy")
docker compose ps
```

#### 4. Testar Pipeline Base

```bash
# Dar permissão de execução
chmod +x scripts/test_pipeline.sh

# Executar testes
./scripts/test_pipeline.sh
```

#### 5. Carregar Dados Iniciais no MinIO

```bash
# Dividir dados em treino/teste/validação
python3 scripts/split_data.py

# Upload para MinIO
python3 scripts/upload_to_minio.py
```

#### 6. Executar Notebooks de Treinamento

Acessar JupyterLab em http://localhost:8888 e executar na ordem:

1. `notebooks/01_exploratory_analysis.ipynb` - Ingestão e validação dos dados
2. `notebooks/02_preprocessing.ipynb` - Análise exploratória
3. `notebooks/03_model_training.ipynb` - Treinamento dos modelos base
4. `notebooks/04_model_evaluation.ipynb` - Avaliação e comparação
5. `notebooks/05_predictions.ipynb` - Otimização e modelo de produção

#### 7. Testar API de Predição

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 54,
    "sex": 1,
    "chest_pain_type": 3,
    "resting_bp": 150,
    "cholesterol": 195,
    "fasting_bs": 0,
    "resting_ecg": 0,
    "max_hr": 122,
    "exercise_angina": 0,
    "oldpeak": 0.0,
    "st_slope": 1
  }'
```

#### 8. Configurar Tabela de Predições no PostgreSQL

```bash
# Copiar script SQL para o container
docker cp scripts/add_predictions_table.sql heart-postgres:/tmp/

# Executar script
docker compose exec postgres psql -U postgres -d mlflow_db -f /tmp/add_predictions_table.sql

# Verificar criação das tabelas
docker compose exec postgres psql -U postgres -d mlflow_db -c "\dt heart_disease.*"
```

#### 9. Subir Streaming Simulator

```bash
# Build do container de streaming
docker compose build streaming

# Executar streaming (modo daemon)
docker compose --profile streaming up streaming
```

#### 10. Testar Streaming com Visualização em Tempo Real

```bash
# Processar 20 amostras com delay de 0.5s
docker compose run --rm streaming python stream_simulator.py \
  --delay 0.5 \
  --max-samples 20 \
  --no-api

# Acessar dashboard ThingsBoard: http://localhost:8080
```

---

### 🪟 Instalação - Windows

#### 1. Instalar Pré-requisitos

- Instalar [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
- Instalar [Python 3.11+](https://www.python.org/downloads/)
- Instalar [Git for Windows](https://git-scm.com/download/win)

#### 2. Clone o Repositório

```powershell
git clone https://github.com/mjuliamenezes/heart-disease-ml-pipeline.git
cd heart-disease-ml-pipeline
```

#### 3. Configure Variáveis de Ambiente

Criar arquivo `.env` na raiz do projeto com o seguinte conteúdo:

```env
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mlflow_db
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin123
MINIO_BUCKET=ml-bucket-heart
MLFLOW_TRACKING_URI=http://mlflow:5000
API_HOST=0.0.0.0
API_PORT=8000
TB_USERNAME=tenant@thingsboard.org
TB_PASSWORD=tenant
THINGSBOARD_TOKEN=ozqbzirn1y9q3j197m6i
STREAM_INTERVAL_SECONDS=5
```

#### 4. Build e Inicialização

```powershell
# Build
docker compose build

# Subir serviços
docker compose up -d

# Verificar status
docker compose ps
```

#### 5. Carregar Dados

```powershell
python scripts\split_data.py
python scripts\upload_to_minio.py
```

#### 6-10. Seguir os mesmos passos do Linux

Os comandos `docker compose`, `curl` e acesso aos notebooks são idênticos no Windows.

---
## 🎯 Dashboard ThingsBoard

O dashboard fornece monitoramento em tempo real das predições. Siga o passo a passo abaixo para configurar.

### 📋 Configuração Inicial do ThingsBoard

#### Passo 1: Primeiro Acesso

```bash
# Aguardar ThingsBoard inicializar (pode levar 2-3 minutos)
docker compose logs -f thingsboard | grep "Started"

# Quando aparecer "ThingsBoard Started", acessar:
# http://localhost:8080
```

**Login Inicial:**
- Email: `tenant@thingsboard.org`
- Password: `tenant`

#### Passo 2: Criar Device

1. No menu lateral esquerdo, clique em **"Devices"**
2. Clique no botão **"+"** (Add device) no canto superior direito
3. Preencha:
   - **Name**: `heart-disease-predictions`
   - **Device profile**: Deixar `default`
4. Clique em **"Add"**

#### Passo 3: Obter Access Token

1. Na lista de devices, clique no device **`heart-disease-predictions`**
2. Clique na aba **"Details"**
3. Procure por **"Copy access token"** e clique para copiar
4. **Atualize o arquivo `.env`** com o token copiado:

```bash
# Editar .env
nano .env

# Substituir a linha:
THINGSBOARD_TOKEN=seu_token_aqui_copiado_do_thingsboard
```

5. **Reiniciar o streaming** para usar o novo token:

```bash
docker compose restart streaming
```

#### Passo 4: Verificar Telemetria 

Antes de criar o dashboard, **execute um teste de streaming** para garantir que os dados estão chegando:

```bash
# Executar teste com 5 amostras
docker compose run --rm streaming python stream_simulator.py \
  --delay 1 \
  --max-samples 5 \
  --no-api
```

**Verificar dados:**
1. No ThingsBoard, vá em **Devices** → **heart-disease-predictions**
2. Clique na aba **"Latest telemetry"**
3. Você deve ver as seguintes chaves:
   - `patient_id`
   - `prediction`
   - `probability`
   - `true_label`
   - `is_correct`
   - `total_predictions`
   - `correct_predictions`
   - `accuracy`

**Se não aparecer nada, verifique:**
- Token está correto no `.env`
- Streaming foi reiniciado após trocar o token
- ThingsBoard está rodando: `docker compose ps thingsboard`

---

### 🎨 Criando o Dashboard (Passo a Passo Detalhado)

#### Passo 1: Criar Dashboard Vazio

1. No menu lateral, clique em **"Dashboards"**
2. Clique no botão **"+"** (Add dashboard)
3. Preencha:
   - **Title**: `Heart Disease ML - Real-time Predictions`
   - **Description**: `Monitoramento em tempo real de predições de doenças cardíacas`
4. Clique em **"Add"**
5. **Abra o dashboard criado** clicando nele
6. Clique em **"Enter edit mode"** (ícone de lápis no canto superior direito)

---

#### Widget 1: Card - Total de Predições

1. Clique em **"+ Add new widget"**
2. Selecione **"Cards"** → **"Simple card"**
3. Clique em **"Add"**

**Configurar Data:**
1. Aba **"Data"**:
   - **Entity alias**: Clique em **"+ Create new"**
   - Name: `Device`
   - Filter type: `Single entity`
   - Type: `Device`
   - Device: Selecione `heart-disease-predictions`
   - Clique **"Add"**
2. **Datasource**: Selecione `Device`
3. **Data key**: Clique em **"+"**
   - Type: `Timeseries`
   - Key: Digite `total_predictions` (ou selecione da lista)
   - Label: `Total`
4. Clique em **"Add"**

---

#### Widget 2: Card - Predições Corretas

1. **"+ Add new widget"** → **"Cards"** → **"Simple card"**
2. **Data**:
   - Entity alias: `Device` (já existe)
   - Key: `correct_predictions`, Label: `Corretas`

---

#### Widget 3: Card - Acurácia Atual 

1. **"+ Add new widget"** → **"Cards"** → **"Simple card"**
2. **Data**:
   - Entity alias: `Device`
   - Key: `accuracy`, Label: `Acurácia`

---

#### Widget 4: Gauge - Acurácia Visual

1. **"+ Add new widget"** → **"Analog gauges"** → **"Radial gauge"**
2. **Data**:
   - Entity alias: `Device`
   - Key: `accuracy`, Label: `Acurácia`
3. **Settings**:
   - Min value: `0`
   - Max value: `100`
   - Units: `%`

---

#### Widget 5: Timeline - Predições vs Real

1. **"+ Add new widget"** → **"Charts"** → **"Timeseries - Line chart"**
2. **Data**:
   - Entity alias: `Device`
   - Adicionar 3 keys:
     1. `prediction` - Label: `Predição`, Color: `#2196F3`
     2. `true_label` - Label: `Label Real`, Color: `#4CAF50`
     3. `is_correct` - Label: `Correto?`, Color: `#FF5722`

---

#### Widget 6: Gráfico de Probabilidade

1. **"+ Add new widget"** → **"Charts"** → **"Timeseries - Line chart"**
2. **Data**:
   - Entity alias: `Device`
   - Key: `probability`, Label: `Probabilidade (%)`

---

#### Widget 7: Tabela de Últimas Predições 

1. **"+ Add new widget"** → **"Tables"** → **"Timeseries table"**
2. **Data**:
   - Entity alias: `Device`
   - Adicionar keys na ordem:
     1. `patient_id` - Label: `ID Paciente`
     2. `prediction` - Label: `Predição`
     3. `true_label` - Label: `Label Real`
     4. `probability` - Label: `Probabilidade (%)`
     5. `is_correct` - Label: `Correto?`
3. **Settings**:
   - **Pagination**:
     - Enable pagination: 
     - Default page size: `10`

---

#### Passo 3: Salvar e Testar

1. Clique em **"Apply changes"** (ícone de disquete)
2. Clique em **"Exit edit mode"** (ícone de olho)

**Testar com dados reais:**

```bash
# Terminal 1: Executar streaming
docker compose run --rm streaming python stream_simulator.py \
  --delay 0.5 \
  --max-samples 30 \
  --no-api

# Terminal 2 (opcional): Ver logs
docker compose logs -f streaming
```

**No navegador:**
- Mantenha o dashboard aberto
- Os widgets devem atualizar em tempo real
- Você verá os gráficos se preenchendo conforme os dados chegam

---

## 🔗 Acesso aos Serviços

| Serviço | URL | Credenciais |
|---------|-----|-------------|
| **JupyterLab** | http://localhost:8888 | Sem senha |
| **MLflow** | http://localhost:5000 | - |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin123 |
| **FastAPI (Swagger)** | http://localhost:8000/docs | - |
| **ThingsBoard** | http://localhost:8080 | tenant@thingsboard.org / tenant |
| **PostgreSQL** | localhost:5432 | postgres / postgres |

---

## 📊 Dataset

**Heart Disease Dataset (Comprehensive)** - Kaggle

- **Fonte:** https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final
- **Registros:** 1190 pacientes
- **Features:** 11 atributos clínicos
- **Target:** Presença de doença cardíaca (0=Saudável, 1=Doença)

### Atributos

| Atributo | Tipo | Descrição |
|----------|------|-----------|
| age | int | Idade do paciente |
| sex | int | Sexo (1=M, 0=F) |
| chest pain type | int | Tipo de dor no peito (0-3) |
| resting bp s | int | Pressão arterial em repouso (mm Hg) |
| cholesterol | int | Colesterol sérico (mg/dl) |
| fasting blood sugar | int | Glicemia em jejum > 120 mg/dl (1=sim, 0=não) |
| resting ecg | int | Resultado ECG em repouso (0-2) |
| max heart rate | int | Frequência cardíaca máxima alcançada |
| exercise angina | int | Angina induzida por exercício (1=sim, 0=não) |
| oldpeak | float | Depressão ST induzida por exercício |
| ST slope | int | Inclinação do segmento ST (0-2) |

---

## 🤖 Modelos Implementados

### Modelos Base (Reprodução do Artigo)

1. **K-Nearest Neighbors (KNN)**
2. **Random Forest**
3. **Logistic Regression**
4. **Support Vector Machine (SVM)**
5. **Naive Bayes**
6. **Decision Tree**

### Modelos de Melhoria

7. **Gradient Boosting** ⭐
8. **Random Forest Tuned**

### Modelos Otimizados (Grid Search)

9. **Random Forest Optimized**
10. **Logistic Regression Optimized**
11. **SVM Optimized**
12. **Gradient Boosting Optimized** 🏆
13. **Random Forest Tuned v2**

---

## 📈 Resultados

### Performance dos Modelos (Top 5)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Gradient Boosting Optimized** 🏆 | **92.51%** | 0.9250 | 0.9251 | 0.9250 | 0.9698 |
| Random Forest Tuned v2 | 91.19% | 0.9123 | 0.9119 | 0.9121 | 0.9654 |
| Random Forest Optimized | 90.74% | 0.9078 | 0.9074 | 0.9076 | 0.9621 |
| Gradient Boosting | 89.87% | 0.8991 | 0.8987 | 0.8989 | 0.9587 |
| Random Forest | 89.43% | 0.8947 | 0.8943 | 0.8945 | 0.9543 |

### Modelo de Produção

**Gradient Boosting Optimized**
- Test Accuracy: **91.38%**
- Validation Accuracy: **92.51%**
- Armazenado em: `models/production_model/`

### Comparação com Artigo Original

| Métrica | Artigo Original (KNN) | Nossa Implementação (GB) | Melhoria |
|---------|----------------------|--------------------------|----------|
| Accuracy | 91.80% | 92.51% | **+0.71%** |

---

## 📁 Estrutura do Projeto

```
heart-disease-ml-pipeline/
├── api/                          # FastAPI application
│   ├── main.py                   # Endpoints principais
│   ├── models.py                 # Modelos Pydantic
│   ├── config.py                 # Configurações
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/                    # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_predictions.ipynb
├── src/                          # Código fonte Python
│   ├── __init__.py
│   ├── s3_utils.py               # Cliente MinIO/S3
│   ├── db_utils.py               # Cliente PostgreSQL
│   ├── mlflow_utils.py           # Cliente MLflow
│   ├── data_preprocessing.py          # Pré-processamento
│   ├── model_training.py          # Treinamento
│   └── model_evaluation.py        # Avaliação
├── streaming/                    # Simulador de streaming
│   ├── stream_simulator.py
│   ├── thingsboard_client.py
│   ├── Dockerfile
│   └── requirements.txt
├── scripts/                      # Scripts auxiliares
│   ├── test_pipeline.sh
│   ├── split_data.py
│   ├── upload_to_minio.py
│   └── add_predictions_table.sql
├── data/                         # Dados locais
│   └── raw/
│       └── heart.csv
├── database/                     # Scripts SQL
│   └── init.sql
├── thingsboard/                  # Configurações ThingsBoard
│   └── config/
│       └── dashboard_config.json
├── mlflow/                       # MLflow setup
│   └── Dockerfile
├── docker-compose.yml            # Orquestração de containers             
├── .gitignore
└── README.md
```

---

## 🔧 Comandos Úteis

### Gerenciamento de Containers

```bash
# Ver logs de um serviço
docker compose logs -f [service_name]

# Restart de serviço específico
docker compose restart [service_name]

# Parar todos os serviços
docker compose down

# Limpar volumes (⚠️ apaga dados)
docker compose down -v

# Rebuild sem cache
docker compose build --no-cache [service_name]

# Ver uso de recursos
docker stats
```

### Streaming

```bash
# Streaming com configurações customizadas
docker compose run --rm streaming python stream_simulator.py \
  --delay 0.5 \
  --max-samples 100 \
  --no-api

# Ver todas as opções
docker compose run --rm streaming python stream_simulator.py --help
```

### Banco de Dados

```bash
# Acessar PostgreSQL
docker compose exec postgres psql -U postgres -d mlflow_db

# Ver tabelas
docker compose exec postgres psql -U postgres -d mlflow_db -c "\dt heart_disease.*"

# Query de predições
docker compose exec postgres psql -U postgres -d mlflow_db -c \
  "SELECT * FROM heart_disease.predictions ORDER BY created_at DESC LIMIT 10;"

# Backup do banco
docker compose exec postgres pg_dump -U postgres mlflow_db > backup.sql
```

### MinIO

```bash
# Listar arquivos
docker compose exec minio mc ls myminio/ml-bucket-heart/

# Copiar arquivo para local
docker compose exec minio mc cp myminio/ml-bucket-heart/models/production_model_metadata.csv /tmp/
```

---


## 🐛 Troubleshooting

### API não inicia

```bash
docker compose logs api
docker compose restart api
```

### ThingsBoard não recebe dados

```bash
# Verificar se ThingsBoard está rodando
docker compose ps thingsboard

# Ver logs
docker compose logs thingsboard | tail -50

# Verificar token
echo $THINGSBOARD_TOKEN
```

### Incompatibilidade de versão scikit-learn

```bash
# Verificar versão no Jupyter
docker compose exec jupyter pip show scikit-learn

# Verificar versão no streaming
docker compose exec streaming pip show scikit-learn

# Devem ser idênticas (1.7.2)
```

### Modelo não carrega

```bash
# Verificar arquivos no MinIO
docker compose exec minio mc ls myminio/ml-bucket-heart/models/

# Testar carregamento direto
docker compose run --rm streaming python -c "
from src.s3_utils import S3Client
s3 = S3Client()
model = s3.load_model('models/production_model/20251201_231253/model.pkl')
print('OK' if model else 'ERRO')
"
```

---

## 📚 Referências

1. **Artigo Original:**  
   Enhancing Prognosis Accuracy for Ischemic Cardiovascular Disease Using K Nearest Neighbor Algorithm: A Robust Approach  
   IEEE Xplore: https://ieeexplore.ieee.org/document/10239171

2. **Dataset:**  
   Kaggle - Heart Disease Dataset (Comprehensive) 
   https://www.kaggle.com/datasets/sid321axn/heart-statlog-cleveland-hungary-final

3. **Tecnologias:**
   - [Docker](https://www.docker.com/)
   - [FastAPI](https://fastapi.tiangolo.com/)
   - [MLflow](https://mlflow.org/)
   - [MinIO](https://min.io/)
   - [ThingsBoard](https://thingsboard.io/)
   - [scikit-learn](https://scikit-learn.org/)

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📧 Contato

Para dúvidas ou sugestões, entre em contato com a equipe através dos emails listados no início deste documento.
