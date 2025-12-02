# 🫀 Heart Disease ML Pipeline

Pipeline completo de Machine Learning para predição de doenças cardíacas com streaming em tempo real e dashboard interativo.

## 📊 Arquitetura

```
┌─────────────┐     ┌──────────┐     ┌─────────────┐
│   MinIO/S3  │────▶│ Jupyter  │────▶│   MLflow    │
│   Storage   │     │ Notebooks│     │  Tracking   │
└─────────────┘     └──────────┘     └─────────────┘
       │                   │                  │
       │                   ▼                  │
       │            ┌──────────┐              │
       └───────────▶│PostgreSQL│◀─────────────┘
                    └──────────┘
                          │
                          ▼
              ┌────────────────────┐
              │  Streaming         │
              │  Simulator         │
              └────────────────────┘
                          │
                          ▼
              ┌────────────────────┐
              │  ThingsBoard       │
              │  Dashboard         │
              └────────────────────┘
```

## 🚀 Início Rápido

### Pré-requisitos
- Docker & Docker Compose
- 8GB RAM mínimo
- 10GB espaço em disco

### 1. Clone e Configure

```bash
git clone <repo>
cd heart-disease-ml-pipeline

# Configurar variáveis de ambiente
cp .env.example .env
nano .env  # Ajustar conforme necessário
```

### 2. Iniciar Infraestrutura

```bash
# Subir todos os serviços
docker compose up -d

# Aguardar inicialização (2-3 minutos)
docker compose ps
```

### 3. Acessar Serviços

- **JupyterLab**: http://localhost:8888
- **MLflow**: http://localhost:5000
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- **API**: http://localhost:8000/docs
- **ThingsBoard**: http://localhost:8080 (tenant@thingsboard.org/tenant)

### 4. Executar Pipeline

```bash
# No JupyterLab, executar notebooks em ordem:
# 01_data_ingestion.ipynb
# 02_eda.ipynb
# 03_model_training.ipynb
# 04_model_evaluation.ipynb
# 05_hyperparameter_tuning.ipynb
```

### 5. Rodar Streaming

```bash
# Executar simulador de streaming
docker compose --profile streaming up streaming

# Ver dashboard em tempo real: http://localhost:8080
```

## 📁 Estrutura do Projeto

```
.
├── api/                    # FastAPI application
│   ├── main.py
│   ├── models.py
│   └── config.py
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_hyperparameter_tuning.ipynb
├── src/                    # Código fonte Python
│   ├── s3_utils.py
│   ├── db_utils.py
│   ├── mlflow_utils.py
│   ├── preprocessing.py
│   ├── model_trainer.py
│   └── model_evaluator.py
├── streaming/              # Simulador de streaming
│   ├── stream_simulator.py
│   └── thingsboard_client.py
├── data/                   # Dados locais
├── database/               # Scripts SQL
│   └── init.sql
├── thingsboard/            # Configurações ThingsBoard
│   └── config/
├── docker-compose.yml
└── README.md
```

## 🤖 Modelos Treinados

### Modelos Base (do Artigo)
- K-Nearest Neighbors (KNN)
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree

### Modelos de Melhoria
- Gradient Boosting
- Random Forest Tuned

### Modelos Otimizados (Grid Search)
- Random Forest Optimized
- Logistic Regression Optimized
- SVM Optimized
- Gradient Boosting Optimized
- Random Forest Tuned v2

### 🏆 Modelo de Produção
**Gradient Boosting Optimized**
- Test Accuracy: 91.38%
- Validation Accuracy: 92.51%
- Path: `models/production_model/`

## 📊 Dataset

**Heart Disease Dataset**
- **Fonte**: UCI Machine Learning Repository
- **Amostras**: 918
- **Features**: 11
- **Target**: Doença cardíaca (0=Saudável, 1=Doença)

### Features
- age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope

## 🎯 Resultados

### Performance dos Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Gradient Boosting Optimized | 91.38% | 0.92 | 0.91 | 0.91 |
| Random Forest Tuned v2 | 90.52% | 0.91 | 0.90 | 0.90 |
| Random Forest Optimized | 89.66% | 0.90 | 0.89 | 0.89 |

### Métricas do Streaming
- **Total de amostras**: 227
- **Acurácia média**: 92.51%
- **Predições/segundo**: ~2

## 🔧 Comandos Úteis

```bash
# Ver logs
docker compose logs -f [service]

# Restart serviço
docker compose restart [service]

# Parar tudo
docker compose down

# Limpar volumes (CUIDADO: apaga dados)
docker compose down -v

# Rebuild serviço
docker compose build [service] --no-cache

# Executar streaming com opções
docker compose run --rm streaming python stream_simulator.py \
  --delay 0.5 \
  --max-samples 50 \
  --no-api

# Acessar container
docker compose exec [service] bash

# Ver uso de recursos
docker stats
```

## 🔬 Desenvolvimento

### Adicionar novo modelo

1. Implementar em `src/model_trainer.py`
2. Treinar no notebook `03_model_training.ipynb`
3. Avaliar no notebook `04_model_evaluation.ipynb`
4. Otimizar no notebook `05_hyperparameter_tuning.ipynb`

### Modificar streaming

1. Editar `streaming/stream_simulator.py`
2. Rebuild: `docker compose build streaming`
3. Testar: `docker compose run --rm streaming python stream_simulator.py --max-samples 5`

## 📈 Dashboard ThingsBoard

### Widgets Disponíveis
- ✅ Gauge de acurácia em tempo real
- ✅ Timeline de predições vs labels reais
- ✅ Gráfico de probabilidades
- ✅ Tabela de últimas predições
- ✅ Cards com métricas agregadas

### Configuração
1. Acessar: http://localhost:8080
2. Login: `tenant@thingsboard.org` / `tenant`
3. Ir em Devices → `heart-disease-predictions`
4. Ver telemetria em "Latest telemetry"

## 🐛 Troubleshooting

### API unhealthy
```bash
docker compose logs api
docker compose restart api
```

### Streaming não conecta
```bash
# Verificar se ThingsBoard está rodando
docker compose ps thingsboard

# Ver logs
docker compose logs thingsboard
```

### Modelo não carrega
```bash
# Verificar versões do scikit-learn
docker compose exec jupyter pip show scikit-learn
docker compose exec streaming pip show scikit-learn

# Devem ser iguais (1.7.2)
```

## 👥 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT.

## 🙏 Agradecimentos

- UCI Machine Learning Repository - Dataset
- Artigo base: [Link do artigo científico]
- Comunidade open-source

---