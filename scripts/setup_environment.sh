#!/bin/bash

# Script de setup inicial do ambiente

echo "=========================================="
echo "🫀 HEART DISEASE ML PIPELINE - SETUP"
echo "=========================================="

# Cores para output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Verificar se Docker está instalado
echo -e "\n${YELLOW}[1/7]${NC} Verificando Docker..."
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker não encontrado. Por favor, instale o Docker.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker instalado${NC}"

# Verificar se Docker Compose está instalado (nova sintaxe)
echo -e "\n${YELLOW}[2/7]${NC} Verificando Docker Compose..."
if ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose não encontrado (docker compose).${NC}"
    echo -e "${YELLOW}Use: docker compose, não docker-compose.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker Compose instalado${NC}"

# Criar estrutura de diretórios
echo -e "\n${YELLOW}[3/7]${NC} Criando estrutura de diretórios..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/stream
mkdir -p notebooks
mkdir -p src
mkdir -p models
mkdir -p api
mkdir -p mlflow/mlruns
mkdir -p database
mkdir -p streaming
mkdir -p scripts
mkdir -p docs
mkdir -p tests
mkdir -p thingsboard/config

echo -e "${GREEN}✅ Estrutura de diretórios criada${NC}"

# Verificar se o dataset existe
echo -e "\n${YELLOW}[4/7]${NC} Verificando dataset..."
if [ ! -f "data/raw/heart.csv" ]; then
    echo -e "${RED}❌ Dataset não encontrado em data/raw/heart.csv${NC}"
    echo -e "${YELLOW}📥 Baixe o dataset e coloque em data/raw/heart.csv${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Dataset encontrado${NC}"

# Instalar dependências Python para scripts
echo -e "\n${YELLOW}[5/7]${NC} Verificando dependências Python..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 não encontrado.${NC}"
    exit 1
fi

if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 não encontrado.${NC}"
    exit 1
fi

echo "   Instalando dependências necessárias..."
pip3 install -q pandas scikit-learn python-dotenv boto3 2>/dev/null

echo -e "${GREEN}✅ Dependências instaladas${NC}"

# Dividir dados
echo -e "\n${YELLOW}[6/7]${NC} Dividindo dataset..."
python3 scripts/split_data.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Erro ao dividir dados${NC}"
    exit 1
fi

# Iniciar Docker Compose (nova sintaxe)
echo -e "\n${YELLOW}[7/7]${NC} Iniciando serviços Docker..."
docker compose up -d

echo -e "\n⏳ Aguardando serviços iniciarem (30 segundos)..."
sleep 30

# Upload para MinIO
echo -e "\n${YELLOW}[BONUS]${NC} Fazendo upload dos dados para MinIO..."
python3 scripts/upload_to_minio.py

# Verificar status dos serviços
echo -e "\n${YELLOW}📊 Status dos serviços:${NC}"
docker compose ps

echo -e "\n=========================================="
echo -e "${GREEN}✅ SETUP CONCLUÍDO COM SUCESSO!${NC}"
echo -e "=========================================="

echo -e "\n🌐 Serviços disponíveis:"
echo "   📊 MinIO Console:  http://localhost:9001"
echo "   📈 MLFlow:         http://localhost:5000"
echo "   📓 JupyterLab:     http://localhost:8888"
echo "   🚀 FastAPI:        http://localhost:8000/docs"
echo "   📺 ThingsBoard:    http://localhost:8080"

echo -e "\n💡 Comandos úteis:"
echo "   docker compose logs -f [serviço]"
echo "   docker compose down"
echo "   docker compose restart [serviço]"