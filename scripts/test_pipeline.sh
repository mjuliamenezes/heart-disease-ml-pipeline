#!/bin/bash

# Script para testar o pipeline completo

echo "=========================================="
echo "🧪 TESTANDO PIPELINE COMPLETO"
echo "=========================================="

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Função para testar endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local method=${3:-GET}
    
    echo -ne "Testing $name... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -sf "$url" 2>&1)
    else
        response=$(curl -sf -X "$method" "$url" 2>&1)
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ OK${NC}"
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        return 1
    fi
}

echo -e "\n${YELLOW}[1/5] Testando Infraestrutura...${NC}"
test_endpoint "MinIO Health" "http://localhost:9000/minio/health/live"
test_endpoint "MLFlow" "http://localhost:5000/health"
test_endpoint "API Health" "http://localhost:8000/health"
test_endpoint "Jupyter" "http://localhost:8888"

echo -e "\n${YELLOW}[2/5] Testando Endpoints da API...${NC}"
test_endpoint "API Root" "http://localhost:8000/"
test_endpoint "API Status" "http://localhost:8000/status"
test_endpoint "API Models" "http://localhost:8000/models"

echo -e "\n${YELLOW}[3/5] Testando Predição...${NC}"
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
  }' 2>/dev/null | python3 -m json.tool

echo -e "\n${YELLOW}[4/5] Verificando MinIO...${NC}"
echo "Arquivos no bucket:"
docker compose exec -T minio mc ls myminio/ml-bucket-heart/ 2>/dev/null || echo "  (Nenhum arquivo ainda)"

echo -e "\n${YELLOW}[5/5] Verificando PostgreSQL...${NC}"
docker compose exec -T postgres psql -U postgres -d mlflow_db -c "\dt heart_disease.*" 2>/dev/null || echo "  (Tabelas não inicializadas)"

echo -e "\n=========================================="
echo -e "${GREEN}✅ TESTES CONCLUÍDOS!${NC}"
echo -e "=========================================="

echo -e "\n📌 Próximos passos:"
echo "   1. Acesse JupyterLab: http://localhost:8888"
echo "   2. Execute notebooks na ordem: 01 -> 02 -> 03 -> 04 -> 05"
echo "   3. Teste streaming: docker-compose --profile streaming up streaming"