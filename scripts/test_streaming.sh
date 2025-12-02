#!/bin/bash

# Script para testar streaming completo

echo "=========================================="
echo "🧪 TESTANDO STREAMING PIPELINE"
echo "=========================================="

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Verificar se API está rodando
echo -e "\n${YELLOW}[1/5] Verificando API...${NC}"
response=$(curl -sf http://localhost:8000/health 2>&1)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ API está rodando${NC}"
else
    echo -e "${RED}❌ API não está respondendo${NC}"
    echo "Execute: docker compose up -d api"
    exit 1
fi

# 2. Verificar modelos disponíveis
echo -e "\n${YELLOW}[2/5] Verificando modelos...${NC}"
models=$(curl -sf http://localhost:8000/models 2>&1)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Modelos disponíveis:${NC}"
    echo "$models" | python3 -m json.tool
else
    echo -e "${RED}❌ Erro ao listar modelos${NC}"
fi

# 3. Testar predição única
echo -e "\n${YELLOW}[3/5] Testando predição única...${NC}"
prediction=$(curl -sf -X POST "http://localhost:8000/predict" \
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
  }' 2>&1)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Predição realizada com sucesso:${NC}"
    echo "$prediction" | python3 -m json.tool
else
    echo -e "${RED}❌ Erro na predição${NC}"
fi

# 4. Verificar banco de dados
echo -e "\n${YELLOW}[4/5] Verificando banco de dados...${NC}"
docker compose exec -T postgres psql -U postgres -d mlflow_db -c \
  "SELECT COUNT(*) as total_predictions FROM heart_disease.predictions;" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Banco de dados acessível${NC}"
else
    echo -e "${RED}❌ Erro ao acessar banco de dados${NC}"
fi

# 5. Iniciar streaming
echo -e "\n${YELLOW}[5/5] Iniciando streaming de teste...${NC}"
echo -e "${YELLOW}Processando 10 amostras...${NC}\n"

docker compose --profile streaming up streaming &
STREAMING_PID=$!

# Aguardar 30 segundos
sleep 30

# Parar streaming
docker compose --profile streaming down 2>/dev/null

echo -e "\n${YELLOW}Verificando resultados do streaming...${NC}"
docker compose exec -T postgres psql -U postgres -d mlflow_db -c \
  "SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as disease,
    SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) as healthy
   FROM heart_disease.predictions 
   WHERE created_at >= NOW() - INTERVAL '1 minute';" 2>/dev/null

echo -e "\n=========================================="
echo -e "${GREEN}✅ TESTES CONCLUÍDOS!${NC}"
echo -e "=========================================="

echo -e "\n📊 Para ver estatísticas completas:"
echo "   docker compose exec postgres psql -U postgres -d mlflow_db -c \"SELECT * FROM heart_disease.hourly_stats;\""

echo -e "\n🚀 Para rodar streaming completo:"
echo "   docker compose --profile streaming up streaming"