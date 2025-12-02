#!/bin/bash

# Script para resetar banco de dados com novo schema

echo "🔄 Resetando banco de dados..."

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. Parar containers
echo -e "${YELLOW}[1/4] Parando containers...${NC}"
docker compose down

# 2. Descobrir nome do volume
echo -e "\n${YELLOW}[2/4] Procurando volume do PostgreSQL...${NC}"
VOLUME=$(docker volume ls | grep postgres | awk '{print $2}')

if [ -z "$VOLUME" ]; then
    echo -e "${YELLOW}⚠️  Nenhum volume do PostgreSQL encontrado. Isso é normal se for a primeira vez.${NC}"
else
    echo -e "Volume encontrado: ${GREEN}$VOLUME${NC}"
    
    # 3. Remover volume
    echo -e "\n${YELLOW}[3/4] Removendo volume...${NC}"
    docker volume rm $VOLUME
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Volume removido${NC}"
    else
        echo -e "${RED}❌ Erro ao remover volume. Tente manualmente:${NC}"
        echo "   docker volume rm $VOLUME"
    fi
fi

# 4. Recriar containers
echo -e "\n${YELLOW}[4/4] Recriando containers...${NC}"
docker compose up -d postgres

# Aguardar PostgreSQL inicializar
echo -e "\n⏳ Aguardando PostgreSQL inicializar (15s)..."
sleep 15

# Verificar se funcionou
echo -e "\n${YELLOW}Verificando schema...${NC}"
docker compose exec postgres psql -U postgres -d mlflow_db -c "\dt heart_disease.*" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Banco de dados resetado com sucesso!${NC}"
else
    echo -e "\n${RED}❌ Erro ao verificar schema${NC}"
fi

echo -e "\n${YELLOW}Para subir todos os serviços:${NC}"
echo "   docker compose up -d"