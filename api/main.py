"""
FastAPI - Heart Disease ML Pipeline
Endpoints para ingestão de dados e predições
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

# Criar aplicação FastAPI
app = FastAPI(
    title="Heart Disease ML Pipeline API",
    description="API para ingestão de dados e predições de doenças cardíacas",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint de health check
@app.get("/health")
async def health_check():
    """Verifica se a API está funcionando"""
    return {
        "status": "healthy",
        "service": "Heart Disease ML API",
        "version": "1.0.0"
    }

# Endpoint de status
@app.get("/")
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Heart Disease ML Pipeline API",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "status": "/status"
        }
    }

# Endpoint de status detalhado
@app.get("/status")
async def get_status():
    """Retorna status detalhado dos serviços"""
    
    # Verificar variáveis de ambiente
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "not_configured")
    postgres_host = os.getenv("POSTGRES_HOST", "not_configured")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "not_configured")
    
    return {
        "api_status": "running",
        "minio_endpoint": minio_endpoint,
        "postgres_host": postgres_host,
        "mlflow_tracking_uri": mlflow_uri,
        "message": "API está funcionando. Endpoints completos serão implementados em breve."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)