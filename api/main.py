"""
FastAPI - Heart Disease ML Pipeline
Endpoints para ingestão de dados e predições
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import sys
import os

# Adicionar src ao path
sys.path.append('/app')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import (
    PatientData, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthResponse, StatusResponse,
    ModelMetrics, IngestDataRequest
)
from config import settings

# Importar utilitários (serão carregados sob demanda para evitar erro no build)
S3Client = None
DatabaseClient = None
MLFlowClient = None

# Criar aplicação FastAPI
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache para o modelo carregado
_cached_model = None
_cached_model_name = None


def get_s3_client():
    """Lazy loading do S3Client"""
    global S3Client
    if S3Client is None:
        from src.s3_utils import S3Client as _S3Client
        S3Client = _S3Client
    return S3Client()


def get_db_client():
    """Lazy loading do DatabaseClient"""
    global DatabaseClient
    if DatabaseClient is None:
        from src.db_utils import DatabaseClient as _DatabaseClient
        DatabaseClient = _DatabaseClient
    return DatabaseClient()


def get_mlflow_client():
    """Lazy loading do MLFlowClient"""
    global MLFlowClient
    if MLFlowClient is None:
        from src.mlflow_utils import MLFlowClient as _MLFlowClient
        MLFlowClient = _MLFlowClient
    return MLFlowClient()


def load_model(model_name: str = None):
    """Carrega modelo do MLFlow com cache"""
    global _cached_model, _cached_model_name
    
    if model_name is None:
        model_name = settings.DEFAULT_MODEL
    
    # Usar cache se mesmo modelo
    if _cached_model is not None and _cached_model_name == model_name:
        return _cached_model
    
    # Carregar modelo
    mlflow_client = get_mlflow_client()
    model_uri = f"models:/{model_name}/latest"
    model = mlflow_client.load_model(model_uri)
    
    if model is None:
        raise HTTPException(status_code=404, detail=f"Modelo '{model_name}' não encontrado")
    
    # Atualizar cache
    _cached_model = model
    _cached_model_name = model_name
    
    return model


# ==================== ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Verifica se a API está funcionando"""
    return HealthResponse(
        status="healthy",
        service="Heart Disease ML API",
        version=settings.API_VERSION
    )


@app.get("/", response_model=dict)
async def root():
    """Endpoint raiz com informações da API"""
    return {
        "message": "Heart Disease ML Pipeline API",
        "version": settings.API_VERSION,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "status": "/status",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "ingest": "/ingest/stream",
            "models": "/models"
        }
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Retorna status detalhado dos serviços"""
    
    # Listar modelos disponíveis
    try:
        mlflow_client = get_mlflow_client()
        # Modelos conhecidos
        models = ['knn', 'random_forest', 'logistic_regression', 
                 'svm', 'naive_bayes', 'decision_tree',
                 'gradient_boosting', 'random_forest_tuned']
    except:
        models = []
    
    return StatusResponse(
        api_status="running",
        minio_endpoint=settings.MINIO_ENDPOINT,
        postgres_host=settings.POSTGRES_HOST,
        mlflow_tracking_uri=settings.MLFLOW_TRACKING_URI,
        models_available=models
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    patient: PatientData,
    model_name: str = None,
    background_tasks: BackgroundTasks = None
):
    """
    Realiza predição para um paciente
    
    Args:
        patient: Dados do paciente
        model_name: Nome do modelo (None = modelo padrão)
    
    Returns:
        Predição e probabilidade
    """
    
    try:
        # Carregar modelo
        model = load_model(model_name)
        model_name = model_name or settings.DEFAULT_MODEL
        
        # Preparar dados para predição
        # Converter PatientData para formato esperado pelo modelo
        import pandas as pd
        
        patient_dict = patient.dict()
        patient_df = pd.DataFrame([patient_dict])
        
        # Ajustar nomes das colunas (API usa snake_case, modelo espera os nomes originais)
        column_mapping = {
            'chest_pain_type': 'chest pain type',
            'resting_bp': 'resting bp s',
            'fasting_bs': 'fasting blood sugar',
            'resting_ecg': 'resting ecg',
            'max_hr': 'max heart rate',
            'exercise_angina': 'exercise angina',
            'st_slope': 'ST slope'
        }
        patient_df = patient_df.rename(columns=column_mapping)
        
        # TODO: Aplicar pré-processamento (One-Hot Encoding, Scaling) se necessário
        
        # Predição
        prediction = int(model.predict(patient_df)[0])
        
        # Probabilidade
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(patient_df)[0][1])
        else:
            probability = float(prediction)
        
        response = PredictionResponse(
            prediction=prediction,
            probability=probability,
            model_name=model_name,
            model_version="1",
            timestamp=datetime.now()
        )
        
        # Salvar predição no banco (background task)
        if background_tasks:
            background_tasks.add_task(
                save_prediction_to_db,
                patient_dict,
                prediction,
                probability,
                model_name
            )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


def save_prediction_to_db(patient_data: dict, prediction: int, probability: float, model_name: str):
    """Salva predição no banco de dados (background task)"""
    try:
        db = get_db_client()
        db.insert_prediction(
            patient_data=patient_data,
            prediction=prediction,
            probability=probability,
            model_name=model_name,
            model_version="1"
        )
    except Exception as e:
        print(f"Erro ao salvar predição no DB: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Realiza predições em lote
    
    Args:
        request: Lista de pacientes e modelo opcional
    
    Returns:
        Lista de predições
    """
    
    try:
        # Carregar modelo
        model = load_model(request.model_name)
        model_name = request.model_name or settings.DEFAULT_MODEL
        
        predictions = []
        
        for patient in request.patients:
            # Mesma lógica do endpoint /predict
            import pandas as pd
            
            patient_dict = patient.dict()
            patient_df = pd.DataFrame([patient_dict])
            
            # Ajustar nomes das colunas
            column_mapping = {
                'chest_pain_type': 'chest pain type',
                'resting_bp': 'resting bp s',
                'fasting_bs': 'fasting blood sugar',
                'resting_ecg': 'resting ecg',
                'max_hr': 'max heart rate',
                'exercise_angina': 'exercise angina',
                'st_slope': 'ST slope'
            }
            patient_df = patient_df.rename(columns=column_mapping)
            
            # Predição
            prediction = int(model.predict(patient_df)[0])
            
            # Probabilidade
            if hasattr(model, 'predict_proba'):
                probability = float(model.predict_proba(patient_df)[0][1])
            else:
                probability = float(prediction)
            
            predictions.append(PredictionResponse(
                prediction=prediction,
                probability=probability,
                model_name=model_name,
                model_version="1",
                timestamp=datetime.now()
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            processed_at=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição em lote: {str(e)}")


@app.post("/ingest/stream")
async def ingest_stream_data(request: IngestDataRequest):
    """
    Ingere dados de um paciente (streaming)
    
    Args:
        request: Dados do paciente e flags de salvamento
    
    Returns:
        Confirmação da ingestão
    """
    
    try:
        data = request.data
        
        # Salvar no banco se solicitado
        if request.save_to_db:
            db = get_db_client()
            db.insert_raw_data(data)
        
        # Salvar no S3 se solicitado
        if request.save_to_s3:
            s3 = get_s3_client()
            import pandas as pd
            import json
            from datetime import datetime
            
            # Criar nome único para o arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"stream/patient_{timestamp}.json"
            
            # Salvar como JSON no S3
            s3.client.put_object(
                Bucket=settings.MINIO_BUCKET,
                Key=filename,
                Body=json.dumps(data)
            )
        
        return {
            "status": "success",
            "message": "Dados ingeridos com sucesso",
            "saved_to_db": request.save_to_db,
            "saved_to_s3": request.save_to_s3,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na ingestão: {str(e)}")


@app.get("/models", response_model=list[str])
async def list_models():
    """Lista modelos disponíveis"""
    
    models = [
        'knn', 'random_forest', 'logistic_regression',
        'svm', 'naive_bayes', 'decision_tree',
        'gradient_boosting', 'random_forest_tuned'
    ]
    
    return models


@app.get("/models/{model_name}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_name: str):
    """Retorna métricas de um modelo específico"""
    
    try:
        db = get_db_client()
        metrics_df = db.get_model_metrics(model_name)
        
        if metrics_df is None or len(metrics_df) == 0:
            raise HTTPException(status_code=404, detail=f"Métricas não encontradas para modelo '{model_name}'")
        
        # Pegar última versão
        latest_metrics = metrics_df.iloc[0]
        
        return ModelMetrics(
            model_name=model_name,
            accuracy=float(latest_metrics['accuracy']),
            precision=float(latest_metrics.get('precision_class_1', 0)),
            recall=float(latest_metrics.get('recall_class_1', 0)),
            f1_score=float(latest_metrics.get('f1_class_1', 0)),
            roc_auc=float(latest_metrics.get('roc_auc', 0)) if latest_metrics.get('roc_auc') else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter métricas: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)