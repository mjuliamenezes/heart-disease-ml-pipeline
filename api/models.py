"""
Modelos Pydantic para validação de dados da API
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class PatientData(BaseModel):
    """Dados de um paciente para predição"""
    age: int = Field(..., ge=18, le=100, description="Idade do paciente")
    sex: int = Field(..., ge=0, le=1, description="Sexo (0=Feminino, 1=Masculino)")
    chest_pain_type: int = Field(..., ge=1, le=4, description="Tipo de dor no peito (1-4)")
    resting_bp: int = Field(..., ge=60, le=260, description="Pressão arterial em repouso")
    cholesterol: int = Field(..., ge=70, le=900, description="Colesterol sérico (mg/dl)")
    fasting_bs: int = Field(..., ge=0, le=1, description="Glicemia em jejum > 120 mg/dl (0=Não, 1=Sim)")
    resting_ecg: int = Field(..., ge=0, le=2, description="Resultados ECG em repouso (0-2)")
    max_hr: int = Field(..., ge=30, le=260, description="Frequência cardíaca máxima")
    exercise_angina: int = Field(..., ge=0, le=1, description="Angina induzida por exercício (0=Não, 1=Sim)")
    oldpeak: float = Field(..., ge=-3.0, le=7.0, description="Depressão ST induzida por exercício")
    st_slope: int = Field(..., ge=0, le=3, description="Inclinação do segmento ST (0-3)")
    
    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


class PredictionResponse(BaseModel):
    """Resposta da predição"""
    prediction: int = Field(..., description="Predição (0=Saudável, 1=Doença)")
    probability: float = Field(..., description="Probabilidade de doença")
    model_name: str = Field(..., description="Nome do modelo usado")
    model_version: str = Field(..., description="Versão do modelo")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp da predição")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.8523,
                "model_name": "random_forest",
                "model_version": "1",
                "timestamp": "2025-01-15T10:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Requisição de predição em lote"""
    patients: list[PatientData] = Field(..., description="Lista de pacientes")
    model_name: Optional[str] = Field(None, description="Nome do modelo (None = melhor modelo)")


class BatchPredictionResponse(BaseModel):
    """Resposta de predição em lote"""
    predictions: list[PredictionResponse]
    total: int
    processed_at: datetime = Field(default_factory=datetime.now)


class HealthResponse(BaseModel):
    """Resposta do health check"""
    status: str
    service: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)


class StatusResponse(BaseModel):
    """Resposta detalhada do status"""
    api_status: str
    minio_endpoint: str
    postgres_host: str
    mlflow_tracking_uri: str
    models_available: list[str]
    timestamp: datetime = Field(default_factory=datetime.now)


class ModelMetrics(BaseModel):
    """Métricas de um modelo"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]


class IngestDataRequest(BaseModel):
    """Requisição para ingestão de dados"""
    data: Dict[str, Any]
    save_to_db: bool = Field(True, description="Salvar no banco de dados")
    save_to_s3: bool = Field(True, description="Salvar no MinIO/S3")