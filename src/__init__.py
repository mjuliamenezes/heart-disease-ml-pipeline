"""
Heart Disease ML Pipeline - Utilitários
Módulos reutilizáveis para processamento, modelagem e avaliação
"""

__version__ = "1.0.0"
__author__ = "Heart Disease ML Team"

# Imports facilitados
from .s3_utils import S3Client
from .db_utils import DatabaseClient
from .mlflow_utils import MLFlowClient
from .data_preprocessing import DataPreprocessor
from .model_training import ModelTrainer
from .model_evaluation import ModelEvaluator

__all__ = [
    'S3Client',
    'DatabaseClient',
    'MLFlowClient',
    'DataPreprocessor',
    'ModelTrainer',
    'ModelEvaluator'
]