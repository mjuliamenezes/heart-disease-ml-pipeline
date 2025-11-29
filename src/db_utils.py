"""
Utilitários para interação com PostgreSQL
"""

import psycopg2
from psycopg2.extras import RealDictCursor, Json
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
import os
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseClient:
    """Cliente para interação com PostgreSQL"""
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None
    ):
        """
        Inicializa cliente PostgreSQL
        
        Args:
            host: Host do PostgreSQL
            port: Porta do PostgreSQL
            database: Nome do banco
            user: Usuário
            password: Senha
        """
        self.host = host or os.getenv('POSTGRES_HOST', 'postgres')
        self.port = port or int(os.getenv('POSTGRES_PORT', 5432))
        self.database = database or os.getenv('POSTGRES_DB', 'mlflow_db')
        self.user = user or os.getenv('POSTGRES_USER', 'postgres')
        self.password = password or os.getenv('POSTGRES_PASSWORD', 'postgres')
        
        self.connection_string = f"host={self.host} port={self.port} dbname={self.database} user={self.user} password={self.password}"
        
        logger.info(f"Database Client inicializado: {self.host}:{self.port}/{self.database}")
    
    @contextmanager
    def get_connection(self):
        """Context manager para conexão com o banco"""
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"❌ Erro na transação: {str(e)}")
            raise
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[List[Dict]]:
        """
        Executa query SELECT e retorna resultados
        
        Args:
            query: Query SQL
            params: Parâmetros da query
        
        Returns:
            Lista de dicionários com resultados
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"❌ Erro ao executar query: {str(e)}")
            return None
    
    def execute_insert(self, query: str, params: tuple = None) -> bool:
        """
        Executa query INSERT/UPDATE/DELETE
        
        Args:
            query: Query SQL
            params: Parâmetros da query
        
        Returns:
            bool: True se sucesso
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    logger.info(f"✅ Query executada com sucesso")
                    return True
        except Exception as e:
            logger.error(f"❌ Erro ao executar insert: {str(e)}")
            return False
    
    def insert_raw_data(self, data: Dict[str, Any]) -> bool:
        """
        Insere dados brutos na tabela raw_data
        
        Args:
            data: Dicionário com dados do paciente
        
        Returns:
            bool: True se sucesso
        """
        query = """
            INSERT INTO heart_disease.raw_data (
                age, sex, chest_pain_type, resting_bp, cholesterol,
                fasting_bs, resting_ecg, max_hr, exercise_angina,
                oldpeak, st_slope, target
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            data.get('age'),
            data.get('sex'),
            data.get('chest_pain_type'),
            data.get('resting_bp'),
            data.get('cholesterol'),
            data.get('fasting_bs'),
            data.get('resting_ecg'),
            data.get('max_hr'),
            data.get('exercise_angina'),
            data.get('oldpeak'),
            data.get('st_slope'),
            data.get('target')
        )
        
        return self.execute_insert(query, params)
    
    def insert_prediction(
        self,
        patient_data: Dict[str, Any],
        prediction: int,
        probability: float,
        model_name: str,
        model_version: str
    ) -> bool:
        """
        Insere predição na tabela predictions
        
        Args:
            patient_data: Dados do paciente
            prediction: Predição (0 ou 1)
            probability: Probabilidade
            model_name: Nome do modelo
            model_version: Versão do modelo
        
        Returns:
            bool: True se sucesso
        """
        query = """
            INSERT INTO heart_disease.predictions (
                patient_data, prediction, probability, model_name, model_version
            ) VALUES (%s, %s, %s, %s, %s)
        """
        
        params = (
            Json(patient_data),
            prediction,
            probability,
            model_name,
            model_version
        )
        
        return self.execute_insert(query, params)
    
    def insert_model_metrics(
        self,
        model_name: str,
        model_version: str,
        metrics: Dict[str, float]
    ) -> bool:
        """
        Insere métricas do modelo na tabela model_metrics
        
        Args:
            model_name: Nome do modelo
            model_version: Versão do modelo
            metrics: Dicionário com métricas
        
        Returns:
            bool: True se sucesso
        """
        query = """
            INSERT INTO heart_disease.model_metrics (
                model_name, model_version, accuracy,
                precision_class_0, precision_class_1,
                recall_class_0, recall_class_1,
                f1_class_0, f1_class_1, roc_auc
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        params = (
            model_name,
            model_version,
            metrics.get('accuracy'),
            metrics.get('precision_class_0'),
            metrics.get('precision_class_1'),
            metrics.get('recall_class_0'),
            metrics.get('recall_class_1'),
            metrics.get('f1_class_0'),
            metrics.get('f1_class_1'),
            metrics.get('roc_auc')
        )
        
        return self.execute_insert(query, params)
    
    def get_recent_predictions(self, limit: int = 100) -> Optional[pd.DataFrame]:
        """
        Retorna predições recentes
        
        Args:
            limit: Número de predições
        
        Returns:
            DataFrame com predições
        """
        query = f"""
            SELECT * FROM heart_disease.predictions
            ORDER BY created_at DESC
            LIMIT {limit}
        """
        
        results = self.execute_query(query)
        
        if results:
            return pd.DataFrame(results)
        return None
    
    def get_model_metrics(self, model_name: str = None) -> Optional[pd.DataFrame]:
        """
        Retorna métricas dos modelos
        
        Args:
            model_name: Nome do modelo (opcional)
        
        Returns:
            DataFrame com métricas
        """
        if model_name:
            query = """
                SELECT * FROM heart_disease.model_metrics
                WHERE model_name = %s
                ORDER BY training_date DESC
            """
            results = self.execute_query(query, (model_name,))
        else:
            query = """
                SELECT * FROM heart_disease.model_metrics
                ORDER BY training_date DESC
            """
            results = self.execute_query(query)
        
        if results:
            return pd.DataFrame(results)
        return None
    
    def read_table_to_df(self, table_name: str, schema: str = 'heart_disease') -> Optional[pd.DataFrame]:
        """
        Lê tabela completa para DataFrame
        
        Args:
            table_name: Nome da tabela
            schema: Schema do banco
        
        Returns:
            DataFrame
        """
        query = f"SELECT * FROM {schema}.{table_name}"
        results = self.execute_query(query)
        
        if results:
            return pd.DataFrame(results)
        return None