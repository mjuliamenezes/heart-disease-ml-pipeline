"""
Utilitários para interação com MinIO/S3
"""

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import pandas as pd
import pickle
import io
import os
from typing import Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3Client:
    """Cliente para interação com MinIO/S3"""
    
    def __init__(
        self,
        endpoint_url: str = None,
        access_key: str = None,
        secret_key: str = None,
        bucket_name: str = None
    ):
        """
        Inicializa cliente S3
        
        Args:
            endpoint_url: URL do MinIO (ex: http://minio:9000)
            access_key: Access key do MinIO
            secret_key: Secret key do MinIO
            bucket_name: Nome do bucket
        """
        # Obter endpoint da variável de ambiente ou usar default
        endpoint = endpoint_url or os.getenv('MINIO_ENDPOINT', 'minio:9000')
        
        # Garantir que endpoint tem protocolo http://
        if not endpoint.startswith(('http://', 'https://')):
            endpoint = f'http://{endpoint}'
        
        self.endpoint_url = endpoint
        self.access_key = access_key or os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = secret_key or os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
        self.bucket_name = bucket_name or os.getenv('MINIO_BUCKET', 'ml-bucket-heart')
        
        # Criar cliente
        self.client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        
        logger.info(f"S3 Client inicializado: {self.endpoint_url}/{self.bucket_name}")
    
    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """
        Faz upload de arquivo para S3
        
        Args:
            local_path: Caminho local do arquivo
            s3_key: Chave no S3 (ex: 'data/train.csv')
        
        Returns:
            bool: True se sucesso
        """
        try:
            self.client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"✅ Upload: {local_path} -> s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro no upload: {str(e)}")
            return False
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Baixa arquivo do S3
        
        Args:
            s3_key: Chave no S3
            local_path: Caminho local de destino
        
        Returns:
            bool: True se sucesso
        """
        try:
            self.client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"✅ Download: s3://{self.bucket_name}/{s3_key} -> {local_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro no download: {str(e)}")
            return False
    
    def read_csv(self, s3_key: str) -> Optional[pd.DataFrame]:
        """
        Lê CSV direto do S3
        
        Args:
            s3_key: Chave no S3
        
        Returns:
            DataFrame ou None
        """
        try:
            obj = self.client.get_object(Bucket=self.bucket_name, Key=s3_key)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            logger.info(f"✅ CSV lido: s3://{self.bucket_name}/{s3_key} ({len(df)} linhas)")
            return df
        except Exception as e:
            logger.error(f"❌ Erro ao ler CSV: {str(e)}")
            return None
    
    def write_csv(self, df: pd.DataFrame, s3_key: str) -> bool:
        """
        Escreve DataFrame como CSV no S3
        
        Args:
            df: DataFrame pandas
            s3_key: Chave no S3
        
        Returns:
            bool: True se sucesso
        """
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=csv_buffer.getvalue()
            )
            logger.info(f"✅ CSV salvo: s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao salvar CSV: {str(e)}")
            return False
    
    def save_model(self, model: Any, s3_key: str) -> bool:
        """
        Salva modelo serializado no S3
        
        Args:
            model: Modelo sklearn/mlflow
            s3_key: Chave no S3
        
        Returns:
            bool: True se sucesso
        """
        try:
            model_bytes = pickle.dumps(model)
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=model_bytes
            )
            logger.info(f"✅ Modelo salvo: s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao salvar modelo: {str(e)}")
            return False
    
    def load_model(self, s3_key: str) -> Optional[Any]:
        """
        Carrega modelo do S3
        
        Args:
            s3_key: Chave no S3
        
        Returns:
            Modelo ou None
        """
        try:
            obj = self.client.get_object(Bucket=self.bucket_name, Key=s3_key)
            model = pickle.loads(obj['Body'].read())
            logger.info(f"✅ Modelo carregado: s3://{self.bucket_name}/{s3_key}")
            return model
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {str(e)}")
            return None
    
    def list_files(self, prefix: str = '') -> list:
        """
        Lista arquivos no S3
        
        Args:
            prefix: Prefixo para filtrar (ex: 'data/')
        
        Returns:
            Lista de chaves
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
                logger.info(f"📋 {len(files)} arquivos encontrados com prefixo '{prefix}'")
                return files
            else:
                logger.info(f"📋 Nenhum arquivo encontrado com prefixo '{prefix}'")
                return []
        except Exception as e:
            logger.error(f"❌ Erro ao listar arquivos: {str(e)}")
            return []
    
    def file_exists(self, s3_key: str) -> bool:
        """
        Verifica se arquivo existe no S3
        
        Args:
            s3_key: Chave no S3
        
        Returns:
            bool: True se existe
        """
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False