"""
Script para fazer upload inicial dos dados para MinIO
ATENÇÃO: Execute este script APÓS iniciar o docker-compose
"""

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

# Carregar variáveis de ambiente
load_dotenv()

def upload_to_minio():
    """Faz upload dos dados processados para o MinIO"""
    
    # Configurações do MinIO (quando executado FORA do Docker)
    MINIO_ENDPOINT = "http://localhost:9000"
    MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "ml-bucket-heart")
    
    print("🔧 Configurando cliente S3 (MinIO)...")
    
    try:
        # Cliente S3
        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        
        # Testar conexão
        s3_client.list_buckets()
        print(f"✅ Conexão com MinIO estabelecida!")
        
    except Exception as e:
        print(f"❌ Erro ao conectar ao MinIO: {str(e)}")
        print("\n💡 Dica: Certifique-se de que o MinIO está rodando:")
        print("   sudo docker-compose ps minio")
        sys.exit(1)
    
    # Verificar se bucket existe
    try:
        s3_client.head_bucket(Bucket=MINIO_BUCKET)
        print(f"✅ Bucket '{MINIO_BUCKET}' encontrado!")
    except:
        print(f"⚠️  Bucket '{MINIO_BUCKET}' não encontrado. Criando...")
        s3_client.create_bucket(Bucket=MINIO_BUCKET)
        print(f"✅ Bucket '{MINIO_BUCKET}' criado!")
    
    # Arquivos para upload
    files_to_upload = [
        ('data/raw/heart.csv', 'raw/heart.csv'),
        ('data/processed/train.csv', 'processed/train.csv'),
        ('data/processed/test.csv', 'processed/test.csv'),
        ('data/processed/validation.csv', 'processed/validation.csv'),
    ]
    
    print(f"\n📤 Iniciando upload de {len(files_to_upload)} arquivos...\n")
    
    uploaded = 0
    for local_path, s3_key in files_to_upload:
        if not os.path.exists(local_path):
            print(f"❌ Arquivo não encontrado: {local_path}")
            continue
        
        try:
            file_size = os.path.getsize(local_path) / 1024  # KB
            print(f"📤 Uploading: {local_path} -> s3://{MINIO_BUCKET}/{s3_key} ({file_size:.2f} KB)")
            
            s3_client.upload_file(
                local_path,
                MINIO_BUCKET,
                s3_key
            )
            
            print(f"   ✅ Upload concluído!")
            uploaded += 1
            
        except Exception as e:
            print(f"   ❌ Erro no upload: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"✅ Upload concluído: {uploaded}/{len(files_to_upload)} arquivos enviados")
    print(f"{'='*60}")
    
    # Listar arquivos no bucket
    print(f"\n📋 Arquivos no bucket '{MINIO_BUCKET}':")
    try:
        response = s3_client.list_objects_v2(Bucket=MINIO_BUCKET)
        if 'Contents' in response:
            for obj in response['Contents']:
                size_kb = obj['Size'] / 1024
                print(f"   📄 {obj['Key']} ({size_kb:.2f} KB)")
        else:
            print("   (vazio)")
    except Exception as e:
        print(f"   ❌ Erro ao listar: {str(e)}")
    
    print(f"\n🌐 Acesse o MinIO Console em: http://localhost:9001")
    print(f"   👤 User: {MINIO_ACCESS_KEY}")
    print(f"   🔑 Pass: {MINIO_SECRET_KEY}")

if __name__ == "__main__":
    print("=" * 60)
    print("🫀 HEART DISEASE - UPLOAD PARA MINIO")
    print("=" * 60)
    print("\n⚠️  ATENÇÃO: Certifique-se de que:")
    print("   1. Os dados foram divididos (split_data.py)")
    print("   2. O Docker Compose está rodando (docker-compose up -d)")
    print("   3. O MinIO está acessível em http://localhost:9000")
    
    input("\n🔄 Pressione ENTER para continuar ou CTRL+C para cancelar...")
    
    try:
        upload_to_minio()
        print("\n✅ PROCESSO CONCLUÍDO!")
        print("\n📌 Próximos passos:")
        print("   1. Acesse JupyterLab: http://localhost:8888")
        print("   2. Acesse MLFlow: http://localhost:5000")
        print("   3. Comece a análise exploratória!")
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        print("\n💡 Dica: Verifique se o Docker Compose está rodando:")
        print("   docker-compose ps")