"""
Simulador de streaming de dados
Lê dados de validação do MinIO e envia para a API em tempo real
"""

import sys
import os

# Adicionar paths corretos
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Importar S3Client do src
try:
    from s3_utils import S3Client
except ImportError:
    try:
        from src.s3_utils import S3Client
    except ImportError:
        print("⚠️ Erro ao importar S3Client. Verifique se src/ está no PYTHONPATH")
        print(f"sys.path: {sys.path}")
        print(f"Arquivos em /app: {os.listdir('/app') if os.path.exists('/app') else 'N/A'}")
        print(f"Arquivos em /app/src: {os.listdir('/app/src') if os.path.exists('/app/src') else 'N/A'}")
        sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamingSimulator:
    def __init__(self, api_url: str = "http://api:8000", delay: float = 2.0):
        self.api_url = api_url
        self.delay = delay
        self.s3 = S3Client()
        
    def load_validation_data(self):
        """Carrega dados de validação do MinIO"""
        logger.info("📥 Carregando dados de validação do MinIO...")
        
        try:
            # Carregar features e labels
            X_val = self.s3.read_csv('processed/X_val_scaled.csv')
            y_val = self.s3.read_csv('processed/y_val.csv')['target']
            
            logger.info(f"✅ {len(X_val)} amostras carregadas")
            return X_val, y_val
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar dados: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_production_model(self):
        """Carrega modelo de produção do MinIO usando metadados"""
        logger.info("📦 Carregando modelo de produção do MinIO...")
        
        try:
            # Carregar metadados
            metadata_df = self.s3.read_csv('models/production_model_metadata.csv')
            metadata = metadata_df.iloc[0]
            
            model_path = metadata['model_path']
            model_name = metadata['model_name']
            
            logger.info(f"   Modelo: {model_name}")
            logger.info(f"   Path: {model_path}")
            logger.info(f"   Test Accuracy: {metadata['test_accuracy']:.4f}")
            logger.info(f"   Val Accuracy: {metadata['validation_accuracy']:.4f}")
            
            # Carregar modelo
            model = self.s3.load_model(model_path)
            
            if model:
                logger.info(f"✅ Modelo de produção carregado com sucesso!")
                return model, metadata
            else:
                logger.error("❌ Falha ao carregar modelo")
                return None, None
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def convert_to_api_format(self, row: pd.Series) -> dict:
        """Converte linha do DataFrame para formato esperado pela API"""
        # A API espera os nomes no formato snake_case conforme PatientData
        return {
            'age': float(row.get('age', 0)),
            'sex': int(row.get('sex', 0)),
            'chest_pain_type': int(row.get('chest pain type', 0)),
            'resting_bp': float(row.get('resting bp s', 0)),
            'cholesterol': float(row.get('cholesterol', 0)),
            'fasting_bs': int(row.get('fasting blood sugar', 0)),
            'resting_ecg': int(row.get('resting ecg', 0)),
            'max_hr': float(row.get('max heart rate', 0)),
            'exercise_angina': int(row.get('exercise angina', 0)),
            'oldpeak': float(row.get('oldpeak', 0)),
            'st_slope': int(row.get('ST slope', 0))
        }
    
    def send_prediction_request(self, patient_data: dict, patient_id: int, true_label: int):
        """Envia requisição de predição para a API"""
        try:
            # Enviar requisição para endpoint /predict
            response = requests.post(
                f"{self.api_url}/predict",
                json=patient_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result
            else:
                logger.error(f"❌ Erro na API: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Erro de conexão: {str(e)}")
            return None
    
    def run(self, max_samples: int = None, use_api: bool = True):
        """
        Executa simulação de streaming
        
        Args:
            max_samples: Número máximo de amostras (None = todas)
            use_api: Se True, usa API. Se False, usa modelo local direto do S3
        """
        logger.info("🚀 Iniciando simulação de streaming...")
        logger.info(f"⏱️  Delay entre requisições: {self.delay}s")
        logger.info(f"🎯 Modo: {'API' if use_api else 'Modelo Direto (S3)'}")
        
        # Carregar dados
        X_val, y_val = self.load_validation_data()
        
        if X_val is None:
            logger.error("❌ Não foi possível carregar dados. Abortando.")
            return
        
        # Se não usar API, carregar modelo direto
        model = None
        model_metadata = None
        if not use_api:
            model, model_metadata = self.load_production_model()
            if model is None:
                logger.error("❌ Não foi possível carregar modelo. Abortando.")
                return
        
        # Limitar amostras se especificado
        if max_samples:
            X_val = X_val.head(max_samples)
            y_val = y_val.head(max_samples)
        
        total = len(X_val)
        correct_predictions = 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 INICIANDO STREAMING DE {total} PACIENTES")
        logger.info(f"{'='*60}\n")
        
        # Processar cada amostra
        for idx, (_, row) in enumerate(X_val.iterrows(), 1):
            patient_id = idx
            true_label = int(y_val.iloc[idx - 1])
            
            logger.info(f"\n{'─'*60}")
            logger.info(f"🏥 Paciente {patient_id}/{total}")
            logger.info(f"📋 Label Real: {'Doença ❤️‍🩹' if true_label == 1 else 'Saudável ✅'}")
            
            if use_api:
                # Usar API
                patient_data = self.convert_to_api_format(row)
                result = self.send_prediction_request(patient_data, patient_id, true_label)
                
                if result:
                    predicted_label = result.get('prediction')
                    probability = result.get('probability', 0.0)
                    model_name = result.get('model_name', 'unknown')
                else:
                    logger.warning(f"⚠️ Falha na predição do paciente {patient_id}")
                    continue
            else:
                # Usar modelo direto do S3
                try:
                    # Preparar dados
                    X = pd.DataFrame([row])
                    
                    # Predição
                    predicted_label = int(model.predict(X)[0])
                    
                    # Probabilidade
                    if hasattr(model, 'predict_proba'):
                        probability = float(model.predict_proba(X)[0][1])
                    else:
                        probability = float(predicted_label)
                    
                    model_name = model_metadata['model_name']
                    
                except Exception as e:
                    logger.error(f"❌ Erro na predição: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Avaliar resultado
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct_predictions += 1
            
            # Log resultado
            emoji = "✅" if is_correct else "❌"
            logger.info(f"🔮 Predição: {'Doença ❤️‍🩹' if predicted_label == 1 else 'Saudável ✅'}")
            logger.info(f"📊 Probabilidade: {probability:.2%}")
            logger.info(f"🤖 Modelo: {model_name}")
            logger.info(f"{emoji} {'CORRETO' if is_correct else 'INCORRETO'}")
            logger.info(f"📈 Acurácia Atual: {correct_predictions}/{idx} ({correct_predictions/idx*100:.1f}%)")
            
            # Delay antes da próxima amostra
            if idx < total:
                time.sleep(self.delay)
        
        # Resumo final
        final_accuracy = correct_predictions / total * 100 if total > 0 else 0
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 RESUMO FINAL")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Total de amostras: {total}")
        logger.info(f"✅ Predições corretas: {correct_predictions}")
        logger.info(f"❌ Predições incorretas: {total - correct_predictions}")
        logger.info(f"📈 Acurácia Final: {final_accuracy:.2f}%")
        logger.info(f"{'='*60}\n")

def wait_for_api(api_url: str, max_retries: int = 30):
    """Aguarda API estar pronta"""
    logger.info("⏳ Aguardando API estar pronta...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                logger.info("✅ API está pronta!")
                return True
        except:
            pass
        
        if i < max_retries - 1:
            time.sleep(2)
    
    logger.error(f"❌ API não está respondendo após {max_retries * 2}s")
    return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulador de Streaming de Dados')
    parser.add_argument('--api-url', default='http://api:8000', help='URL da API')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay entre requisições (segundos)')
    parser.add_argument('--max-samples', type=int, default=None, help='Número máximo de amostras')
    parser.add_argument('--no-api', action='store_true', help='Não usar API, carregar modelo direto do S3')
    
    args = parser.parse_args()
    
    use_api = not args.no_api
    
    # Se usar API, aguardar estar pronta
    if use_api:
        if not wait_for_api(args.api_url):
            logger.warning("⚠️ API não está respondendo. Tentando com modelo direto do S3...")
            use_api = False
    
    # Iniciar simulação
    simulator = StreamingSimulator(api_url=args.api_url, delay=args.delay)
    simulator.run(max_samples=args.max_samples, use_api=use_api)