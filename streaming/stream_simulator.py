"""
Simulador de streaming de dados
Envia dados de validação linha por linha para a API
"""

import pandas as pd
import requests
import time
import os
from datetime import datetime

def stream_validation_data():
    """Simula streaming enviando dados de validação"""
    
    # Configurações
    API_URL = os.getenv("API_URL", "http://api:8000")
    DATA_PATH = os.getenv("DATA_PATH", "/data/validation.csv")
    INTERVAL = int(os.getenv("STREAM_INTERVAL_SECONDS", "5"))
    
    print(f"{'='*60}")
    print(f"🫀 HEART DISEASE - STREAMING SIMULATOR")
    print(f"{'='*60}")
    print(f"📊 Data path: {DATA_PATH}")
    print(f"🌐 API URL: {API_URL}")
    print(f"⏱️  Interval: {INTERVAL} seconds")
    print(f"{'='*60}\n")
    
    # Verificar se arquivo existe
    if not os.path.exists(DATA_PATH):
        print(f"❌ Erro: Arquivo {DATA_PATH} não encontrado!")
        return
    
    # Carregar dados de validação
    df = pd.read_csv(DATA_PATH)
    print(f"✅ {len(df)} registros carregados\n")
    print("🚀 Iniciando streaming...\n")
    
    # Enviar dados linha por linha
    for idx, row in df.iterrows():
        try:
            # Preparar dados
            data = row.to_dict()
            
            # Enviar para API (endpoint será implementado)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] 📤 Enviando registro {idx+1}/{len(df)}: {data}")
            
            # TODO: Implementar POST para /ingest/stream quando endpoint estiver pronto
            # response = requests.post(f"{API_URL}/ingest/stream", json=data)
            # print(f"   ✅ Status: {response.status_code}")
            
            print(f"   ⏸️  (Endpoint ainda não implementado - simulação)")
            
            # Aguardar intervalo
            if idx < len(df) - 1:
                time.sleep(INTERVAL)
                
        except Exception as e:
            print(f"   ❌ Erro: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("✅ Streaming concluído!")
    print(f"{'='*60}")

if __name__ == "__main__":
    stream_validation_data()