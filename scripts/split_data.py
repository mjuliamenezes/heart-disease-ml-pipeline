"""
Script para dividir o dataset em treino, teste e validação
Treino: 60% | Teste: 20% | Validação: 20%
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_heart_dataset():
    """Divide o dataset heart.csv em train, test e validation"""
    
    # Caminhos
    raw_data_path = './data/raw/heart.csv'
    processed_dir = './data/processed'
    
    # Criar diretórios se não existirem
    os.makedirs(processed_dir, exist_ok=True)
    
    # Verificar se arquivo existe
    if not os.path.exists(raw_data_path):
        print(f"❌ Erro: Arquivo {raw_data_path} não encontrado!")
        print("📥 Baixe o dataset do Kaggle e coloque em data/raw/heart.csv")
        return False
    
    # Carregar dados
    print("📊 Carregando dataset...")
    df = pd.read_csv(raw_data_path)
    print(f"✅ Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Informações sobre o dataset
    print(f"\n📋 Colunas: {list(df.columns)}")
    print(f"🎯 Variável alvo: {df.columns[-1]}")
    print(f"\n📊 Distribuição da variável alvo:")
    print(df.iloc[:, -1].value_counts())
    
    # Dividir em treino+teste (80%) e validação (20%)
    print("\n🔀 Dividindo dados...")
    train_test, validation = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df.iloc[:, -1]  # Mantém proporção da variável alvo
    )
    
    # Dividir treino (75% de 80% = 60% total) e teste (25% de 80% = 20% total)
    train, test = train_test_split(
        train_test,
        test_size=0.25,  # 25% de 80% = 20% do total
        random_state=42,
        stratify=train_test.iloc[:, -1]
    )
    
    # Salvar splits
    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    validation_path = os.path.join(processed_dir, 'validation.csv')
    
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    validation.to_csv(validation_path, index=False)
    
    print(f"\n✅ Dados divididos e salvos:")
    print(f"   📁 Treino:     {train_path} ({len(train)} amostras, {len(train)/len(df)*100:.1f}%)")
    print(f"   📁 Teste:      {test_path} ({len(test)} amostras, {len(test)/len(df)*100:.1f}%)")
    print(f"   📁 Validação:  {validation_path} ({len(validation)} amostras, {len(validation)/len(df)*100:.1f}%)")
    
    # Verificar distribuição das classes
    print(f"\n📊 Distribuição da variável alvo por split:")
    print(f"Treino:\n{train.iloc[:, -1].value_counts(normalize=True)}")
    print(f"\nTeste:\n{test.iloc[:, -1].value_counts(normalize=True)}")
    print(f"\nValidação:\n{validation.iloc[:, -1].value_counts(normalize=True)}")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("🫀 HEART DISEASE - DIVISÃO DO DATASET")
    print("=" * 60)
    
    success = split_heart_dataset()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
        print("=" * 60)
        print("\n📌 Próximos passos:")
        print("   1. Execute: python scripts/upload_to_minio.py")
        print("   2. Inicie os serviços: docker-compose up -d")
    else:
        print("\n❌ Processo falhou. Verifique os erros acima.")