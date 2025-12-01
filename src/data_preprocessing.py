"""
Utilitários para pré-processamento de dados
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Classe para pré-processamento de dados"""
    
    def __init__(self):
        """Inicializa preprocessador"""
        self.scaler = None
        self.encoder = None  # Adicionar encoder
        self.feature_names = None
        logger.info("DataPreprocessor inicializado")
    
    def load_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa features e target
        
        Args:
            df: DataFrame com dados
            target_col: Nome da coluna target
        
        Returns:
            Tuple (X, y)
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        self.feature_names = X.columns.tolist()
        
        logger.info(f"✅ Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features")
        logger.info(f"📊 Distribuição do target: {dict(y.value_counts())}")
        
        return X, y
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Trata valores faltantes
        
        Args:
            df: DataFrame
            strategy: Estratégia ('mean', 'median', 'mode', 'drop')
        
        Returns:
            DataFrame tratado
        """
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            logger.info("✅ Nenhum valor faltante encontrado")
            return df
        
        logger.info(f"⚠️  {missing_count} valores faltantes encontrados")
        
        if strategy == 'drop':
            df_clean = df.dropna()
            logger.info(f"✅ Linhas com valores faltantes removidas: {len(df) - len(df_clean)}")
        elif strategy == 'mean':
            df_clean = df.fillna(df.mean(numeric_only=True))
            logger.info("✅ Valores faltantes preenchidos com média")
        elif strategy == 'median':
            df_clean = df.fillna(df.median(numeric_only=True))
            logger.info("✅ Valores faltantes preenchidos com mediana")
        elif strategy == 'mode':
            df_clean = df.fillna(df.mode().iloc[0])
            logger.info("✅ Valores faltantes preenchidos com moda")
        else:
            df_clean = df
            logger.warning(f"⚠️  Estratégia '{strategy}' não reconhecida")
        
        return df_clean
    
    def normalize_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame = None,
        method: str = 'standard'
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Normaliza features
        
        Args:
            X_train: Features de treino
            X_test: Features de teste (opcional)
            method: Método ('standard' ou 'minmax')
        
        Returns:
            Tuple (X_train_scaled, X_test_scaled)
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            logger.warning(f"⚠️  Método '{method}' não reconhecido. Usando StandardScaler")
            self.scaler = StandardScaler()
        
        # Fit no treino
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        logger.info(f"✅ Features normalizadas ({method}): {X_train_scaled.shape}")
        
        # Transform no teste (se fornecido)
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            logger.info(f"✅ Features de teste normalizadas: {X_test_scaled.shape}")
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def balance_classes(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'smote',
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balanceia classes usando SMOTE
        
        Args:
            X: Features
            y: Target
            method: Método de balanceamento
            random_state: Seed
        
        Returns:
            Tuple (X_balanced, y_balanced)
        """
        original_dist = dict(y.value_counts())
        logger.info(f"📊 Distribuição original: {original_dist}")
        
        if method == 'smote':
            smote = SMOTE(random_state=random_state)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Converter de volta para DataFrame/Series
            X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
            y_balanced = pd.Series(y_balanced, name=y.name)
            
            balanced_dist = dict(y_balanced.value_counts())
            logger.info(f"✅ Distribuição balanceada (SMOTE): {balanced_dist}")
            
            return X_balanced, y_balanced
        else:
            logger.warning(f"⚠️  Método '{method}' não implementado. Retornando dados originais")
            return X, y
    
    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: list = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers
        
        Args:
            df: DataFrame
            columns: Colunas para analisar (None = todas numéricas)
            method: Método ('iqr' ou 'zscore')
            threshold: Threshold (1.5 para IQR, 3 para zscore)
        
        Returns:
            DataFrame sem outliers
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_clean = df.copy()
        total_outliers = 0
        
        if method == 'iqr':
            for col in columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                total_outliers += outliers
                
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
                ]
        
        elif method == 'zscore':
            for col in columns:
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outliers = (z_scores > threshold).sum()
                total_outliers += outliers
                
                df_clean = df_clean[z_scores <= threshold]
        
        logger.info(f"✅ Outliers removidos ({method}): {total_outliers} registros")
        logger.info(f"📊 Dataset: {len(df)} -> {len(df_clean)} linhas")
        
        return df_clean
    
    def get_feature_info(self, df: pd.DataFrame) -> dict:
        """
        Retorna informações sobre as features
        
        Args:
            df: DataFrame
        
        Returns:
            Dicionário com informações
        """
        info = {
            'n_samples': len(df),
            'n_features': len(df.columns),
            'features': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(exclude=[np.number]).columns.tolist()
        }
        
        logger.info(f"📊 Info: {info['n_samples']} amostras, {info['n_features']} features")
        
        return info
    
    def clean_heart_disease_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pré-processamento específico para Heart Disease dataset
        Baseado na análise exploratória realizada
        
        Args:
            df: DataFrame bruto
        
        Returns:
            DataFrame limpo
        """
        logger.info("🔄 Iniciando limpeza customizada do Heart Disease dataset...")
        
        df_clean = df.copy()
        original_shape = df_clean.shape
        
        # 1. Remover duplicatas (23% dos dados!)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        logger.info(f"✅ Duplicatas removidas: {original_shape[0]} -> {len(df_clean)} linhas")
        
        # 2. Tratar valores impossíveis
        # resting bp s = 0 (fisiologicamente impossível)
        if (df_clean["resting bp s"] == 0).any():
            rbp_median = df_clean.loc[df_clean["resting bp s"] > 0, "resting bp s"].median()
            n_zeros = (df_clean["resting bp s"] == 0).sum()
            df_clean.loc[df_clean["resting bp s"] == 0, "resting bp s"] = rbp_median
            logger.info(f"✅ resting bp s: {n_zeros} valores 0 substituídos por mediana ({rbp_median})")
        
        # cholesterol = 0 (14.45% dos dados!)
        if (df_clean["cholesterol"] == 0).any():
            chol_median = df_clean.loc[df_clean["cholesterol"] > 0, "cholesterol"].median()
            n_zeros = (df_clean["cholesterol"] == 0).sum()
            df_clean.loc[df_clean["cholesterol"] == 0, "cholesterol"] = chol_median
            logger.info(f"✅ cholesterol: {n_zeros} valores 0 substituídos por mediana ({chol_median})")
        
        # 3. Verificar distribuição do target
        target_dist = df_clean["target"].value_counts()
        logger.info(f"📊 Distribuição do target: {dict(target_dist)}")
        
        logger.info(f"✅ Dataset limpo: {df_clean.shape}")
        
        return df_clean
    
    def apply_onehot_encoding(
        self, 
        df: pd.DataFrame, 
        categorical_cols: list = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Aplica One-Hot Encoding nas colunas categóricas
        
        Args:
            df: DataFrame
            categorical_cols: Lista de colunas categóricas (None = default)
            fit: Se True, faz fit_transform. Se False, apenas transform
        
        Returns:
            DataFrame com encoding
        """
        from sklearn.preprocessing import OneHotEncoder
        
        if categorical_cols is None:
            categorical_cols = ["chest pain type", "resting ecg", "ST slope"]
        
        # Se for o primeiro dataset (fit=True), criar e ajustar encoder
        if fit or self.encoder is None:
            logger.info(f"🔄 Criando OneHotEncoder e fazendo fit em: {categorical_cols}")
            self.encoder = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
            encoded = self.encoder.fit_transform(df[categorical_cols])
        else:
            # Se não for o primeiro, apenas transformar usando encoder já ajustado
            logger.info(f"🔄 Aplicando OneHotEncoder já ajustado em: {categorical_cols}")
            encoded = self.encoder.transform(df[categorical_cols])
        
        # Criar DataFrame com features codificadas
        encoded_df = pd.DataFrame(
            encoded,
            columns=self.encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )
        
        # Remover colunas originais e adicionar codificadas
        df_encoded = pd.concat([
            df.drop(columns=categorical_cols).reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ], axis=1)
        
        logger.info(f"✅ One-Hot Encoding: {df.shape[1]} -> {df_encoded.shape[1]} colunas")
        
        return df_encoded
        """
        Cria novas features (feature engineering)
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()
        
        # Exemplo: Criar feature de risco cardiovascular
        if 'age' in df_new.columns and 'cholesterol' in df_new.columns:
            df_new['age_cholesterol_risk'] = df_new['age'] * df_new['cholesterol'] / 1000
            logger.info("✅ Feature criada: age_cholesterol_risk")
        
        # Exemplo: Criar feature de pressão elevada
        if 'resting_bp' in df_new.columns:
            df_new['high_bp'] = (df_new['resting_bp'] > 140).astype(int)
            logger.info("✅ Feature criada: high_bp")
        
        # Exemplo: Criar feature de colesterol elevado
        if 'cholesterol' in df_new.columns:
            df_new['high_cholesterol'] = (df_new['cholesterol'] > 240).astype(int)
            logger.info("✅ Feature criada: high_cholesterol")
        
        logger.info(f"📊 Feature engineering concluído: {len(df_new.columns)} features totais")
        
        return df_new
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        normalize: bool = True,
        balance: bool = False,
        remove_outliers: bool = False
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Pipeline completo de pré-processamento
        
        Args:
            df: DataFrame
            target_col: Nome da coluna target
            normalize: Normalizar features
            balance: Balancear classes
            remove_outliers: Remover outliers
        
        Returns:
            Tuple (X_processed, y)
        """
        logger.info("🔄 Iniciando pipeline de pré-processamento...")
        
        # Tratar valores faltantes
        df_clean = self.handle_missing_values(df)
        
        # Remover outliers (se solicitado)
        if remove_outliers:
            df_clean = self.remove_outliers(df_clean)
        
        # Separar X e y
        X, y = self.load_data(df_clean, target_col)
        
        # Balancear classes (se solicitado)
        if balance:
            X, y = self.balance_classes(X, y)
        
        # Normalizar (se solicitado)
        if normalize:
            X, _ = self.normalize_features(X)
        
        logger.info("✅ Pipeline de pré-processamento concluído!")
        
        return X, y