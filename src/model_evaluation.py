"""
Utilitários para avaliação de modelos
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Classe para avaliação de modelos"""
    
    def __init__(self):
        """Inicializa evaluator"""
        self.metrics_history = []
        logger.info("ModelEvaluator inicializado")
    
    def evaluate_model(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Avalia modelo com múltiplas métricas
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
            y_pred_proba: Probabilidades (opcional)
            model_name: Nome do modelo
        
        Returns:
            Dicionário com métricas
        """
        logger.info(f"📊 Avaliando modelo: {model_name}")
        
        # Métricas básicas
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Métricas por classe
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, (prec, rec, f1) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            metrics[f'precision_class_{i}'] = prec
            metrics[f'recall_class_{i}'] = rec
            metrics[f'f1_class_{i}'] = f1
        
        # ROC AUC (se probabilidades fornecidas)
        if y_pred_proba is not None:
            try:
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    # Classificação binária
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multiclasse
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except Exception as e:
                logger.warning(f"⚠️  Erro ao calcular ROC AUC: {str(e)}")
                metrics['roc_auc'] = None
        
        # Log das métricas
        logger.info(f"✅ Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"✅ Precision: {metrics['precision']:.4f}")
        logger.info(f"✅ Recall: {metrics['recall']:.4f}")
        logger.info(f"✅ F1-Score: {metrics['f1_score']:.4f}")
        if metrics.get('roc_auc'):
            logger.info(f"✅ ROC AUC: {metrics['roc_auc']:.4f}")
        
        # Salvar no histórico
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Retorna matriz de confusão
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
        
        Returns:
            Matriz de confusão
        """
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"📊 Matriz de Confusão:\n{cm}")
        return cm
    
    def get_classification_report(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        target_names: list = None
    ) -> str:
        """
        Retorna relatório de classificação
        
        Args:
            y_true: Valores verdadeiros
            y_pred: Predições
            target_names: Nomes das classes
        
        Returns:
            String com relatório
        """
        report = classification_report(y_true, y_pred, target_names=target_names)
        logger.info(f"📊 Classification Report:\n{report}")
        return report
    
    def compare_models(
        self,
        metrics_list: list = None
    ) -> pd.DataFrame:
        """
        Compara múltiplos modelos
        
        Args:
            metrics_list: Lista de métricas (usa histórico se None)
        
        Returns:
            DataFrame com comparação
        """
        if metrics_list is None:
            metrics_list = self.metrics_history
        
        if not metrics_list:
            logger.warning("⚠️  Nenhuma métrica para comparar")
            return None
        
        df = pd.DataFrame(metrics_list)
        
        # Ordenar por accuracy
        df = df.sort_values('accuracy', ascending=False)
        
        logger.info(f"📊 Comparação de {len(df)} modelos:")
        logger.info(f"\n{df[['model_name', 'accuracy', 'precision', 'recall', 'f1_score']]}")
        
        return df
    
    def get_best_model_metrics(
        self,
        metric: str = 'accuracy'
    ) -> Optional[Dict[str, Any]]:
        """
        Retorna métricas do melhor modelo
        
        Args:
            metric: Métrica para comparação
        
        Returns:
            Dicionário com métricas do melhor modelo
        """
        if not self.metrics_history:
            logger.warning("⚠️  Nenhuma métrica no histórico")
            return None
        
        best_metrics = max(self.metrics_history, key=lambda x: x.get(metric, 0))
        
        logger.info(f"🏆 Melhor modelo ({metric}): {best_metrics['model_name']}")
        logger.info(f"   {metric}: {best_metrics[metric]:.4f}")
        
        return best_metrics
    
    def calculate_roc_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calcula curva ROC
        
        Args:
            y_true: Valores verdadeiros
            y_pred_proba: Probabilidades
        
        Returns:
            Dicionário com fpr, tpr, thresholds
        """
        try:
            # Para classificação binária
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
            else:
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            
            roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim == 2 else y_pred_proba)
            
            logger.info(f"✅ ROC Curve calculada (AUC: {roc_auc:.4f})")
            
            return {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc
            }
        except Exception as e:
            logger.error(f"❌ Erro ao calcular ROC curve: {str(e)}")
            return None
    
    def calculate_precision_recall_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Calcula curva Precision-Recall
        
        Args:
            y_true: Valores verdadeiros
            y_pred_proba: Probabilidades
        
        Returns:
            Dicionário com precision, recall, thresholds
        """
        try:
            # Para classificação binária
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
            else:
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            
            logger.info(f"✅ Precision-Recall Curve calculada")
            
            return {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds
            }
        except Exception as e:
            logger.error(f"❌ Erro ao calcular Precision-Recall curve: {str(e)}")
            return None
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Retorna resumo de todas as métricas
        
        Returns:
            DataFrame com resumo
        """
        if not self.metrics_history:
            logger.warning("⚠️  Nenhuma métrica no histórico")
            return None
        
        df = pd.DataFrame(self.metrics_history)
        
        summary = df.describe()
        
        logger.info(f"📊 Resumo estatístico das métricas:")
        logger.info(f"\n{summary}")
        
        return summary
    
    def export_metrics(self, filepath: str) -> bool:
        """
        Exporta métricas para CSV
        
        Args:
            filepath: Caminho do arquivo
        
        Returns:
            bool: True se sucesso
        """
        try:
            if not self.metrics_history:
                logger.warning("⚠️  Nenhuma métrica para exportar")
                return False
            
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(filepath, index=False)
            
            logger.info(f"✅ Métricas exportadas: {filepath}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao exportar métricas: {str(e)}")
            return False