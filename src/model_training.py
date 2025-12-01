"""
Utilitários para treinamento de modelos
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Classe para treinamento de modelos"""
    
    def __init__(self):
        """Inicializa trainer"""
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        logger.info("ModelTrainer inicializado")
    
    def get_model(self, model_name: str, **kwargs) -> Any:
        """
        Retorna instância de modelo
        
        Args:
            model_name: Nome do modelo
            **kwargs: Hiperparâmetros do modelo
        
        Returns:
            Instância do modelo
        """
        # Mapa de aliases (random_forest_tuned -> random_forest)
        model_aliases = {
            'random_forest_tuned': 'random_forest',
            'random_forest_optimized': 'random_forest',
            'logistic_regression_tuned': 'logistic_regression',
            'svm_tuned': 'svm'
        }
        
        # Resolver alias
        resolved_name = model_aliases.get(model_name, model_name)
        
        models_dict = {
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svm': SVC,
            'logistic_regression': LogisticRegression,
            'decision_tree': DecisionTreeClassifier,
            'naive_bayes': GaussianNB,
            'knn': KNeighborsClassifier
        }
        
        if resolved_name not in models_dict:
            logger.error(f"❌ Modelo '{model_name}' não encontrado")
            return None
        
        # Parâmetros default
        default_params = {
            'random_forest': {'n_estimators': 100, 'random_state': 42},
            'gradient_boosting': {'n_estimators': 100, 'random_state': 42},
            'svm': {'kernel': 'rbf', 'probability': True, 'random_state': 42},
            'logistic_regression': {'max_iter': 1000, 'random_state': 42},
            'decision_tree': {'random_state': 42},
            'naive_bayes': {},
            'knn': {'n_neighbors': 5}
        }
        
        # Mesclar parâmetros default com kwargs
        params = {**default_params.get(resolved_name, {}), **kwargs}
        
        model = models_dict[resolved_name](**params)
        logger.info(f"🤖 Modelo criado: {model_name} com params: {params}")
        
        return model
    
    def train_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str = "model"
    ) -> Any:
        """
        Treina modelo
        
        Args:
            model: Instância do modelo
            X_train: Features de treino
            y_train: Target de treino
            model_name: Nome do modelo
        
        Returns:
            Modelo treinado
        """
        logger.info(f"🔄 Treinando {model_name}...")
        
        model.fit(X_train, y_train)
        
        # Score de treino
        train_score = model.score(X_train, y_train)
        logger.info(f"✅ {model_name} treinado! Score treino: {train_score:.4f}")
        
        # Salvar modelo
        self.models[model_name] = model
        
        return model
    
    def train_multiple_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        models_config: Dict[str, Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Treina múltiplos modelos
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            models_config: Dicionário com configurações dos modelos
        
        Returns:
            Dicionário com modelos treinados
        """
        if models_config is None:
            # Configuração padrão
            models_config = {
                'random_forest': {'n_estimators': 100},
                'gradient_boosting': {'n_estimators': 100},
                'logistic_regression': {},
                'svm': {'kernel': 'rbf'}
            }
        
        logger.info(f"🔄 Treinando {len(models_config)} modelos...")
        
        trained_models = {}
        
        for model_name, params in models_config.items():
            try:
                model = self.get_model(model_name, **params)
                trained_model = self.train_model(model, X_train, y_train, model_name)
                trained_models[model_name] = trained_model
            except Exception as e:
                logger.error(f"❌ Erro ao treinar {model_name}: {str(e)}")
        
        logger.info(f"✅ {len(trained_models)} modelos treinados com sucesso!")
        
        return trained_models
    
    def cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, float]:
        """
        Realiza validação cruzada
        
        Args:
            model: Modelo
            X: Features
            y: Target
            cv: Número de folds
            scoring: Métrica
        
        Returns:
            Dicionário com resultados
        """
        logger.info(f"🔄 Validação cruzada ({cv} folds, {scoring})...")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max()
        }
        
        logger.info(f"✅ CV Score: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        return results
    
    def hyperparameter_tuning(
        self,
        model: Any,
        param_grid: Dict[str, list],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Realiza busca de hiperparâmetros com GridSearchCV
        
        Args:
            model: Modelo base
            param_grid: Grade de parâmetros
            X_train: Features de treino
            y_train: Target de treino
            cv: Número de folds
            scoring: Métrica
        
        Returns:
            Tuple (melhor_modelo, melhores_params)
        """
        logger.info(f"🔄 Buscando melhores hiperparâmetros (GridSearchCV)...")
        logger.info(f"📊 Grade: {param_grid}")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"✅ Melhores params: {best_params}")
        logger.info(f"✅ Melhor score: {best_score:.4f}")
        
        return best_model, best_params
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: list
    ) -> Optional[pd.DataFrame]:
        """
        Retorna importância das features
        
        Args:
            model: Modelo treinado
            feature_names: Lista de nomes das features
        
        Returns:
            DataFrame com importância das features
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                logger.info(f"📊 Top 5 features mais importantes:")
                for idx, row in importance_df.head(5).iterrows():
                    logger.info(f"   {row['feature']}: {row['importance']:.4f}")
                
                return importance_df
            else:
                logger.warning("⚠️  Modelo não possui feature_importances_")
                return None
        except Exception as e:
            logger.error(f"❌ Erro ao obter feature importance: {str(e)}")
            return None
    
    def predict(
        self,
        model: Any,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Realiza predições
        
        Args:
            model: Modelo treinado
            X: Features
        
        Returns:
            Array com predições
        """
        predictions = model.predict(X)
        logger.info(f"✅ Predições realizadas: {len(predictions)} amostras")
        return predictions
    
    def predict_proba(
        self,
        model: Any,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Realiza predições de probabilidade
        
        Args:
            model: Modelo treinado
            X: Features
        
        Returns:
            Array com probabilidades
        """
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            logger.info(f"✅ Probabilidades calculadas: {len(probas)} amostras")
            return probas
        else:
            logger.warning("⚠️  Modelo não possui predict_proba")
            return None
    
    def train_article_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Treina os 6 modelos do artigo
        KNN, Random Forest, Logistic Regression, SVM, Gaussian NB, Decision Tree
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
        
        Returns:
            Dicionário com modelos treinados
        """
        logger.info("🔄 Treinando modelos do artigo...")
        
        article_models = {
            'knn': {'n_neighbors': 5},
            'random_forest': {'n_estimators': 100, 'max_depth': 10},
            'logistic_regression': {'max_iter': 1000, 'C': 1.0},
            'svm': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
            'naive_bayes': {},
            'decision_tree': {'max_depth': 10, 'min_samples_split': 5}
        }
        
        return self.train_multiple_models(X_train, y_train, article_models)
    
    def train_improved_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Treina modelos adicionais para melhorias
        Gradient Boosting, XGBoost (se disponível), etc.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
        
        Returns:
            Dicionário com modelos treinados
        """
        logger.info("🔄 Treinando modelos de melhoria...")
        
        improved_models = {
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5
            },
            'random_forest_tuned': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        }
        
        # Tentar adicionar XGBoost se disponível
        try:
            from xgboost import XGBClassifier
            
            xgb_model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            xgb_model.fit(X_train, y_train)
            self.models['xgboost'] = xgb_model
            logger.info("✅ XGBoost treinado!")
            
        except ImportError:
            logger.warning("⚠️  XGBoost não instalado. Instale com: pip install xgboost")
        
        return self.train_multiple_models(X_train, y_train, improved_models)
    
    def save_best_model(self, model: Any, model_name: str) -> None:
        """
        Salva melhor modelo
        
        Args:
            model: Modelo
            model_name: Nome do modelo
        """
        self.best_model = model
        self.best_model_name = model_name
        logger.info(f"🏆 Melhor modelo salvo: {model_name}")
        """
        Salva melhor modelo
        
        Args:
            model: Modelo
            model_name: Nome do modelo
        """
        self.best_model = model
        self.best_model_name = model_name
        logger.info(f"🏆 Melhor modelo salvo: {model_name}")