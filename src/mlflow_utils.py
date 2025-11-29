"""
Utilitários para interação com MLFlow
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLFlowClient:
    """Cliente para interação com MLFlow"""
    
    def __init__(
        self,
        tracking_uri: str = None,
        experiment_name: str = "heart-disease-prediction"
    ):
        """
        Inicializa cliente MLFlow
        
        Args:
            tracking_uri: URI do MLFlow tracking server
            experiment_name: Nome do experimento
        """
        self.tracking_uri = tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        self.experiment_name = experiment_name
        
        # Configurar MLFlow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Criar ou obter experimento
        try:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"✅ Experimento criado: {self.experiment_name}")
        except:
            self.experiment_id = mlflow.get_experiment_by_name(self.experiment_name).experiment_id
            logger.info(f"✅ Experimento existente: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
        
        # Cliente para operações avançadas
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        logger.info(f"MLFlow Client inicializado: {self.tracking_uri}")
    
    def start_run(self, run_name: str = None) -> mlflow.ActiveRun:
        """
        Inicia um novo run do MLFlow
        
        Args:
            run_name: Nome do run (opcional)
        
        Returns:
            ActiveRun do MLFlow
        """
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"🚀 Run iniciado: {run.info.run_id}")
        return run
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Registra parâmetros no MLFlow
        
        Args:
            params: Dicionário com parâmetros
        """
        mlflow.log_params(params)
        logger.info(f"📝 Parâmetros registrados: {len(params)} itens")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Registra métricas no MLFlow
        
        Args:
            metrics: Dicionário com métricas
        """
        mlflow.log_metrics(metrics)
        logger.info(f"📊 Métricas registradas: {len(metrics)} itens")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str = None
    ) -> None:
        """
        Registra modelo no MLFlow
        
        Args:
            model: Modelo sklearn
            artifact_path: Caminho do artefato
            registered_model_name: Nome para registro (opcional)
        """
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )
        logger.info(f"🤖 Modelo registrado: {artifact_path}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """
        Registra artefato (arquivo) no MLFlow
        
        Args:
            local_path: Caminho local do arquivo
            artifact_path: Caminho do artefato no MLFlow
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"📎 Artefato registrado: {local_path}")
    
    def end_run(self) -> None:
        """Finaliza o run atual do MLFlow"""
        mlflow.end_run()
        logger.info("✅ Run finalizado")
    
    def load_model(self, model_uri: str) -> Any:
        """
        Carrega modelo do MLFlow
        
        Args:
            model_uri: URI do modelo (ex: 'models:/model_name/1')
        
        Returns:
            Modelo carregado
        """
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"✅ Modelo carregado: {model_uri}")
            return model
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {str(e)}")
            return None
    
    def get_best_run(
        self,
        metric_name: str = "accuracy",
        order: str = "DESC"
    ) -> Optional[Dict]:
        """
        Retorna o melhor run baseado em uma métrica
        
        Args:
            metric_name: Nome da métrica
            order: Ordem (DESC ou ASC)
        
        Returns:
            Dicionário com informações do run
        """
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.{metric_name} {order}"],
                max_results=1
            )
            
            if runs:
                best_run = runs[0]
                logger.info(f"🏆 Melhor run: {best_run.info.run_id} ({metric_name}={best_run.data.metrics.get(metric_name)})")
                return {
                    'run_id': best_run.info.run_id,
                    'metrics': best_run.data.metrics,
                    'params': best_run.data.params,
                    'tags': best_run.data.tags
                }
            return None
        except Exception as e:
            logger.error(f"❌ Erro ao buscar melhor run: {str(e)}")
            return None
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model"
    ) -> Optional[str]:
        """
        Registra modelo no Model Registry
        
        Args:
            run_id: ID do run
            model_name: Nome do modelo
            artifact_path: Caminho do artefato
        
        Returns:
            Version do modelo registrado
        """
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            result = mlflow.register_model(model_uri, model_name)
            logger.info(f"✅ Modelo registrado: {model_name} v{result.version}")
            return result.version
        except Exception as e:
            logger.error(f"❌ Erro ao registrar modelo: {str(e)}")
            return None
    
    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str = "Production"
    ) -> bool:
        """
        Move modelo para um stage (Staging, Production, Archived)
        
        Args:
            model_name: Nome do modelo
            version: Versão do modelo
            stage: Stage de destino
        
        Returns:
            bool: True se sucesso
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"✅ Modelo {model_name} v{version} -> {stage}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao transicionar modelo: {str(e)}")
            return False
    
    def get_production_model(self, model_name: str) -> Optional[Any]:
        """
        Carrega modelo em produção
        
        Args:
            model_name: Nome do modelo
        
        Returns:
            Modelo carregado
        """
        try:
            model_uri = f"models:/{model_name}/Production"
            model = self.load_model(model_uri)
            return model
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo de produção: {str(e)}")
            return None
    
    def compare_runs(self, run_ids: list) -> Optional[Dict]:
        """
        Compara múltiplos runs
        
        Args:
            run_ids: Lista de IDs dos runs
        
        Returns:
            Dicionário com comparação
        """
        try:
            comparison = {}
            
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                comparison[run_id] = {
                    'metrics': run.data.metrics,
                    'params': run.data.params
                }
            
            logger.info(f"📊 Comparação de {len(run_ids)} runs")
            return comparison
        except Exception as e:
            logger.error(f"❌ Erro ao comparar runs: {str(e)}")
            return None