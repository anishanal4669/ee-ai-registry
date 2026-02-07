import os
import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
from dataclasses import dataclass, asdict
import json

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities.model_registry import ModelVersion
except ImportError:
    raise ImportError("mlflow is required for the ModelRegistry. Please install it with 'pip install mlflow'.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Data class for model configuration"""
    name: str
    version: str
    stage: str
    model_type: str
    endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    parameters: Dict[str, Any] = None
    tags: Dict[str, str] = None
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary"""
        return asdict(self)
    
class ModelRegistry:
    def __init__(
            self,
            tracking_uri: Optional[str] = None,
            registry_uri: Optional[str] = None,
            enable_mlflow: bool = True
    ):
        self.enable_mlflow = enable_mlflow
        if self.enable_mlflow:
            self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            self.registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI", self.tracking_uri)
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_registry_uri(self.registry_uri)
            self.client = MlflowClient(tracking_uri=self.tracking_uri, registry_uri=self.registry_uri)
            logger.info(f"Connected to MLflow Tracking URI: {self.tracking_uri} and Registry URI: {self.registry_uri}")
        else:
            self.client = None
            logger.warning("MLflow integration is disabled. ModelRegistry will not function properly.")

        self._local_cache: Dict[str, List[ModelConfig]] = {}

    def register_model(
            self,
            name: str,
            model_type: str,
            endpoint: Optional[str] = None,
            deployment_name: Optional[str] = None,
            parameters: Dict[str, Any] = None,
            tags: Dict[str, str] = None,
            description: str = "",
            run_id: Optional[str] = None,
            artifact_path: str = "model"
    ) -> ModelConfig:
        version = "1"

        if self.enable_mlflow:
            try:
                try:
                    existing_versions = self.client.search_model_versions(f"name='{name}'")
                    if existing_versions:
                        latest_version = max(int(v.version) for v in existing_versions)
                        version = str(latest_version + 1)
                except Exception as e:
                    pass

                if run_id:
                    model_uri = f"runs:/{run_id}/{artifact_path}"
                    result = mlflow.register_model(model_uri, name)
                    version = result.version
                    logger.info(f"Registered model '{name}' version '{version}' from run '{run_id}'")
                else:
                    try:
                        self.client.create_registered_model(name, description=description)
                    except Exception as e:
                        pass
                
                    result = self.client.create_model_version(
                        name=name,
                        source="registry",
                        run_id=run_id or "no-run",
                        description=description,
                    )
                    version = result.version
                    logger.info(f"Registered model '{name}' version '{version}' without run association")
                if tags:
                    for key, value in tags.items():
                        self.client.set_model_version_tag(name, version, key, value)
                
                config_data = {
                    "model_type": model_type,
                    "endpoint": endpoint,
                    "deployment_name": deployment_name,
                    "parameters": parameters or {},
                }
                self.client.set_model_version_tag(name, version, "config", json.dumps(config_data))
            except Exception as e:
                logger.error(f"Failed to register model '{name}': {e}")
            
        model_config = ModelConfig(
            name=name,
            version=version,
            stage="None",
            model_type=model_type,
            endpoint=endpoint,
            deployment_name=deployment_name,
            parameters=parameters or {},
            tags=tags or {},
            description=description,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
            status="active"
        )

        if name not in self._local_cache:
            self._local_cache[name] = []
        self._local_cache[name].append(model_config)
        
        logger.info(f"Model '{name}' version '{version}' registered successfully with local cache")
        return model_config
    
    def get_model(
            self,
            name: str,
            version: Optional[str] = None,
            stage: Optional[str] = None
    ) -> Optional[ModelConfig]:
        if self.enable_mlflow:
            try:
                if version:
                    mv = self.client.get_model_version(name, version)
                elif stages:
                    versions = self.client.get_latest_versions(name, stages=[stage])
                    mv = versions[0] if versions else None
                else:
                    versions = self.client.search_model_versions(f"name='{name}'")
                    if versions:
                        mv = max(versions, key=lambda v: int(v.version))
                    else:
                        mv = None
                if mv:
                    return self._mlflow_to_model_config(mv)
            except Exception as e:
                logger.error(f"Failed to retrieve model '{name}': {e}")
            
        if name in self._local_cache:
            models = self._local_cache[name]
            if version:
                for mc in models:
                    if mc.version == version:
                        return mc
            elif stage:
                for mc in models:
                    if mc.stage == stage:
                        return mc
            if models:
                return max(models, key=lambda mc: int(mc.version))
            
        logger.warning(f"Model '{name}' not found in registry or local cache")
        return None
    
    def get_production_model(self, name: str) -> Optional[ModelConfig]:
        return self.get_model(name, stage="Production")
    
    def transition_stage(
            self,
            name: str,
            version: str,
            stage: Literal["Staging", "Production", "Archived"]
    ):
        if self.enable_mlflow:
            try:
                self.client.transition_model_version_stage(
                    name=name, 
                    version=version,
                    stage=stage,
                    archive_existing_versions=archive_existing
                )
                logger.info(f"Transitioned model '{name}' version '{version}' to stage '{stage}'")
            except Exception as e:
                logger.error(f"Failed to transition model '{name}' to stage '{stage}': {e}")

        if name in self._local_cache:
            models = self._local_cache[name]
            for mc in models:
                if mc.version == version:
                    mc.stage = stage
                    mc.updated_at = datetime.utcnow().isoformat()
                    return mc
        raise ValueError(f"Model '{name}' version '{version}' not found for stage transition")
    
    def update_model_config(
            self,
            name: str,
            version: str,
            endpoint: Optional[str] = None,
            deployment_name: Optional[str] = None,
            parameters: Optional[Dict[str, Any]] = None,
            tags: Optional[Dict[str, str]] = None,
            description: Optional[str] = None
    ) -> ModelConfig:
        
        model = self.get_model(name, version)
        if not model:
            raise ValueError(f"Model '{name}' version '{version}' not found for update")
        
        if endpoint:
            model.endpoint = endpoint
        if deployment_name:
            model.deployment_name = deployment_name
        if parameters:
            model.parameters.update(parameters)
        if tags:
            model.tags.update(tags)
        if description:
            model.description = description
        model.updated_at = datetime.utcnow().isoformat()

        if self.enable_mlflow:
            try:
                if description:
                    self.client.update_model_version(
                        name=name,
                        version=version,
                        description=description
                    )
                if tags:
                    for key, value in tags.items():
                        self.client.set_model_version_tag(name, version, key, value)
                config_data = {
                    "model_type": model.model_type,
                    "endpoint": model.endpoint,
                    "deployment_name": model.deployment_name,
                    "parameters": model.parameters,
                }
                self.client.set_model_version_tag(name, version, "config", json.dumps(config_data))
            except Exception as e:
                logger.error(f"Failed to update model '{name}' version '{version}': {e}")
        
        logger.info(f"Model '{name}' version '{version}' updated successfully")
        return model
    
    def list_model_versions(
            self,
            name: str,
            stage: Optional[str] = None
    ) -> List[ModelConfig]:
        if self.enable_mlflow:
            try:
                if stage:
                    versions = self.client.get_latest_versions(name, stages=[stage])
                else:
                    versions = self.client.search_model_versions(f"name='{name}'")
                return [self._mlflow_to_model_config(mv) for mv in versions]
            except Exception as e:
                logger.error(f"Failed to list versions for model '{name}': {e}")

        if name in self._local_cache:
            models = self._local_cache[name]
            if stage:
                models = [mc for mc in models if mc.stage == stage]
            return sorted(models, key=lambda mc: int(mc.version))
        return []
    
    def delete_model_version(
            self,
            name: str,
            version: str
    ):
        if self.enable_mlflow:
            try:
                self.client.delete_model_version(name, version)
                logger.info(f"Deleted model '{name}' version '{version}' from MLflow registry")
            except Exception as e:
                logger.error(f"Failed to delete model '{name}' version '{version}': {e}")

        if name in self._local_cache:
            models = self._local_cache[name]
            self._local_cache[name] = [mc for mc in models if mc.version != version]
            logger.info(f"Deleted model '{name}' version '{version}' from local cache")
    
    def archive_model(self,name: str,version: str):
        return self.transition_stage(name, version, stage="Archived")
    
    def compare_models(
            self,
            name: str,
            version1: str,
            version2: str
    ) -> Dict[str, Any]:
        model1 = self.get_model(name, version1)
        model2 = self.get_model(name, version2)
        if not model1 or not model2:
            raise ValueError(f"One or both models '{name}' versions '{version1}' and '{version2}' not found for comparison")
        
        return {
            "version1": model1.to_dict(),
            "version2": model2.to_dict(),
            "differences": {
                "stage": model1.stage != model2.stage,
                "endpoint": model1.endpoint != model2.endpoint,
                "deployment_name": model1.deployment_name != model2.deployment_name,
                "parameters": model1.parameters != model2.parameters,
            }
        }
   
    def export_model(self, name: str, version: str) -> Dict[str, Any]:
            model = self.get_model(name, version)
            if not model:
                raise ValueError(f"Model '{name}' version '{version}' not found for export")
            return model.to_dict()
    
    def import_model(self, model_data: Dict[str, Any]) -> ModelConfig:
        return self.register_model(
            name=model_data["name"],
            model_type=model_data["model_type"],
            endpoint=model_data.get("endpoint"),
            deployment_name=model_data.get("deployment_name"),
            parameters=model_data.get("parameters"),
            tags=model_data.get("tags"),
            description=model_data.get("description", ""),
        )

    def _mlflow_to_model_config(self, mlflow_version: ModelVersion) -> ModelConfig:
        config_str = mlflow_version.tags.get("config", "{}")

        try:
            config_str = mlflow_version.tags.get("config", "{}")
        except:
            config_data = {}

        return ModelConfig(
            name=mlflow_version.name,
            version=mlflow_version.version,
            stage=mlflow_version.current_stage,
            model_type=config_data.get("model_type", "unknown"),
            endpoint=config_data.get("endpoint"),
            deployment_name=config_data.get("deployment_name"),
            parameters=config_data.get("parameters", {}),
            tags={k: v for k, v in mlflow_version.tags.items() if k != "config"},
            description=mlflow_version.description or "",
            created_at=mlflow_version.creation_timestamp.isoformat(),
            updated_at=mlflow_version.last_updated_timestamp.isoformat(),
            status="active" if mlflow_version.current_stage != "Archived" else "archived"
        )
