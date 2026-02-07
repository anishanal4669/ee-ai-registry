import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, asdict
import json

try:
    from langfuse import Langfuse
except ImportError:
    raise ImportError("Langfuse SDK is required for PromptRegistry. Please install with `pip install langfuse`.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PromptVersion:
    name: str
    version: str
    template: str
    variables: List[str]
    model_config: Optional[Dict[str, Any]] = None
    tags: List[str] = None
    description: str = ""
    created_at: str = ""
    created_by: str = ""
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
class PromptRegistry:
    def __init__(
            self,
            langfuse_public_key: Optional[str] = None,
            langfuse_secret_key: Optional[str] = None,
            langfuse_host: Optional[str] = None,
            enable_langfuse: bool = True
    ):
        self.enable_langfuse = enable_langfuse

        if enable_langfuse:
            self.langfuse = Langfuse(
                public_key=langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
                host=langfuse_host or os.getenv("LANGFUSE_HOST", "https://api.langfuse.com")
            )
            logger.info("Langfuse client initialized for PromptRegistry")
        else:
            self.langfuse = None
            logger.info("Langfuse integration disabled for PromptRegistry")
        self.prompt_versions: Dict[str, PromptVersion] = {}

    def create_prompt(
            self,
            name: str,
            template: str,
            variables: List[str],
            model_config: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
            description: str = "",
            created_by: str = "system"
        ) -> PromptVersion:
        version = 1

        if self.enable_langfuse:
            try:
                existing = self.langfuse.get_prompt(name)
                if existing:
                    all_versions = self.list_prompt_versions(name)
                    version = max([int(v["version"]) for v in all_versions]) + 1 if all_versions else 1
            except Exception as e:
                logger.warning(f"Failed to fetch existing prompt from Langfuse: {e}")

        prompt_version = PromptVersion(
            name=name,
            version=str(version),
            template=template,
            variables=variables,
            model_config=model_config or {},
            tags=tags or [],
            description=description,
            created_at=datetime.utcnow().isoformat(),
            created_by=created_by,
            status="active"
        )

        if self.enable_langfuse:
            try:
                self.langfuse.create_prompt(
                    name=name,
                    prompt=template,
                    config=model_config or {},
                    labels=tags or [],
                )
                logger.info(f"Prompt {name}@{version} created in Langfuse")
            except Exception as e:
                logger.error(f"Failed to create prompt in Langfuse: {e}")
                raise e
        
        if name not in self._local_cache:
            self._local_cache[name] = []
        self._local_cache[name].append(prompt_version)
        return prompt_version
    
    def get_prompt(
            self,
            name: str,
            version: Optional[int] = None,
            fallback_to_latest: bool = True
    ) -> Optional[PromptVersion]:
        if self.enable_langfuse:
            try:
                langfuse_prompt = self.langfuse.get_prompt(
                    name=name,
                    version=version,
                    fallback=fallback_to_latest
                )
                if langfuse_prompt:
                    return self._langfuse_to_prompt_version(langfuse_prompt)
            except Exception as e:
                logger.error(f"Failed to fetch prompt from Langfuse: {e}")
                
        if name in self._local_cache:
            versions = self._local_cache[name]

            if version is not None:
                for pv in versions:
                    if pv.version == version:
                        return pv
                    
            active_versions = [v for v in versions if v.status == "active"]
            if active_versions:
                return max(active_versions, key=lambda v: v.version)
            
        logger.warning(f"Prompt '{name}' version {version} not found")
        return None
    
    def update_prompt(
            self,
            name: str,
            version: str,
            template: Optional[str] = None,
            variables: Optional[List[str]] = None,
            model_config: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
            description: Optional[str] = None,
            status: Optional[str] = None,
            updated_by: str = "system"
    ) -> Optional[PromptVersion]:
        """Update an existing prompt version."""
        prompt = self.get_prompt(name, version)
        if not prompt:
            logger.warning(f"Prompt '{name}' version {version} not found for update.")
            return None
        if template is not None:
            prompt.template = template
        if variables is not None:
            prompt.variables = variables
        if model_config is not None:
            prompt.model_config = model_config
        if tags is not None:
            prompt.tags = tags
        if description is not None:
            prompt.description = description
        if status is not None:
            prompt.status = status
        prompt.created_at = datetime.utcnow().isoformat()
        prompt.created_by = updated_by
        # Update in Langfuse if enabled
        if self.enable_langfuse:
            try:
                self.langfuse.update_prompt(
                    name=name,
                    version=version,
                    prompt=prompt.template,
                    config=prompt.model_config or {},
                    labels=prompt.tags or [],
                )
                logger.info(f"Prompt {name}@{version} updated in Langfuse")
            except Exception as e:
                logger.error(f"Failed to update prompt in Langfuse: {e}")
        # Update local cache
        if name in self._local_cache:
            for idx, pv in enumerate(self._local_cache[name]):
                if pv.version == version:
                    self._local_cache[name][idx] = prompt
                    break
        return prompt

    def list_prompts(self) -> List[str]:
        """List all prompt names in the registry."""
        names = set(self._local_cache.keys())
        if self.enable_langfuse:
            try:
                langfuse_prompts = self.langfuse.list_prompts()
                names.update([p['name'] for p in langfuse_prompts])
            except Exception as e:
                logger.warning(f"Failed to list prompts from Langfuse: {e}")
        return list(names)

    def import_prompt(self, file_path: str, created_by: str = "import") -> PromptVersion:
        """Import a prompt from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        prompt_version = PromptVersion(**data)
        if prompt_version.name not in self._local_cache:
            self._local_cache[prompt_version.name] = []
        self._local_cache[prompt_version.name].append(prompt_version)
        if self.enable_langfuse:
            try:
                self.langfuse.create_prompt(
                    name=prompt_version.name,
                    prompt=prompt_version.template,
                    config=prompt_version.model_config or {},
                    labels=prompt_version.tags or [],
                )
                logger.info(f"Prompt {prompt_version.name}@{prompt_version.version} imported to Langfuse")
            except Exception as e:
                logger.error(f"Failed to import prompt to Langfuse: {e}")
        return prompt_version

    def export_prompt(self, name: str, version: Optional[str] = None, file_path: Optional[str] = None) -> Optional[str]:
        """Export a prompt version to a JSON file."""
        prompt = self.get_prompt(name, version)
        if not prompt:
            logger.warning(f"Prompt '{name}' version {version} not found for export.")
            return None
        export_path = file_path or f"{name}_v{prompt.version}.json"
        with open(export_path, 'w') as f:
            json.dump(prompt.to_dict(), f, indent=2)
        logger.info(f"Prompt {name}@{prompt.version} exported to {export_path}")
        return export_path