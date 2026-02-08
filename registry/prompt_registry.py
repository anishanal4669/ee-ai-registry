import os
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import re

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
        self._local_cache: Dict[str, List[PromptVersion]] = {}

    def create_prompt(
            self,
            name_or_prompt: Union[str, PromptVersion],
            template: Optional[str] = None,
            variables: Optional[List[str]] = None,
            model_config: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
            description: str = "",
            created_by: str = "system"
        ) -> PromptVersion:
        # Handle both PromptVersion object and individual parameters
        if isinstance(name_or_prompt, PromptVersion):
            # If PromptVersion object is passed, use its attributes
            prompt_obj = name_or_prompt
            name = prompt_obj.name
            template = prompt_obj.template
            variables = prompt_obj.variables
            model_config = prompt_obj.model_config
            tags = prompt_obj.tags
            description = prompt_obj.description
            created_by = prompt_obj.created_by or "system"
        else:
            # If individual parameters are passed
            name = name_or_prompt
            if template is None:
                raise ValueError("template is required when creating prompt with individual parameters")
            if variables is None:
                # Try to extract variables from template
                variables = list(set(
                    re.findall(r'\{(\w+)\}', template) + 
                    re.findall(r'\$\{(\w+)\}', template)
                ))
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
    
    def list_prompt_versions(self, name: str) -> List[Dict[str, Any]]:
        """List all versions of a specific prompt."""
        versions = []
        
        # Get from local cache
        if name in self._local_cache:
            versions.extend([
                {"version": pv.version, "status": pv.status, "created_at": pv.created_at}
                for pv in self._local_cache[name]
            ])
        
        # Get from Langfuse if enabled
        if self.enable_langfuse:
            try:
                langfuse_versions = self.langfuse.list_prompt_versions(name)
                for lv in langfuse_versions:
                    versions.append({
                        "version": str(lv.get("version", "1")),
                        "status": "active",
                        "created_at": lv.get("created_at", "")
                    })
            except Exception as e:
                logger.warning(f"Failed to list prompt versions from Langfuse: {e}")
        
        return versions
    
    def render_prompt(
        self,
        name: str,
        variables: Dict[str, Any],
        version: Optional[str] = None
    ) -> Optional[str]:
        """
        Render a prompt template with the provided variables.
        
        Args:
            name: Name of the prompt
            variables: Dictionary of variables to substitute in the template
            version: Optional specific version (defaults to latest)
            
        Returns:
            Rendered prompt string or None if prompt not found
        """
        prompt = self.get_prompt(name, version)
        if not prompt:
            logger.warning(f"Prompt '{name}' not found for rendering")
            return None
        
        try:
            # Simple template substitution using string format
            # Support both {variable} and ${variable} formats
            rendered = prompt.template
            
            for var_name, var_value in variables.items():
                # Replace {variable} format
                rendered = rendered.replace(f"{{{var_name}}}", str(var_value))
                # Replace ${variable} format
                rendered = rendered.replace(f"${{{var_name}}}", str(var_value))
            
            # Check if any required variables are missing
            missing_vars = []
            for var in prompt.variables:
                if var not in variables:
                    missing_vars.append(var)
            
            if missing_vars:
                logger.warning(
                    f"Prompt '{name}' rendered with missing variables: {missing_vars}"
                )
            
            return rendered
            
        except Exception as e:
            logger.error(f"Failed to render prompt '{name}': {e}")
            return None
    
    def _langfuse_to_prompt_version(self, langfuse_prompt: Any) -> PromptVersion:
        """
        Convert a Langfuse prompt object to a PromptVersion.
        
        Args:
            langfuse_prompt: Langfuse prompt object
            
        Returns:
            PromptVersion instance
        """
        try:
            # Langfuse prompt object structure
            # Extract data from the Langfuse prompt object
            if hasattr(langfuse_prompt, 'name'):
                name = langfuse_prompt.name
            else:
                name = langfuse_prompt.get('name', 'unknown')
            
            if hasattr(langfuse_prompt, 'version'):
                version = str(langfuse_prompt.version)
            else:
                version = str(langfuse_prompt.get('version', '1'))
            
            if hasattr(langfuse_prompt, 'prompt'):
                template = langfuse_prompt.prompt
            else:
                template = langfuse_prompt.get('prompt', '')
            
            if hasattr(langfuse_prompt, 'config'):
                model_config = langfuse_prompt.config or {}
            else:
                model_config = langfuse_prompt.get('config', {})
            
            if hasattr(langfuse_prompt, 'labels'):
                tags = langfuse_prompt.labels or []
            else:
                tags = langfuse_prompt.get('labels', [])
            
            # Extract variables from template
            # Look for {variable} and ${variable} patterns
            variables = list(set(
                re.findall(r'\{(\w+)\}', template) + 
                re.findall(r'\$\{(\w+)\}', template)
            ))
            
            # Get metadata
            created_at = ""
            created_by = "langfuse"
            
            if hasattr(langfuse_prompt, 'created_at'):
                created_at = str(langfuse_prompt.created_at)
            elif isinstance(langfuse_prompt, dict) and 'created_at' in langfuse_prompt:
                created_at = str(langfuse_prompt['created_at'])
            
            if hasattr(langfuse_prompt, 'created_by'):
                created_by = langfuse_prompt.created_by
            elif isinstance(langfuse_prompt, dict) and 'created_by' in langfuse_prompt:
                created_by = langfuse_prompt['created_by']
            
            return PromptVersion(
                name=name,
                version=version,
                template=template,
                variables=variables,
                model_config=model_config,
                tags=tags,
                description="",
                created_at=created_at,
                created_by=created_by,
                status="active"
            )
            
        except Exception as e:
            logger.error(f"Failed to convert Langfuse prompt to PromptVersion: {e}")
            # Return a minimal PromptVersion as fallback
            return PromptVersion(
                name="unknown",
                version="1",
                template="",
                variables=[],
                status="active"
            )

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
    
    def archive_prompt(
            self,
            name: str,
            version: Optional[int] = None,
        ):
        if name in self._local_cache:
            for prompt_version in self._local_cache[name]:
                if version is None or prompt_version.version == version:
                    prompt_version.status = "archived"
                    self.update_prompt(
                        name=prompt_version.name,
                        version=prompt_version.version,
                        status="archived",
                        updated_by="system"
                    )
                    logger.info(f"Prompt {name}@{prompt_version.version} archived")

    def render_prompt(
            self,
            name: str,
            variables: Dict[str, Any],
            version: Optional[int] = None,
    ) -> str:
        prompt = self.get_prompt(name, version)
        if not prompt:
            raise ValueError(f"Prompt '{name}' version {version} not found for rendering.")
        
        template = prompt.template
        for var_name, var_value in variables.items():
            placeholder = "{" + var_name + "}"
            template = template.replace(placeholder, str(var_value))

    def _langfuse_to_prompt_version(self, langfuse_prompt: Dict[str, Any]) -> PromptVersion:
        return PromptVersion(
            name=langfuse_prompt.get("name"),
            version=str(langfuse_prompt.get("version")),
            template=langfuse_prompt.get("prompt"),
            variables=langfuse_prompt.get("config", {}).get("variables", []),
            model_config=langfuse_prompt.get("config", {}),
            tags=langfuse_prompt.get("labels", []),
            description=langfuse_prompt.get("description", ""),
            created_at=langfuse_prompt.get("created_at", ""),
            created_by=langfuse_prompt.get("created_by", ""),
            status=langfuse_prompt.get("status", "active")
        )