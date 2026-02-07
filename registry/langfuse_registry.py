import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LangfuseRegistry:
    """A small registry wrapper that persists prompt versions, model configs and routing
    rules locally and optionally emits records to Langfuse when LANGFUSE_API_KEY is set.

    Why this approach:
      - Keeps a simple on-disk canonical copy (JSON) that the gateway can read fast.
      - Optionally mirrors events to Langfuse for observability / auditing if available.

    Files created under `store_dir`:
      - prompts.json
      - models.json
      - routing.json
    """

    def __init__(self, store_dir: Optional[str] = None):
        self.store_dir = Path(store_dir or Path.cwd() / "registry_store")
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.prompts_file = self.store_dir / "prompts.json"
        self.models_file = self.store_dir / "models.json"
        self.routing_file = self.store_dir / "routing.json"

        # Load or init stores
        self._prompts = self._load_json(self.prompts_file) or {}
        self._models = self._load_json(self.models_file) or {}
        self._routing = self._load_json(self.routing_file) or {}

        # Langfuse integration (best-effort): if LANGFUSE_API_KEY is present we will
        # attempt to call into langfuse for audit/visibility. The code does not strictly
        # depend on Langfuse so the registry works offline.
        self.langfuse_enabled = bool(os.getenv("LANGFUSE_API_KEY"))
        if self.langfuse_enabled:
            try:
                import langfuse

                # This is intentionally permissive: different orgs may initialize Langfuse
                # differently (host, api_key, public_key). We try to construct a client
                # if the SDK is available. If not available the registry still works.
                lf_host = os.getenv("LANGFUSE_HOST")
                lf_api_key = os.getenv("LANGFUSE_API_KEY")
                # SDK usage varies by version; attach client if available.
                try:
                    self.lf_client = langfuse.Client(api_key=lf_api_key, host=lf_host)  # type: ignore
                    logger.info("Langfuse client initialized for registry mirroring")
                except Exception:
                    # Some langfuse SDKs provide `Langfuse` or other entrypoints; we'll
                    # store the module and attempt to call common functions dynamically.
                    self.lf_client = langfuse
                    logger.info("Langfuse module imported (no typed client instantiated)")
            except Exception:
                self.langfuse_enabled = False
                self.lf_client = None
                logger.warning("Langfuse SDK not available or failed to initialize; continuing without it")
        else:
            self.lf_client = None

    # ---------- low-level helpers ----------
    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                return None
        return None

    def _save_json(self, path: Path, data: Dict[str, Any]):
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to write {path}: {e}")

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _emit_langfuse_event(self, event_type: str, payload: Dict[str, Any]):
        if not self.langfuse_enabled or not self.lf_client:
            return
        try:
            # Many Langfuse SDKs provide a flexible `log_event` / `track` API. Use a
            # best-effort dynamic call so this code doesn't hard-depend on a single
            # SDK shape. If your org uses Langfuse, replace this with the concrete
            # client method you use (for example, client.log_event(...)).
            if hasattr(self.lf_client, "log_event"):
                self.lf_client.log_event(event_type, payload)  # type: ignore
            elif hasattr(self.lf_client, "track"):
                self.lf_client.track(event_type, payload)  # type: ignore
            else:
                # try a conservative call to `send` or `create` if present
                for name in ("send", "create", "post"):
                    fn = getattr(self.lf_client, name, None)
                    if fn:
                        try:
                            fn(event_type, payload)  # type: ignore
                            break
                        except Exception:
                            continue
        except Exception as e:
            logger.debug(f"Langfuse mirror failed (non-fatal): {e}")

    # ---------- prompts ----------
    def register_prompt(self, prompt_name: str, content: str, version: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register a prompt (content) under `prompt_name`. Version is optional; if not
        provided a monotonically-incremented numeric version will be assigned (as string).
        Returns the stored prompt record.
        """
        prompts = self._prompts.setdefault(prompt_name, [])
        if version is None:
            # compute next version
            try:
                latest = max(int(p["version"]) for p in prompts) if prompts else 0
            except Exception:
                latest = len(prompts)
            version = str(latest + 1)

        record = {
            "name": prompt_name,
            "version": str(version),
            "content": content,
            "metadata": metadata or {},
            "created_at": self._now(),
        }
        prompts.append(record)
        self._save_json(self.prompts_file, self._prompts)

        # mirror to Langfuse for audit/visibility (best-effort)
        try:
            self._emit_langfuse_event("prompt.registered", {"prompt": prompt_name, "version": version, "metadata": metadata or {}})
        except Exception:
            pass

        logger.info(f"Registered prompt {prompt_name}@{version}")
        return record

    def get_prompt(self, prompt_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        prompts = self._prompts.get(prompt_name, [])
        if not prompts:
            return None
        if version:
            for p in prompts:
                if p.get("version") == str(version):
                    return p
            return None
        # return latest
        try:
            return max(prompts, key=lambda p: int(p.get("version", 0)))
        except Exception:
            return prompts[-1]

    def list_prompt_versions(self, prompt_name: str) -> List[Dict[str, Any]]:
        return self._prompts.get(prompt_name, [])

    # ---------- model configs ----------
    def register_model_config(self, name: str, config: Dict[str, Any], version: Optional[str] = None) -> Dict[str, Any]:
        models = self._models.setdefault(name, [])
        if version is None:
            try:
                latest = max(int(m["version"]) for m in models) if models else 0
            except Exception:
                latest = len(models)
            version = str(latest + 1)

        record = {
            "name": name,
            "version": str(version),
            "config": config,
            "created_at": self._now(),
        }
        models.append(record)
        self._save_json(self.models_file, self._models)

        try:
            self._emit_langfuse_event("model.registered", {"model": name, "version": version, "config": config})
        except Exception:
            pass

        logger.info(f"Registered model config {name}@{version}")
        return record

    def get_model_config(self, name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        models = self._models.get(name, [])
        if not models:
            return None
        if version:
            for m in models:
                if m.get("version") == str(version):
                    return m
            return None
        try:
            return max(models, key=lambda m: int(m.get("version", 0)))
        except Exception:
            return models[-1]

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    # ---------- routing rules ----------
    def set_routing_rule(self, model_name: str, rule_id: str, rule: Dict[str, Any]) -> Dict[str, Any]:
        model_rules = self._routing.setdefault(model_name, {})
        model_rules[rule_id] = {"rule": rule, "updated_at": self._now()}
        self._save_json(self.routing_file, self._routing)
        try:
            self._emit_langfuse_event("routing.rule.updated", {"model": model_name, "rule_id": rule_id, "rule": rule})
        except Exception:
            pass
        logger.info(f"Set routing rule {rule_id} for model {model_name}")
        return model_rules[rule_id]

    def get_routing_rule(self, model_name: str, rule_id: str) -> Optional[Dict[str, Any]]:
        return self._routing.get(model_name, {}).get(rule_id)

    def list_routing_rules(self, model_name: str) -> Dict[str, Any]:
        return self._routing.get(model_name, {})


__all__ = ["LangfuseRegistry"]
