"""
EE AI Registry - Enterprise AI Model and Prompt Registry

This package provides centralized management for AI models and prompts
using MLflow and Langfuse respectively.
"""

from registry.model_registry import ModelRegistry, ModelConfig
from registry.prompt_registry import PromptRegistry, PromptVersion
from registry.routing_engine import (
    RoutingEngine,
    RoutingStrategy,
    ModelCapabilities,
    RoutingRequest,
    RoutingResult
)

__version__ = "0.1.0"

__all__ = [
    "ModelRegistry",
    "ModelConfig",
    "PromptRegistry",
    "PromptVersion",
    "RoutingEngine",
    "RoutingStrategy",
    "ModelCapabilities",
    "RoutingRequest",
    "RoutingResult",
]
