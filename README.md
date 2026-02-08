# EE AI Registry

Enterprise AI Model and Prompt Registry for AI Gateway integration.

## Overview

EE AI Registry provides a centralized registry for managing AI models and prompts across your enterprise AI infrastructure. It integrates with MLflow for model management and Langfuse for prompt versioning.

## Features

- **Model Registry**: Manage AI model versions, configurations, and metadata using MLflow
- **Prompt Registry**: Version and manage prompts using Langfuse
- **Routing Engine**: Intelligent routing based on model capabilities and requirements

## Installation

### As a Library (Recommended for ee-ai-gateway)

```bash
pip install -e /path/to/ee-ai-registry
```

Or add to requirements.txt:
```
-e ../ee-ai-registry
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Model Registry

```python
from registry import ModelRegistry, ModelConfig

# Initialize the registry
model_registry = ModelRegistry(
    tracking_uri="http://localhost:5000",
    enable_mlflow=True
)

# Register a model
model_config = ModelConfig(
    name="gpt-4",
    version="1.0.0",
    stage="production",
    model_type="chat",
    endpoint="https://api.openai.com/v1/chat/completions"
)
model_registry.register_model(model_config)

# Get a model
model = model_registry.get_model("gpt-4", version="1.0.0")
```

### Prompt Registry

```python
from registry import PromptRegistry, PromptVersion

# Initialize the registry
prompt_registry = PromptRegistry(
    langfuse_public_key="your_public_key",
    langfuse_secret_key="your_secret_key",
    enable_langfuse=True
)

# Create a prompt
prompt = PromptVersion(
    name="chat_greeting",
    version="1.0",
    template="Hello {name}, how can I help you today?",
    variables=["name"]
)
prompt_registry.create_prompt(prompt)

# Get a prompt
prompt = prompt_registry.get_prompt("chat_greeting", version="1.0")
```

### Routing Engine

```python
from registry import RoutingEngine

routing_engine = RoutingEngine(model_registry)
best_model = routing_engine.select_model(
    task_type="chat",
    requirements={"max_tokens": 4096}
)
```

## Configuration

Set the following environment variables:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_REGISTRY_URI=http://localhost:5000

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://api.langfuse.com
```

## Dependencies

- fastapi>=0.109.1
- pydantic>=2.5.3
- mlflow>=2.8.0
- langfuse>=2.0.0
- msal>=1.25.0

## License

MIT License
