"""
Routing Engine for AI Model Selection

This module provides intelligent routing logic for selecting the best AI model
based on request requirements, model capabilities, load, cost, rules, and other factors.
"""

import logging
from typing import Dict, Any, List, Optional, Literal, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import random
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Available routing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_COST = "least_cost"
    LEAST_LATENCY = "least_latency"
    CAPABILITY_MATCH = "capability_match"
    WEIGHTED_RANDOM = "weighted_random"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"
    FALLBACK_CHAIN = "fallback_chain"
    RULE_BASED = "rule_based"  # New: Rule-based routing


@dataclass
class ModelCapabilities:
    """Defines capabilities and limits of a model"""
    model_name: str
    model_type: str  # chat, completion, embedding, etc.
    max_tokens: int = 4096
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False
    context_window: int = 4096
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0
    avg_latency_ms: float = 1000.0
    rate_limit_rpm: int = 100  # Requests per minute
    rate_limit_tpm: int = 100000  # Tokens per minute
    availability: float = 99.9  # Percentage
    priority: int = 1  # Lower number = higher priority
    tags: List[str] = field(default_factory=list)


@dataclass
class RoutingRequest:
    """Request for model routing"""
    task_type: str  # chat, completion, embedding, etc.
    required_capabilities: Dict[str, Any] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    streaming: bool = False
    cost_preference: Literal["low", "medium", "high"] = "medium"
    latency_preference: Literal["low", "medium", "high"] = "medium"
    user_group: Optional[str] = None  # free, pro, enterprise
    user_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    exclude_models: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingResult:
    """Result of routing decision"""
    selected_model: str
    fallback_models: List[str] = field(default_factory=list)
    routing_strategy: str = ""
    rule_matched: Optional[str] = None  # Name of matched rule if rule-based
    confidence: float = 1.0
    reason: str = ""
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingRule:
    """
    Defines a routing rule with conditions and actions
    
    A rule consists of:
    - name: Unique identifier
    - priority: Lower number = higher priority (evaluated first)
    - conditions: List of conditions that must be met
    - target_model: Model to route to if conditions match
    - enabled: Whether the rule is active
    """
    name: str
    priority: int
    conditions: List[Dict[str, Any]]  # List of condition dictionaries
    target_model: str
    fallback_models: List[str] = field(default_factory=list)
    enabled: bool = True
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RoutingRule':
        """Create from dictionary"""
        return cls(**data)


class RoutingEngine:
    """
    Intelligent routing engine for AI model selection
    
    The routing engine selects the best model for a request based on:
    - Configurable routing rules
    - Model capabilities and limitations
    - Request requirements
    - Cost and latency preferences
    - Load balancing considerations
    - Fallback strategies
    """
    
    def __init__(self, model_registry=None):
        """
        Initialize the routing engine
        
        Args:
            model_registry: Optional ModelRegistry instance for dynamic model discovery
        """
        self.model_registry = model_registry
        self.model_capabilities: Dict[str, ModelCapabilities] = {}
        self.routing_history: List[Dict[str, Any]] = []
        self.model_health: Dict[str, Dict[str, Any]] = {}
        self.routing_rules: Dict[str, RoutingRule] = {}  # name -> rule
        self.custom_conditions: Dict[str, Callable] = {}  # name -> condition function
        
        # Round-robin state
        self._round_robin_index: Dict[str, int] = {}
        
        # Statistics
        self._model_usage_count: Dict[str, int] = {}
        self._rule_match_count: Dict[str, int] = {}
    
    # ================== Rule Management ==================
    
    def add_rule(self, rule: RoutingRule) -> RoutingRule:
        """
        Add a routing rule
        
        Args:
            rule: RoutingRule to add
            
        Returns:
            The added rule
            
        Raises:
            ValueError: If rule name already exists
        """
        if rule.name in self.routing_rules:
            raise ValueError(f"Rule '{rule.name}' already exists")
        
        # Validate the rule
        self.validate_rule(rule)
        
        self.routing_rules[rule.name] = rule
        logger.info(f"Added routing rule: {rule.name} (priority: {rule.priority})")
        return rule
    
    def remove_rule(self, name: str) -> bool:
        """
        Remove a routing rule
        
        Args:
            name: Name of the rule to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.routing_rules:
            del self.routing_rules[name]
            logger.info(f"Removed routing rule: {name}")
            return True
        return False
    
    def get_rule(self, name: str) -> Optional[RoutingRule]:
        """
        Get a routing rule by name
        
        Args:
            name: Name of the rule
            
        Returns:
            RoutingRule or None if not found
        """
        return self.routing_rules.get(name)
    
    def enable_rule(self, name: str) -> bool:
        """
        Enable a routing rule
        
        Args:
            name: Name of the rule
            
        Returns:
            True if enabled, False if not found
        """
        rule = self.routing_rules.get(name)
        if rule:
            rule.enabled = True
            rule.updated_at = datetime.utcnow().isoformat()
            logger.info(f"Enabled routing rule: {name}")
            return True
        return False
    
    def disable_rule(self, name: str) -> bool:
        """
        Disable a routing rule
        
        Args:
            name: Name of the rule
            
        Returns:
            True if disabled, False if not found
        """
        rule = self.routing_rules.get(name)
        if rule:
            rule.enabled = False
            rule.updated_at = datetime.utcnow().isoformat()
            logger.info(f"Disabled routing rule: {name}")
            return True
        return False
    
    def list_rules(self, enabled_only: bool = False) -> List[RoutingRule]:
        """
        List all routing rules
        
        Args:
            enabled_only: If True, only return enabled rules
            
        Returns:
            List of routing rules sorted by priority
        """
        rules = list(self.routing_rules.values())
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        # Sort by priority (lower number = higher priority)
        rules.sort(key=lambda r: r.priority)
        
        return rules
    
    def validate_rule(self, rule: RoutingRule) -> bool:
        """
        Validate a routing rule
        
        Args:
            rule: Rule to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If rule is invalid
        """
        # Check rule has a name
        if not rule.name:
            raise ValueError("Rule must have a name")
        
        # Check priority is valid
        if rule.priority < 0:
            raise ValueError("Rule priority must be >= 0")
        
        # Check target model exists
        if rule.target_model not in self.model_capabilities:
            logger.warning(
                f"Target model '{rule.target_model}' not registered in routing engine"
            )
        
        # Check conditions are valid
        if not rule.conditions:
            logger.warning(f"Rule '{rule.name}' has no conditions")
        
        for condition in rule.conditions:
            if "field" not in condition:
                raise ValueError(f"Condition missing 'field' in rule '{rule.name}'")
            if "operator" not in condition:
                raise ValueError(f"Condition missing 'operator' in rule '{rule.name}'")
        
        return True
    
    def register_custom_condition(
        self, 
        name: str, 
        condition_func: Callable[[RoutingRequest, Any], bool]
    ):
        """
        Register a custom condition function
        
        Args:
            name: Name of the custom condition
            condition_func: Function that takes (request, value) and returns bool
            
        Example:
            def is_premium_user(request, value):
                return request.user_group in ["pro", "enterprise"]
            
            engine.register_custom_condition("is_premium_user", is_premium_user)
        """
        self.custom_conditions[name] = condition_func
        logger.info(f"Registered custom condition: {name}")
    
    def export_rules(self, file_path: str):
        """
        Export routing rules to a JSON file
        
        Args:
            file_path: Path to save rules
        """
        rules_data = [rule.to_dict() for rule in self.routing_rules.values()]
        
        with open(file_path, 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        logger.info(f"Exported {len(rules_data)} rules to {file_path}")
    
    def import_rules(self, file_path: str, overwrite: bool = False):
        """
        Import routing rules from a JSON file
        
        Args:
            file_path: Path to load rules from
            overwrite: If True, overwrite existing rules with same name
        """
        with open(file_path, 'r') as f:
            rules_data = json.load(f)
        
        imported_count = 0
        skipped_count = 0
        
        for rule_data in rules_data:
            rule = RoutingRule.from_dict(rule_data)
            
            if rule.name in self.routing_rules and not overwrite:
                logger.warning(f"Skipping existing rule: {rule.name}")
                skipped_count += 1
                continue
            
            try:
                self.validate_rule(rule)
                self.routing_rules[rule.name] = rule
                imported_count += 1
            except ValueError as e:
                logger.error(f"Failed to import rule '{rule.name}': {e}")
                skipped_count += 1
        
        logger.info(
            f"Imported {imported_count} rules, skipped {skipped_count} from {file_path}"
        )
    
    # ================== Model Management ==================
    
    def register_model_capabilities(self, capabilities: ModelCapabilities):
        """
        Register capabilities for a model
        
        Args:
            capabilities: ModelCapabilities instance
        """
        self.model_capabilities[capabilities.model_name] = capabilities
        logger.info(f"Registered capabilities for model: {capabilities.model_name}")
    
    def update_model_health(
        self, 
        model_name: str, 
        is_healthy: bool = True,
        latency_ms: Optional[float] = None,
        error_rate: Optional[float] = None
    ):
        """
        Update health status for a model
        
        Args:
            model_name: Name of the model
            is_healthy: Whether the model is healthy
            latency_ms: Current latency in milliseconds
            error_rate: Current error rate (0-1)
        """
        if model_name not in self.model_health:
            self.model_health[model_name] = {
                "is_healthy": True,
                "last_check": datetime.now(),
                "consecutive_failures": 0,
                "recent_latencies": [],
                "recent_errors": []
            }
        
        health = self.model_health[model_name]
        health["is_healthy"] = is_healthy
        health["last_check"] = datetime.now()
        
        if not is_healthy:
            health["consecutive_failures"] += 1
        else:
            health["consecutive_failures"] = 0
        
        if latency_ms is not None:
            health["recent_latencies"].append(latency_ms)
            # Keep only last 100 latencies
            health["recent_latencies"] = health["recent_latencies"][-100:]
        
        if error_rate is not None:
            health["recent_errors"].append(error_rate)
            health["recent_errors"] = health["recent_errors"][-100:]
    
    # ================== Routing Logic ==================
    
    def route(
        self, 
        request: RoutingRequest,
        strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH,
        include_fallbacks: bool = True
    ) -> RoutingResult:
        """
        Route a request to the best model
        
        Args:
            request: RoutingRequest with requirements
            strategy: Routing strategy to use
            include_fallbacks: Whether to include fallback models
            
        Returns:
            RoutingResult with selected model and metadata
        """
        # Try rule-based routing first if strategy is RULE_BASED
        if strategy == RoutingStrategy.RULE_BASED:
            result = self._route_by_rules(request)
            if result:
                if include_fallbacks and not result.fallback_models:
                    eligible_models = self._filter_eligible_models(request)
                    result.fallback_models = self._get_fallback_models(
                        result.selected_model, eligible_models
                    )
                self._record_routing(request, result)
                return result
            else:
                # Fall back to capability match if no rules match
                logger.warning("No rules matched, falling back to capability match")
                strategy = RoutingStrategy.CAPABILITY_MATCH
        
        # Get eligible models
        eligible_models = self._filter_eligible_models(request)
        
        if not eligible_models:
            raise ValueError(f"No eligible models found for request type: {request.task_type}")
        
        # Apply routing strategy
        if strategy == RoutingStrategy.ROUND_ROBIN:
            result = self._route_round_robin(eligible_models, request)
        elif strategy == RoutingStrategy.LEAST_COST:
            result = self._route_least_cost(eligible_models, request)
        elif strategy == RoutingStrategy.LEAST_LATENCY:
            result = self._route_least_latency(eligible_models, request)
        elif strategy == RoutingStrategy.CAPABILITY_MATCH:
            result = self._route_capability_match(eligible_models, request)
        elif strategy == RoutingStrategy.WEIGHTED_RANDOM:
            result = self._route_weighted_random(eligible_models, request)
        elif strategy == RoutingStrategy.PRIORITY_BASED:
            result = self._route_priority_based(eligible_models, request)
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            result = self._route_load_balanced(eligible_models, request)
        elif strategy == RoutingStrategy.FALLBACK_CHAIN:
            result = self._route_fallback_chain(eligible_models, request)
        else:
            result = self._route_capability_match(eligible_models, request)
        
        # Add fallback models if requested
        if include_fallbacks and not result.fallback_models:
            result.fallback_models = self._get_fallback_models(
                result.selected_model, 
                eligible_models
            )
        
        # Record routing decision
        self._record_routing(request, result)
        
        return result
    
    def _route_by_rules(self, request: RoutingRequest) -> Optional[RoutingResult]:
        """
        Route based on configured rules
        
        Args:
            request: RoutingRequest
            
        Returns:
            RoutingResult if a rule matches, None otherwise
        """
        # Get enabled rules sorted by priority
        rules = self.list_rules(enabled_only=True)
        
        for rule in rules:
            if self._evaluate_conditions(rule.conditions, request):
                # Rule matched!
                self._rule_match_count[rule.name] = self._rule_match_count.get(rule.name, 0) + 1
                
                logger.info(
                    f"Rule '{rule.name}' matched for request (priority: {rule.priority})"
                )
                
                # Get model capabilities for cost/latency estimation
                capabilities = self.model_capabilities.get(rule.target_model)
                
                result = RoutingResult(
                    selected_model=rule.target_model,
                    fallback_models=rule.fallback_models.copy(),
                    routing_strategy="rule_based",
                    rule_matched=rule.name,
                    confidence=1.0,
                    reason=f"Matched rule '{rule.name}': {rule.description}",
                    estimated_cost=self._estimate_cost(capabilities, request) if capabilities else 0.0,
                    estimated_latency_ms=capabilities.avg_latency_ms if capabilities else 0.0,
                    metadata={"rule_priority": rule.priority}
                )
                
                return result
        
        # No rules matched
        return None
    
    def _evaluate_conditions(
        self, 
        conditions: List[Dict[str, Any]], 
        request: RoutingRequest
    ) -> bool:
        """
        Evaluate if all conditions are met
        
        Args:
            conditions: List of condition dictionaries
            request: RoutingRequest to evaluate against
            
        Returns:
            True if all conditions match
        """
        for condition in conditions:
            if not self._evaluate_comparison(condition, request):
                return False
        return True
    
    def _evaluate_comparison(
        self, 
        condition: Dict[str, Any], 
        request: RoutingRequest
    ) -> bool:
        """
        Evaluate a single condition
        
        Condition format:
        {
            "field": "user_group",
            "operator": "equals",
            "value": "enterprise"
        }
        
        Supported operators:
        - equals, not_equals
        - in, not_in
        - greater_than, less_than, greater_or_equal, less_or_equal
        - contains, not_contains
        - regex_match
        - custom (uses registered custom condition function)
        
        Args:
            condition: Condition dictionary
            request: RoutingRequest
            
        Returns:
            True if condition matches
        """
        field = condition.get("field")
        operator = condition.get("operator")
        expected_value = condition.get("value")
        
        # Get actual value from request
        if field.startswith("metadata."):
            # Access metadata fields
            metadata_field = field.split(".", 1)[1]
            actual_value = request.metadata.get(metadata_field)
        elif hasattr(request, field):
            actual_value = getattr(request, field)
        else:
            logger.warning(f"Unknown field '{field}' in condition")
            return False
        
        # Handle custom conditions
        if operator == "custom":
            custom_func_name = condition.get("custom_function")
            if custom_func_name in self.custom_conditions:
                return self.custom_conditions[custom_func_name](request, expected_value)
            else:
                logger.warning(f"Custom condition '{custom_func_name}' not registered")
                return False
        
        # Evaluate comparison
        try:
            if operator == "equals":
                return actual_value == expected_value
            elif operator == "not_equals":
                return actual_value != expected_value
            elif operator == "in":
                return actual_value in expected_value
            elif operator == "not_in":
                return actual_value not in expected_value
            elif operator == "greater_than":
                return actual_value > expected_value
            elif operator == "less_than":
                return actual_value < expected_value
            elif operator == "greater_or_equal":
                return actual_value >= expected_value
            elif operator == "less_or_equal":
                return actual_value <= expected_value
            elif operator == "contains":
                return expected_value in str(actual_value)
            elif operator == "not_contains":
                return expected_value not in str(actual_value)
            elif operator == "regex_match":
                return bool(re.match(expected_value, str(actual_value)))
            else:
                logger.warning(f"Unknown operator '{operator}' in condition")
                return False
        except Exception as e:
            logger.error(f"Error evaluating condition: {e}")
            return False
    
    def _filter_eligible_models(self, request: RoutingRequest) -> List[ModelCapabilities]:
        """
        Filter models based on request requirements
        
        Args:
            request: RoutingRequest
            
        Returns:
            List of eligible ModelCapabilities
        """
        eligible = []
        
        for model_name, capabilities in self.model_capabilities.items():
            # Skip excluded models
            if model_name in request.exclude_models:
                continue
            
            # Check if model is healthy
            if not self._is_model_healthy(model_name):
                logger.debug(f"Skipping unhealthy model: {model_name}")
                continue
            
            # Check task type
            if capabilities.model_type != request.task_type:
                continue
            
            # Check token requirements
            if request.max_tokens and request.max_tokens > capabilities.max_tokens:
                continue
            
            # Check streaming support
            if request.streaming and not capabilities.supports_streaming:
                continue
            
            # Check required capabilities
            if request.required_capabilities:
                if request.required_capabilities.get("function_calling") and not capabilities.supports_function_calling:
                    continue
                if request.required_capabilities.get("vision") and not capabilities.supports_vision:
                    continue
                if request.required_capabilities.get("json_mode") and not capabilities.supports_json_mode:
                    continue
                if request.required_capabilities.get("min_context_window"):
                    if capabilities.context_window < request.required_capabilities["min_context_window"]:
                        continue
            
            # Check tags
            if request.tags:
                if not any(tag in capabilities.tags for tag in request.tags):
                    continue
            
            eligible.append(capabilities)
        
        return eligible
    
    def _is_model_healthy(self, model_name: str) -> bool:
        """Check if a model is healthy"""
        if model_name not in self.model_health:
            return True  # Assume healthy if no health data
        
        health = self.model_health[model_name]
        # Consider unhealthy after 3 consecutive failures
        return health["is_healthy"] and health["consecutive_failures"] < 3
    
    # ================== Strategy Implementations ==================
    
    def _route_round_robin(
        self, 
        eligible_models: List[ModelCapabilities], 
        request: RoutingRequest
    ) -> RoutingResult:
        """Round-robin routing across eligible models"""
        key = request.task_type
        
        if key not in self._round_robin_index:
            self._round_robin_index[key] = 0
        
        index = self._round_robin_index[key] % len(eligible_models)
        selected = eligible_models[index]
        
        self._round_robin_index[key] += 1
        
        return RoutingResult(
            selected_model=selected.model_name,
            routing_strategy="round_robin",
            confidence=1.0,
            reason="Round-robin load balancing",
            estimated_cost=self._estimate_cost(selected, request),
            estimated_latency_ms=selected.avg_latency_ms
        )
    
    def _route_least_cost(
        self, 
        eligible_models: List[ModelCapabilities], 
        request: RoutingRequest
    ) -> RoutingResult:
        """Route to the least expensive model"""
        sorted_models = sorted(
            eligible_models,
            key=lambda m: m.cost_per_1k_input_tokens + m.cost_per_1k_output_tokens
        )
        
        selected = sorted_models[0]
        
        return RoutingResult(
            selected_model=selected.model_name,
            routing_strategy="least_cost",
            confidence=1.0,
            reason=f"Lowest cost: ${selected.cost_per_1k_input_tokens:.4f} input + ${selected.cost_per_1k_output_tokens:.4f} output per 1K tokens",
            estimated_cost=self._estimate_cost(selected, request),
            estimated_latency_ms=selected.avg_latency_ms
        )
    
    def _route_least_latency(
        self, 
        eligible_models: List[ModelCapabilities], 
        request: RoutingRequest
    ) -> RoutingResult:
        """Route to the fastest model"""
        # Use actual latency data if available
        def get_avg_latency(model: ModelCapabilities) -> float:
            if model.model_name in self.model_health:
                recent = self.model_health[model.model_name].get("recent_latencies", [])
                if recent:
                    return sum(recent) / len(recent)
            return model.avg_latency_ms
        
        sorted_models = sorted(eligible_models, key=get_avg_latency)
        selected = sorted_models[0]
        avg_latency = get_avg_latency(selected)
        
        return RoutingResult(
            selected_model=selected.model_name,
            routing_strategy="least_latency",
            confidence=1.0,
            reason=f"Lowest latency: {avg_latency:.2f}ms average",
            estimated_cost=self._estimate_cost(selected, request),
            estimated_latency_ms=avg_latency
        )
    
    def _route_capability_match(
        self, 
        eligible_models: List[ModelCapabilities], 
        request: RoutingRequest
    ) -> RoutingResult:
        """Route based on best capability match"""
        # Score each model based on how well it matches requirements
        scored_models = []
        
        for model in eligible_models:
            score = 0.0
            
            # Prefer models with exact capability matches
            if request.required_capabilities.get("function_calling") and model.supports_function_calling:
                score += 10
            if request.required_capabilities.get("vision") and model.supports_vision:
                score += 10
            if request.required_capabilities.get("json_mode") and model.supports_json_mode:
                score += 5
            
            # Consider cost preference
            cost_score = 1.0 / (1.0 + model.cost_per_1k_input_tokens + model.cost_per_1k_output_tokens)
            if request.cost_preference == "low":
                score += cost_score * 5
            elif request.cost_preference == "medium":
                score += cost_score * 2
            
            # Consider latency preference
            latency_score = 1000.0 / (model.avg_latency_ms + 1.0)
            if request.latency_preference == "low":
                score += latency_score * 5
            elif request.latency_preference == "medium":
                score += latency_score * 2
            
            # Prefer higher availability
            score += model.availability / 10
            
            # Tag matching
            if request.tags:
                tag_matches = sum(1 for tag in request.tags if tag in model.tags)
                score += tag_matches * 3
            
            scored_models.append((model, score))
        
        # Select model with highest score
        scored_models.sort(key=lambda x: x[1], reverse=True)
        selected = scored_models[0][0]
        
        return RoutingResult(
            selected_model=selected.model_name,
            routing_strategy="capability_match",
            confidence=min(scored_models[0][1] / 50.0, 1.0),  # Normalize to 0-1
            reason=f"Best capability match (score: {scored_models[0][1]:.2f})",
            estimated_cost=self._estimate_cost(selected, request),
            estimated_latency_ms=selected.avg_latency_ms
        )
    
    def _route_weighted_random(
        self, 
        eligible_models: List[ModelCapabilities], 
        request: RoutingRequest
    ) -> RoutingResult:
        """Route randomly with weights based on model quality"""
        # Calculate weights based on availability and inverse latency
        weights = []
        for model in eligible_models:
            weight = (model.availability / 100.0) * (1000.0 / (model.avg_latency_ms + 1.0))
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Select randomly
        selected = random.choices(eligible_models, weights=weights)[0]
        
        return RoutingResult(
            selected_model=selected.model_name,
            routing_strategy="weighted_random",
            confidence=0.8,
            reason="Weighted random selection based on availability and performance",
            estimated_cost=self._estimate_cost(selected, request),
            estimated_latency_ms=selected.avg_latency_ms
        )
    
    def _route_priority_based(
        self, 
        eligible_models: List[ModelCapabilities], 
        request: RoutingRequest
    ) -> RoutingResult:
        """Route based on model priority"""
        sorted_models = sorted(eligible_models, key=lambda m: m.priority)
        selected = sorted_models[0]
        
        return RoutingResult(
            selected_model=selected.model_name,
            routing_strategy="priority_based",
            confidence=1.0,
            reason=f"Highest priority model (priority: {selected.priority})",
            estimated_cost=self._estimate_cost(selected, request),
            estimated_latency_ms=selected.avg_latency_ms
        )
    
    def _route_load_balanced(
        self, 
        eligible_models: List[ModelCapabilities], 
        request: RoutingRequest
    ) -> RoutingResult:
        """Route based on current load (tracked via health metrics)"""
        # Select model with lowest recent error rate
        best_model = None
        best_score = float('inf')
        
        for model in eligible_models:
            if model.model_name in self.model_health:
                health = self.model_health[model.model_name]
                recent_errors = health.get("recent_errors", [])
                error_rate = sum(recent_errors) / len(recent_errors) if recent_errors else 0.0
                
                # Combine error rate with consecutive failures
                score = error_rate + (health["consecutive_failures"] * 0.1)
            else:
                score = 0.0  # No data = best score
            
            if best_model is None or score < best_score:
                best_model = model
                best_score = score
        
        selected = best_model if best_model else eligible_models[0]
        
        return RoutingResult(
            selected_model=selected.model_name,
            routing_strategy="load_balanced",
            confidence=1.0 - best_score,
            reason=f"Load-balanced selection (health score: {1.0 - best_score:.2f})",
            estimated_cost=self._estimate_cost(selected, request),
            estimated_latency_ms=selected.avg_latency_ms
        )
    
    def _route_fallback_chain(
        self, 
        eligible_models: List[ModelCapabilities], 
        request: RoutingRequest
    ) -> RoutingResult:
        """Route with explicit fallback chain based on priority"""
        sorted_models = sorted(eligible_models, key=lambda m: m.priority)
        selected = sorted_models[0]
        fallbacks = [m.model_name for m in sorted_models[1:]]
        
        return RoutingResult(
            selected_model=selected.model_name,
            fallback_models=fallbacks,
            routing_strategy="fallback_chain",
            confidence=1.0,
            reason=f"Priority-based with {len(fallbacks)} fallback(s)",
            estimated_cost=self._estimate_cost(selected, request),
            estimated_latency_ms=selected.avg_latency_ms
        )
    
    def _get_fallback_models(
        self, 
        selected_model: str, 
        eligible_models: List[ModelCapabilities]
    ) -> List[str]:
        """Get fallback models for a selected model"""
        # Sort by priority and exclude the selected model
        fallbacks = [
            m.model_name 
            for m in sorted(eligible_models, key=lambda m: m.priority)
            if m.model_name != selected_model
        ]
        
        # Return top 3 fallbacks
        return fallbacks[:3]
    
    def _estimate_cost(self, model: ModelCapabilities, request: RoutingRequest) -> float:
        """Estimate cost for a request"""
        if not request.max_tokens:
            estimated_tokens = 1000  # Default estimate
        else:
            estimated_tokens = request.max_tokens
        
        # Estimate (tokens / 1000) * cost_per_1k
        input_cost = (estimated_tokens / 2000) * model.cost_per_1k_input_tokens
        output_cost = (estimated_tokens / 2000) * model.cost_per_1k_output_tokens
        
        return input_cost + output_cost
    
    def _record_routing(self, request: RoutingRequest, result: RoutingResult):
        """Record routing decision for analytics"""
        self.routing_history.append({
            "timestamp": datetime.now(),
            "request_type": request.task_type,
            "selected_model": result.selected_model,
            "strategy": result.routing_strategy,
            "rule_matched": result.rule_matched,
            "confidence": result.confidence,
            "estimated_cost": result.estimated_cost,
            "estimated_latency_ms": result.estimated_latency_ms,
            "user_group": request.user_group,
            "user_id": request.user_id
        })
        
        # Update usage count
        self._model_usage_count[result.selected_model] = \
            self._model_usage_count.get(result.selected_model, 0) + 1
        
        # Keep only last 1000 routing decisions
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
    
    # ================== Analytics & Optimization ==================
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about routing decisions"""
        if not self.routing_history:
            return {}
        
        total_requests = len(self.routing_history)
        model_usage = {}
        strategy_usage = {}
        rule_usage = {}
        total_estimated_cost = 0.0
        
        for entry in self.routing_history:
            # Model usage
            model = entry["selected_model"]
            model_usage[model] = model_usage.get(model, 0) + 1
            
            # Strategy usage
            strategy = entry["strategy"]
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            # Rule usage
            if entry.get("rule_matched"):
                rule = entry["rule_matched"]
                rule_usage[rule] = rule_usage.get(rule, 0) + 1
            
            # Total cost
            total_estimated_cost += entry.get("estimated_cost", 0.0)
        
        return {
            "total_requests": total_requests,
            "model_usage": model_usage,
            "strategy_usage": strategy_usage,
            "rule_usage": rule_usage,
            "total_estimated_cost": total_estimated_cost,
            "avg_cost_per_request": total_estimated_cost / total_requests if total_requests > 0 else 0.0,
            "unique_models_used": len(model_usage),
            "total_rules": len(self.routing_rules),
            "enabled_rules": len([r for r in self.routing_rules.values() if r.enabled])
        }
    
    def get_model_usage_stats(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed usage statistics for models
        
        Args:
            model_name: Optional specific model name, otherwise all models
            
        Returns:
            Dictionary with usage statistics
        """
        stats = {}
        
        if model_name:
            models_to_check = [model_name]
        else:
            models_to_check = list(self.model_capabilities.keys())
        
        for model in models_to_check:
            # Count requests
            request_count = self._model_usage_count.get(model, 0)
            
            # Get health data
            health_data = self.model_health.get(model, {})
            recent_latencies = health_data.get("recent_latencies", [])
            recent_errors = health_data.get("recent_errors", [])
            
            # Calculate stats
            avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0.0
            avg_error_rate = sum(recent_errors) / len(recent_errors) if recent_errors else 0.0
            
            # Estimated cost from history
            model_cost = sum(
                entry.get("estimated_cost", 0.0)
                for entry in self.routing_history
                if entry["selected_model"] == model
            )
            
            stats[model] = {
                "request_count": request_count,
                "avg_latency_ms": avg_latency,
                "avg_error_rate": avg_error_rate,
                "total_estimated_cost": model_cost,
                "is_healthy": health_data.get("is_healthy", True),
                "consecutive_failures": health_data.get("consecutive_failures", 0),
                "usage_percentage": (request_count / len(self.routing_history) * 100) if self.routing_history else 0.0
            }
        
        return stats if not model_name else stats.get(model_name, {})
    
    def optimize_routing(self) -> Dict[str, Any]:
        """
        Analyze routing patterns and suggest optimizations
        
        Returns:
            Dictionary with optimization suggestions
        """
        suggestions = []
        stats = self.get_routing_stats()
        model_stats = self.get_model_usage_stats()
        
        # Check for underutilized models
        total_requests = stats.get("total_requests", 0)
        for model, model_stat in model_stats.items():
            usage_pct = model_stat.get("usage_percentage", 0)
            if usage_pct < 5 and total_requests > 100:
                suggestions.append({
                    "type": "underutilized_model",
                    "model": model,
                    "message": f"Model '{model}' used in only {usage_pct:.1f}% of requests",
                    "recommendation": "Consider removing this model or adjusting routing rules"
                })
        
        # Check for high error rate models
        for model, model_stat in model_stats.items():
            error_rate = model_stat.get("avg_error_rate", 0)
            if error_rate > 0.1:  # More than 10% error rate
                suggestions.append({
                    "type": "high_error_rate",
                    "model": model,
                    "error_rate": error_rate,
                    "message": f"Model '{model}' has high error rate: {error_rate*100:.1f}%",
                    "recommendation": "Check model health or reduce routing to this model"
                })
        
        # Check for cost optimization opportunities
        model_usage = stats.get("model_usage", {})
        if model_usage:
            # Find most used expensive models
            for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True)[:3]:
                cap = self.model_capabilities.get(model)
                if cap and (cap.cost_per_1k_input_tokens + cap.cost_per_1k_output_tokens) > 0.01:
                    # Check if there's a cheaper alternative
                    for alt_model, alt_cap in self.model_capabilities.items():
                        if (alt_model != model and 
                            alt_cap.model_type == cap.model_type and
                            (alt_cap.cost_per_1k_input_tokens + alt_cap.cost_per_1k_output_tokens) < 
                            (cap.cost_per_1k_input_tokens + cap.cost_per_1k_output_tokens) * 0.5):
                            suggestions.append({
                                "type": "cost_optimization",
                                "current_model": model,
                                "alternative_model": alt_model,
                                "message": f"Consider routing some traffic from '{model}' to cheaper '{alt_model}'",
                                "potential_savings": f"{((cap.cost_per_1k_input_tokens - alt_cap.cost_per_1k_input_tokens) * count):.2f}"
                            })
                            break
        
        # Check rule effectiveness
        rule_usage = stats.get("rule_usage", {})
        for rule_name, rule in self.routing_rules.items():
            if rule.enabled and rule_usage.get(rule_name, 0) == 0 and total_requests > 100:
                suggestions.append({
                    "type": "unused_rule",
                    "rule": rule_name,
                    "message": f"Rule '{rule_name}' never matched in recent requests",
                    "recommendation": "Review rule conditions or consider disabling"
                })
        
        return {
            "suggestions": suggestions,
            "total_suggestions": len(suggestions),
            "analyzed_requests": total_requests,
            "analyzed_models": len(model_stats)
        }
