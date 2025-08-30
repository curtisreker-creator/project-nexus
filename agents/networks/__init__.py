# File: agents/networks/__init__.py
"""
Project NEXUS Neural Networks Package
Complete neural architecture for multi-agent reinforcement learning
FINAL FIXED VERSION - All import and type issues resolved
"""

import warnings

# Core network components with error handling
try:
    from .spatial_cnn import SpatialCNN, EnhancedSpatialCNN
    SPATIAL_CNN_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Could not import spatial CNN modules: {e}")
    SpatialCNN = None
    EnhancedSpatialCNN = None
    SPATIAL_CNN_AVAILABLE = False

try:
    from .multimodal_fusion import (
        MultiModalFusion, AgentStateEncoder, AttentionalFusion, 
        prepare_agent_state_batch, validate_agent_dict
    )
    FUSION_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Could not import fusion modules: {e}")
    MultiModalFusion = None
    AgentStateEncoder = None
    AttentionalFusion = None
    prepare_agent_state_batch = None
    validate_agent_dict = None
    FUSION_AVAILABLE = False

try:
    from .ppo_networks import (
        PPOActorCritic, PPOActorCriticWithComm, PolicyHead, ValueHead,
        CommunicationModule, create_ppo_network
    )
    PPO_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Could not import PPO network modules: {e}")
    PPOActorCritic = None
    PPOActorCriticWithComm = None
    PolicyHead = None
    ValueHead = None
    CommunicationModule = None
    create_ppo_network = None
    PPO_AVAILABLE = False

# Network factory and utilities
try:
    from .network_factory import (
        NetworkFactory,
        create_lightweight_network,
        create_standard_network, 
        create_advanced_network,
        create_performance_network,
        create_custom_network
    )
    FACTORY_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Could not import network factory: {e}")
    NetworkFactory = None
    create_lightweight_network = None
    create_standard_network = None
    create_advanced_network = None
    create_performance_network = None
    create_custom_network = None
    FACTORY_AVAILABLE = False

# Version and metadata
__version__ = "0.2.2"
__author__ = "Project NEXUS Team"
__description__ = "Neural architecture for multi-agent reinforcement learning"

# Package exports with conditional availability
__all__ = []

# Add available components to exports
if SPATIAL_CNN_AVAILABLE:
    __all__.extend(["SpatialCNN", "EnhancedSpatialCNN"])

if FUSION_AVAILABLE:
    __all__.extend([
        "MultiModalFusion", "AgentStateEncoder", "AttentionalFusion",
        "prepare_agent_state_batch", "validate_agent_dict"
    ])

if PPO_AVAILABLE:
    __all__.extend([
        "PPOActorCritic", "PPOActorCriticWithComm", "PolicyHead", 
        "ValueHead", "CommunicationModule", "create_ppo_network"
    ])

if FACTORY_AVAILABLE:
    __all__.extend([
        "NetworkFactory", "create_lightweight_network", "create_standard_network",
        "create_advanced_network", "create_performance_network", "create_custom_network"
    ])

# Configuration defaults
DEFAULT_NETWORK_CONFIG = {
    "spatial_channels": 5,
    "spatial_dim": 256,
    "state_dim": 128,
    "fusion_dim": 512,
    "action_dim": 14,
    "use_enhanced_cnn": False,
    "use_attention_fusion": False,
    "enable_communication": False
}

def get_network_info():
    """Get information about available network configurations"""
    return {
        "version": __version__,
        "components_available": {
            "spatial_cnn": SPATIAL_CNN_AVAILABLE,
            "fusion": FUSION_AVAILABLE,
            "ppo_networks": PPO_AVAILABLE,
            "factory": FACTORY_AVAILABLE
        },
        "total_exports": len(__all__),
        "presets": ["lightweight", "standard", "advanced", "performance"] if FACTORY_AVAILABLE else [],
        "default_config": DEFAULT_NETWORK_CONFIG,
        "parameter_ranges": {
            "lightweight": "~150K parameters",
            "standard": "~800K parameters", 
            "advanced": "~1.5M parameters",
            "performance": "~3M parameters"
        } if FACTORY_AVAILABLE else {}
    }

def validate_imports():
    """Validate that all required dependencies are available"""
    missing_deps = []
    
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import yaml
    except ImportError:
        missing_deps.append("pyyaml")
    
    if missing_deps:
        warnings.warn(f"Missing required dependencies: {', '.join(missing_deps)}")
        return False
    
    return True

def get_available_functions():
    """Get list of available functions based on successful imports"""
    available = []
    
    if create_standard_network is not None:
        available.append("create_standard_network")
    if create_lightweight_network is not None:
        available.append("create_lightweight_network") 
    if create_advanced_network is not None:
        available.append("create_advanced_network")
    if NetworkFactory is not None:
        available.append("NetworkFactory")
    if prepare_agent_state_batch is not None:
        available.append("prepare_agent_state_batch")
    
    return available

def create_default_network(device=None):
    """Create a network with safe defaults, handling import failures gracefully"""
    if create_standard_network is not None:
        try:
            # Add an assertion to satisfy the linter about the None check
            assert create_standard_network is not None
            return create_standard_network(device=device)
        except Exception as e:
            warnings.warn(f"Could not create standard network: {e}")
    
    if create_lightweight_network is not None:
        try:
            assert create_lightweight_network is not None
            return create_lightweight_network(device=device)
        except Exception as e:
            warnings.warn(f"Could not create lightweight network: {e}")
    
    warnings.warn("No network creation functions available")
    return None

def safe_create_network(preset="standard", **kwargs):
    """Safe network creation with fallback options"""
    factory_funcs = {
        "lightweight": create_lightweight_network,
        "standard": create_standard_network,
        "advanced": create_advanced_network,
        "performance": create_performance_network
    }
    
    # Try requested preset first
    if preset in factory_funcs and factory_funcs[preset] is not None:
        try:
            # We can assert here as well for consistency, though Pylance doesn't flag this one
            func = factory_funcs[preset]
            assert func is not None
            return func(**kwargs)
        except Exception as e:
            warnings.warn(f"Could not create {preset} network: {e}")
    
    # Try fallback options
    for fallback_preset, func in factory_funcs.items():
        if func is not None and fallback_preset != preset:
            try:
                warnings.warn(f"Falling back to {fallback_preset} network")
                return func(**kwargs)
            except Exception:
                continue
    
    warnings.warn("Could not create any network configuration")
    return None

# Export safe creation function
if FACTORY_AVAILABLE:
    __all__.append("safe_create_network")

# Initialize package
if __name__ != "__main__":
    deps_ok = validate_imports()
    info = get_network_info()
    
    if not deps_ok:
        warnings.warn("Some network dependencies are missing")
    else:
        available_components = sum(info["components_available"].values())
        if available_components > 0:
            print(f"Project NEXUS Networks v{__version__} loaded successfully")
            print(f"Available components: {available_components}/4")
        else:
            warnings.warn("No network components could be imported")