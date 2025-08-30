"""
Network Factory for Project NEXUS
Creates neural networks from configuration files and presets
COMPLETELY FIXED VERSION - All syntax, type and import issues resolved
"""
import os
import yaml
import torch
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
import logging
import warnings
from pathlib import Path

# Type checking imports to avoid runtime import issues
if TYPE_CHECKING:
    from .ppo_networks import PPOActorCritic, PPOActorCriticWithComm
else:
    # Try to import our custom modules with fallback
    try:
        from .ppo_networks import PPOActorCritic, PPOActorCriticWithComm
    except ImportError:
        try:
            from ppo_networks import PPOActorCritic, PPOActorCriticWithComm
        except ImportError:
            warnings.warn("Could not import network modules. Factory functionality may be limited.")
            PPOActorCritic = None
            PPOActorCriticWithComm = None

# Set up logging
logger = logging.getLogger(__name__)

class NetworkFactory:
    """
    Factory class for creating neural networks from configurations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the network factory
        
        Args:
            config_path: Path to network configuration file
        """
        self.config_path = self._find_config_path(config_path)
        self.config = self._load_config()
        
    def _find_config_path(self, config_path: Optional[str]) -> Optional[str]:
        """Find the configuration file with multiple fallback locations"""
        if config_path is not None:
            if os.path.exists(config_path):
                return config_path
            else:
                logger.warning(f"Specified config path {config_path} not found")
        
        # Try multiple possible locations
        possible_paths = [
            "configs/network.yaml",
            "../configs/network.yaml", 
            "../../configs/network.yaml",
            os.path.join(os.path.dirname(__file__), "../../configs/network.yaml"),
            os.path.join(Path.cwd(), "configs", "network.yaml"),
            "network.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found config file at: {path}")
                return path
        
        logger.warning("No network configuration file found, using defaults")
        return None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load network configuration from YAML file"""
        if self.config_path is None:
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded network configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default network configuration"""
        return {
            'network': {
                'spatial_channels': 5,
                'spatial_dim': 256,
                'state_dim': 128,
                'fusion_dim': 512,
                'action_dim': 14,
                'policy_hidden_dim': 256,
                'value_hidden_dim': 256,
                'use_enhanced_cnn': False,
                'use_attention_fusion': False,
                'enable_communication': False,
                'dropout_rate': 0.1,
                'init_method': 'orthogonal',
                'policy_init_gain': 0.01,
                'value_init_gain': 1.0
            },
            'presets': {
                'lightweight': {
                    'spatial_dim': 128,
                    'state_dim': 64,
                    'fusion_dim': 256,
                    'policy_hidden_dim': 128,
                    'value_hidden_dim': 128,
                    'use_enhanced_cnn': False,
                    'use_attention_fusion': False,
                    'enable_communication': False
                },
                'standard': {
                    'spatial_dim': 256,
                    'state_dim': 128,
                    'fusion_dim': 512,
                    'policy_hidden_dim': 256,
                    'value_hidden_dim': 256,
                    'use_enhanced_cnn': False,
                    'use_attention_fusion': False,
                    'enable_communication': False
                },
                'advanced': {
                    'spatial_dim': 384,
                    'state_dim': 192,
                    'fusion_dim': 768,
                    'policy_hidden_dim': 384,
                    'value_hidden_dim': 384,
                    'use_enhanced_cnn': True,
                    'use_attention_fusion': True,
                    'enable_communication': True
                },
                'performance': {
                    'spatial_dim': 512,
                    'state_dim': 256,
                    'fusion_dim': 1024,
                    'policy_hidden_dim': 512,
                    'value_hidden_dim': 512,
                    'use_enhanced_cnn': True,
                    'use_attention_fusion': True,
                    'enable_communication': True
                }
            },
            'device': {
                'type': 'auto'
            }
        }
    
    def create_network(self, preset: Optional[str] = None, 
                      override_config: Optional[Dict[str, Any]] = None,
                      device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
        """
        Create a PPO network from configuration
        
        Args:
            preset: Configuration preset name ('lightweight', 'standard', 'advanced', 'performance')
            override_config: Dictionary to override specific config values
            device: Target device for the network
            
        Returns:
            Configured PPO network or None if creation fails
        """
        if PPOActorCritic is None:
            raise ImportError("PPOActorCritic not available. Check module imports.")
        
        network = None
        
        try:
            # Start with base config
            network_config = self.config.get('network', {}).copy()
            
            # Apply preset if specified
            if preset is not None:
                preset_config = self._get_preset_config(preset)
                network_config.update(preset_config)
            
            # Apply overrides
            if override_config is not None:
                network_config.update(override_config)
            
            # Determine device
            if device is None:
                device = self._get_optimal_device()
            
            # Create network based on configuration
            if network_config.get('enable_communication', False) and PPOActorCriticWithComm is not None:
                network = PPOActorCriticWithComm(
                    spatial_channels=network_config['spatial_channels'],
                    spatial_dim=network_config['spatial_dim'],
                    state_dim=network_config['state_dim'],
                    fusion_dim=network_config['fusion_dim'],
                    action_dim=network_config['action_dim'],
                    use_enhanced_cnn=network_config.get('use_enhanced_cnn', False),
                    use_attention_fusion=network_config.get('use_attention_fusion', False),
                    enable_communication=True
                )
            else:
                if network_config.get('enable_communication', False):
                    warnings.warn("Communication requested but PPOActorCriticWithComm not available")
                
                network = PPOActorCritic(
                    spatial_channels=network_config['spatial_channels'],
                    spatial_dim=network_config['spatial_dim'],
                    state_dim=network_config['state_dim'],
                    fusion_dim=network_config['fusion_dim'],
                    action_dim=network_config['action_dim'],
                    use_enhanced_cnn=network_config.get('use_enhanced_cnn', False),
                    use_attention_fusion=network_config.get('use_attention_fusion', False)
                )
            
            # Move to device
            if network is not None:
                network = network.to(device)
                
                # Apply custom initialization if specified
                self._apply_initialization(network, network_config)
                
                # Log network info
                param_count = sum(p.numel() for p in network.parameters())
                logger.info(f"Created network with {param_count:,} parameters on {device}")
            
            return network
            
        except Exception as e:
            logger.error(f"Failed to create network: {e}")
            return None
    
    def _get_preset_config(self, preset: str) -> Dict[str, Any]:
        """Get configuration for specified preset"""
        presets = self.config.get('presets', {})
        
        if preset not in presets:
            available_presets = list(presets.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {available_presets}")
        
        return presets[preset]
    
    def _get_optimal_device(self) -> torch.device:
        """Determine optimal device based on configuration and availability"""
        device_config = self.config.get('device', {})
        device_type = device_config.get('type', 'auto')
        
        if device_type == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("Selected CUDA device")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Selected MPS device")
            else:
                device = torch.device('cpu')
                logger.info("Selected CPU device")
        else:
            try:
                device = torch.device(device_type)
                logger.info(f"Selected device: {device}")
            except Exception as e:
                logger.warning(f"Could not create device '{device_type}': {e}. Falling back to CPU")
                device = torch.device('cpu')
        
        return device
    
    def _apply_initialization(self, network: torch.nn.Module, config: Dict[str, Any]):
        """Apply custom weight initialization"""
        init_method = config.get('init_method', 'orthogonal')
        policy_gain = float(config.get('policy_init_gain', 0.01))  # Ensure float
        value_gain = float(config.get('value_init_gain', 1.0))    # Ensure float
        
        def init_weights(module, gain=1.0):
            if isinstance(module, torch.nn.Linear):
                try:
                    if init_method == 'orthogonal':
                        torch.nn.init.orthogonal_(module.weight, gain=gain)  # type: ignore
                    elif init_method == 'xavier':
                        torch.nn.init.xavier_normal_(module.weight, gain=gain)
                    elif init_method == 'kaiming':
                        torch.nn.init.kaiming_normal_(module.weight)
                    
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0)
                except Exception as e:
                    warnings.warn(f"Weight initialization failed for module: {e}")
        
        try:
            # Apply different gains to policy and value heads
            if hasattr(network, 'policy_head'):
                network.policy_head.apply(lambda m: init_weights(m, policy_gain))
            if hasattr(network, 'value_head'):
                network.value_head.apply(lambda m: init_weights(m, value_gain))
            
            logger.debug(f"Applied {init_method} initialization")
        except Exception as e:
            warnings.warn(f"Custom initialization failed: {e}")
    
    def get_config_summary(self, preset: Optional[str] = None) -> str:
        """Get a summary of the current network configuration"""
        try:
            network_config = self.config.get('network', {}).copy()
            
            if preset is not None:
                preset_config = self._get_preset_config(preset)
                network_config.update(preset_config)
            
            summary_lines = [
                "=== NETWORK CONFIGURATION SUMMARY ===",
                f"Spatial Channels: {network_config.get('spatial_channels', 5)}",
                f"Spatial Dimension: {network_config.get('spatial_dim', 256)}",
                f"State Dimension: {network_config.get('state_dim', 128)}",
                f"Fusion Dimension: {network_config.get('fusion_dim', 512)}",
                f"Action Dimension: {network_config.get('action_dim', 14)}",
                f"Enhanced CNN: {network_config.get('use_enhanced_cnn', False)}",
                f"Attention Fusion: {network_config.get('use_attention_fusion', False)}",
                f"Communication: {network_config.get('enable_communication', False)}",
                f"Dropout Rate: {network_config.get('dropout_rate', 0.1)}",
                f"Initialization: {network_config.get('init_method', 'orthogonal')}",
            ]
            
            if preset:
                summary_lines.insert(1, f"Preset: {preset}")
            
            return "\n".join(summary_lines)
        except Exception as e:
            return f"Error generating config summary: {e}"
    
    def save_config(self, filepath: str):
        """Save current configuration to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Could not save config to {filepath}: {e}")
    
    def benchmark_network(self, preset: Optional[str] = None, 
                         batch_size: int = 32, num_runs: int = 100) -> Dict[str, Union[float, int, str]]:
        """
        Benchmark network performance
        
        Args:
            preset: Configuration preset to benchmark
            batch_size: Batch size for benchmarking
            num_runs: Number of forward passes to average
            
        Returns:
            Dictionary with timing results
        """
        try:
            network = self.create_network(preset=preset)
            if network is None:
                raise RuntimeError("Could not create network for benchmarking")
            
            network.eval()
            
            # Create test data
            observations = torch.randn(batch_size, 5, 15, 15, device=next(network.parameters()).device)
            agent_states = torch.randn(batch_size, 8, device=observations.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    try:
                        network(observations, agent_states)
                    except Exception as e:
                        warnings.warn(f"Warmup failed: {e}")
                        break
            
            # Benchmark forward pass
            if observations.device.type == 'cuda':
                torch.cuda.synchronize()
            
            import time
            start = time.perf_counter()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    try:
                        network(observations, agent_states)
                    except Exception as e:
                        warnings.warn(f"Benchmark run failed: {e}")
                        break
            
            if observations.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            
            total_time = (end - start) * 1000  # Convert to milliseconds
            avg_time_per_batch = total_time / num_runs
            avg_time_per_sample = avg_time_per_batch / batch_size
            
            param_count = sum(p.numel() for p in network.parameters())
            
            results = {
                'avg_time_per_batch_ms': avg_time_per_batch,
                'avg_time_per_sample_ms': avg_time_per_sample,
                'throughput_samples_per_sec': 1000.0 / avg_time_per_sample if avg_time_per_sample > 0 else 0.0,
                'parameter_count': param_count,
                'device': str(observations.device),
                'batch_size': batch_size,
                'num_runs': num_runs
            }
            
            logger.info(f"Benchmark results: {avg_time_per_batch:.2f}ms/batch, "
                       f"{results['throughput_samples_per_sec']:.1f} samples/sec")
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            return {'error': str(e)}

# Convenience functions for common use cases with error handling
def create_lightweight_network(device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
    """Create lightweight network for fast experimentation"""
    try:
        factory = NetworkFactory()
        return factory.create_network(preset='lightweight', device=device)
    except Exception as e:
        warnings.warn(f"Could not create lightweight network: {e}")
        return None

def create_standard_network(device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
    """Create standard network for main training"""
    try:
        factory = NetworkFactory()
        return factory.create_network(preset='standard', device=device)
    except Exception as e:
        warnings.warn(f"Could not create standard network: {e}")
        return None

def create_advanced_network(device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
    """Create advanced network with all features"""
    try:
        factory = NetworkFactory()
        return factory.create_network(preset='advanced', device=device)
    except Exception as e:
        warnings.warn(f"Could not create advanced network: {e}")
        return None

def create_performance_network(device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
    """Create high-performance network for final training"""
    try:
        factory = NetworkFactory()
        return factory.create_network(preset='performance', device=device)
    except Exception as e:
        warnings.warn(f"Could not create performance network: {e}")
        return None

def create_custom_network(config_overrides: Dict[str, Any], 
                         device: Optional[torch.device] = None) -> Optional[torch.nn.Module]:
    """Create network with custom configuration overrides"""
    try:
        factory = NetworkFactory()
        return factory.create_network(override_config=config_overrides, device=device)
    except Exception as e:
        warnings.warn(f"Could not create custom network: {e}")
        return None

if __name__ == "__main__":
    # Test network factory with comprehensive error handling
    print("Testing Network Factory with Error Handling...")
    
    try:
        factory = NetworkFactory()
        
        # Test different presets
        presets = ['lightweight', 'standard']
        
        for preset in presets:
            print(f"\n=== Testing {preset.upper()} preset ===")
            try:
                print(factory.get_config_summary(preset=preset))
                
                # Create and test network
                network = factory.create_network(preset=preset)
                if network is not None:
                    param_count = sum(p.numel() for p in network.parameters())
                    print(f"Parameter count: {param_count:,}")
                    
                    # Quick forward pass test
                    test_obs = torch.randn(2, 5, 15, 15, device=next(network.parameters()).device)
                    test_states = torch.randn(2, 8, device=test_obs.device)
                    
                    with torch.no_grad():
                        output = network(test_obs, test_states)
                        print(f"Forward pass successful: output shapes {[o.shape for o in output]}")
                else:
                    print(f"Failed to create {preset} network")
                    
            except Exception as e:
                print(f"Error testing {preset} preset: {e}")
        
        print("\nNetwork factory testing completed!")
        
    except Exception as e:
        print(f"Factory initialization failed: {e}")
        import traceback
        traceback.print_exc()