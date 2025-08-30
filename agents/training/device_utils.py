# File: agents/training/device_utils.py
"""
Device Compatibility Utilities for Project NEXUS
Handles Apple Silicon MPS device compatibility issues
"""
import torch
import warnings
import logging
from typing import Optional, Union, Tuple

logger = logging.getLogger(__name__)

def get_optimal_device(prefer_device: Optional[str] = None) -> torch.device:
    """
    Get optimal device with MPS compatibility checks
    
    Args:
        prefer_device: Preferred device string ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        Optimal device for training
    """
    if prefer_device == 'cpu':
        return torch.device('cpu')
    
    if prefer_device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    
    if prefer_device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Check for known MPS issues
            if _check_mps_compatibility():
                return torch.device('mps')
            else:
                logger.warning("MPS device has compatibility issues, falling back to CPU")
                return torch.device('cpu')
        else:
            logger.warning("MPS not available, falling back to CPU")
            return torch.device('cpu')
    
    # Auto device selection
    if torch.cuda.is_available():
        logger.info("Using CUDA device")
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if _check_mps_compatibility():
            logger.info("Using MPS device")
            return torch.device('mps')
        else:
            logger.warning("MPS device has known issues, using CPU")
            return torch.device('cpu')
    else:
        logger.info("Using CPU device")
        return torch.device('cpu')

def _check_mps_compatibility() -> bool:
    """
    Check for known MPS compatibility issues
    
    Returns:
        True if MPS should be safe to use
    """
    try:
        # Test basic MPS operations
        device = torch.device('mps')
        
        # Test tensor creation
        test_tensor = torch.randn(2, 2, device=device)
        
        # Test basic operations
        result = test_tensor + 1.0
        result = torch.matmul(test_tensor, test_tensor.T)
        
        # Test convolution (common failure point)
        conv = torch.nn.Conv2d(1, 1, 3, padding=1).to(device)
        test_input = torch.randn(1, 1, 5, 5, device=device)
        conv_result = conv(test_input)
        
        return True
        
    except Exception as e:
        logger.warning(f"MPS compatibility test failed: {e}")
        return False

def ensure_tensor_device(tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    """
    Ensure tensor is on target device with safety checks
    
    Args:
        tensor: Input tensor
        target_device: Target device
        
    Returns:
        Tensor on target device
    """
    if tensor.device == target_device:
        return tensor
    
    try:
        return tensor.to(target_device)
    except Exception as e:
        logger.warning(f"Device transfer failed: {e}. Keeping tensor on {tensor.device}")
        return tensor

def safe_to_device(module_or_tensor: Union[torch.nn.Module, torch.Tensor], 
                  device: torch.device) -> Union[torch.nn.Module, torch.Tensor]:
    """
    Safely move module or tensor to device with error handling
    
    Args:
        module_or_tensor: PyTorch module or tensor
        device: Target device
        
    Returns:
        Module/tensor on target device (or original device if transfer fails)
    """
    try:
        return module_or_tensor.to(device)
    except Exception as e:
        current_device = next(module_or_tensor.parameters()).device if hasattr(module_or_tensor, 'parameters') else module_or_tensor.device
        logger.warning(f"Device transfer to {device} failed: {e}. Keeping on {current_device}")
        return module_or_tensor

def check_tensor_device_consistency(*tensors: torch.Tensor) -> Tuple[bool, Optional[str]]:
    """
    Check if all tensors are on the same device
    
    Args:
        *tensors: Variable number of tensors to check
        
    Returns:
        Tuple of (is_consistent, error_message)
    """
    if not tensors:
        return True, None
    
    reference_device = tensors[0].device
    
    for i, tensor in enumerate(tensors[1:], 1):
        if tensor.device != reference_device:
            return False, f"Tensor {i} on {tensor.device}, expected {reference_device}"
    
    return True, None

def create_device_compatible_tensor(data, device: torch.device, **kwargs) -> torch.Tensor:
    """
    Create tensor directly on target device
    
    Args:
        data: Data to create tensor from
        device: Target device
        **kwargs: Additional tensor creation arguments
        
    Returns:
        Tensor on target device
    """
    try:
        return torch.tensor(data, device=device, **kwargs)
    except Exception as e:
        logger.warning(f"Direct device tensor creation failed: {e}. Creating on CPU first")
        cpu_tensor = torch.tensor(data, **kwargs)
        return safe_to_device(cpu_tensor, device)

# Environment variable for MPS fallback
import os
def setup_mps_fallback():
    """Setup MPS CPU fallback environment variable if needed"""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Enable CPU fallback for unsupported MPS operations
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        logger.info("MPS CPU fallback enabled")

# Auto-setup on import
setup_mps_fallback()