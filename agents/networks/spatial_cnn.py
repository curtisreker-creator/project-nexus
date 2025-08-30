# File: agents/networks/spatial_cnn.py
"""
Spatial CNN for processing 15x15 grid observations in Project NEXUS
Extracts spatial features from 5-channel observation tensor
FIXED VERSION - Addresses device compatibility and error handling issues
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import warnings

class SpatialCNN(nn.Module):
    """
    Convolutional Neural Network for spatial feature extraction
    
    Input: (batch_size, 5, 15, 15) - 5 channels: empty, resources, agents, buildings, activity
    Output: (batch_size, feature_dim) - flattened spatial features
    """
    
    def __init__(self, input_channels: int = 5, feature_dim: int = 256):
        super(SpatialCNN, self).__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # Spatial feature extraction layers
        self.conv_layers = nn.Sequential(
            # First conv block: 5 -> 32 channels
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # Second conv block: 32 -> 64 channels with pooling
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),  # 15x15 -> 7x7 (with padding effects)
            
            # Third conv block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            
            # Fourth conv block: 128 -> 128 channels with pooling
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, stride=2),  # 7x7 -> 3x3 (with padding effects)
        )
        
        # Calculate actual flattened size by doing a forward pass
        self.flattened_size = self._calculate_flattened_size()
        
        # Fully connected layers for feature compression
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_flattened_size(self) -> int:
        """Calculate the actual flattened size after convolutions"""
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, self.input_channels, 15, 15)
                dummy_output = self.conv_layers(dummy_input)
                return int(dummy_output.numel())
        except Exception as e:
            # Fallback calculation if forward pass fails
            warnings.warn(f"Could not calculate flattened size dynamically: {e}. Using fallback.")
            # Conservative estimate: after 2 max pools, roughly 4x4x128
            return 4 * 4 * 128
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial CNN
        
        Args:
            x: Input tensor of shape (batch_size, 5, 15, 15)
            
        Returns:
            Spatial features of shape (batch_size, feature_dim)
        """
        # Validate input shape and type
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
            
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got {x.dim()}D tensor")
            
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[1]}")
            
        if x.shape[2] != 15 or x.shape[3] != 15:
            raise ValueError(f"Expected 15x15 spatial dimensions, got {x.shape[2]}x{x.shape[3]}")
        
        # Check for NaN or infinite values
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values")
        if torch.isinf(x).any():
            raise ValueError("Input contains infinite values")
        
        try:
            # Extract spatial features through convolutions
            spatial_features = self.conv_layers(x)
            
            # Flatten for fully connected layers
            batch_size = x.size(0)
            flattened = spatial_features.view(batch_size, -1)
            
            # Verify flattened size matches expectation
            if flattened.size(1) != self.flattened_size:
                # Update flattened size if it's different (handles dynamic sizing)
                self.flattened_size = flattened.size(1)
                # Recreate FC layers with correct input size
                self.fc_layers = nn.Sequential(
                    nn.Linear(self.flattened_size, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(512, self.feature_dim),
                    nn.ReLU(inplace=True)
                ).to(x.device)
                # Reinitialize weights
                for module in self.fc_layers.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.kaiming_normal_(module.weight)
                        nn.init.constant_(module.bias, 0)
            
            # Final feature compression
            features = self.fc_layers(flattened)
            
            return features
            
        except RuntimeError as e:
            raise RuntimeError(f"Forward pass failed: {e}")
    
    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Get intermediate feature maps for visualization/analysis
        
        Args:
            x: Input tensor of shape (batch_size, 5, 15, 15)
            
        Returns:
            Tuple of feature maps from each conv layer
        """
        feature_maps = []
        current = x
        
        try:
            for i, layer in enumerate(self.conv_layers):
                current = layer(current)
                if isinstance(layer, nn.Conv2d):
                    feature_maps.append(current.detach().clone())
        except Exception as e:
            warnings.warn(f"Could not extract feature maps: {e}")
            return tuple()
        
        return tuple(feature_maps)

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for focusing on important grid regions
    """
    
    def __init__(self, feature_channels: int = 128):
        super(SpatialAttention, self).__init__()
        
        self.attention_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels // 4, 1, 1, bias=True),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize attention weights"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention to feature maps"""
        try:
            attention_weights = self.attention_conv(x)
            return x * attention_weights
        except Exception as e:
            warnings.warn(f"Attention mechanism failed, returning original features: {e}")
            return x

class EnhancedSpatialCNN(SpatialCNN):
    """
    Enhanced Spatial CNN with attention mechanisms
    For advanced implementations with error handling
    """
    
    def __init__(self, input_channels: int = 5, feature_dim: int = 256, use_attention: bool = True):
        # Initialize parent class first
        super().__init__(input_channels, feature_dim)
        
        self.use_attention = use_attention
        if use_attention:
            try:
                self.attention = SpatialAttention(128)
            except Exception as e:
                warnings.warn(f"Could not initialize attention mechanism: {e}")
                self.use_attention = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with optional attention"""
        # Validate input
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        if x.dim() != 4 or x.shape[1] != self.input_channels:
            raise ValueError(f"Expected input shape (batch, {self.input_channels}, 15, 15), got {x.shape}")
        
        try:
            # Process through conv layers up to the last one
            current = x
            for layer in self.conv_layers[:-1]:  # All except last layer
                current = layer(current)
            
            # Apply attention if enabled and available
            if self.use_attention and hasattr(self, 'attention'):
                try:
                    current = self.attention(current)
                except Exception as e:
                    warnings.warn(f"Attention failed, continuing without: {e}")
            
            # Final conv layer
            current = self.conv_layers[-1](current)
            
            # Flatten and process through FC layers
            batch_size = x.size(0)
            flattened = current.view(batch_size, -1)
            
            # Handle dynamic sizing
            if flattened.size(1) != self.flattened_size:
                self.flattened_size = flattened.size(1)
                self.fc_layers = nn.Sequential(
                    nn.Linear(self.flattened_size, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(512, self.feature_dim),
                    nn.ReLU(inplace=True)
                ).to(x.device)
            
            features = self.fc_layers(flattened)
            return features
            
        except Exception as e:
            # Fallback to parent class implementation
            warnings.warn(f"Enhanced forward pass failed, using standard CNN: {e}")
            return super().forward(x)

def create_spatial_cnn(config: dict) -> SpatialCNN:
    """
    Factory function to create spatial CNN from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured spatial CNN
    """
    try:
        if config.get('use_enhanced_cnn', False):
            return EnhancedSpatialCNN(
                input_channels=config.get('input_channels', 5),
                feature_dim=config.get('spatial_dim', 256),
                use_attention=config.get('use_attention', True)
            )
        else:
            return SpatialCNN(
                input_channels=config.get('input_channels', 5),
                feature_dim=config.get('spatial_dim', 256)
            )
    except Exception as e:
        warnings.warn(f"Could not create CNN with config, using defaults: {e}")
        return SpatialCNN()

if __name__ == "__main__":
    # Test the spatial CNN with error handling
    print("Testing Spatial CNN with Error Handling...")
    
    try:
        # Create test data
        batch_size = 4
        test_input = torch.randn(batch_size, 5, 15, 15)
        
        # Test standard CNN
        print("Testing standard CNN...")
        cnn = SpatialCNN(feature_dim=256)
        output = cnn(test_input)
        
        print(f"✅ Input shape: {test_input.shape}")
        print(f"✅ Output shape: {output.shape}")
        print(f"✅ Parameters: {sum(p.numel() for p in cnn.parameters()):,}")
        
        # Test enhanced CNN
        print("\nTesting enhanced CNN...")
        enhanced_cnn = EnhancedSpatialCNN(feature_dim=256, use_attention=True)
        enhanced_output = enhanced_cnn(test_input)
        
        print(f"✅ Enhanced output shape: {enhanced_output.shape}")
        print(f"✅ Enhanced parameters: {sum(p.numel() for p in enhanced_cnn.parameters()):,}")
        
        # Test error handling with invalid input
        print("\nTesting error handling...")
        try:
            invalid_input = torch.randn(2, 3, 10, 10)  # Wrong shape
            cnn(invalid_input)
        except ValueError as e:
            print(f"✅ Correctly caught invalid input: {e}")
        
        # Test device compatibility
        if torch.backends.mps.is_available():
            print("\nTesting MPS compatibility...")
            device = torch.device('mps')
            mps_cnn = SpatialCNN().to(device)
            mps_input = test_input.to(device)
            mps_output = mps_cnn(mps_input)
            print(f"✅ MPS output shape: {mps_output.shape}")
        
        print("\n✅ All Spatial CNN tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()