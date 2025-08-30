# Installation Guide - Project NEXUS

This guide provides detailed installation instructions for Project NEXUS on macOS with Apple Silicon support.

## System Requirements

### Hardware Requirements
- **Recommended**: Apple Silicon Mac (M1/M2/M3) with 16GB+ RAM
- **Minimum**: Intel Mac with 8GB RAM (limited MPS support)
- **Storage**: 5GB free space for environment and dependencies

### Software Requirements
- **macOS**: 12.0+ (Monterey or later)
- **Python**: 3.9 - 3.11 (3.9 recommended)
- **Conda**: Miniconda or Anaconda
- **Git**: For repository cloning
- **Xcode Command Line Tools**: `xcode-select --install`

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository
```bash
# Clone the repository
git clone https://github.com/curtisreker-creator/project-nexus.git
cd project-nexus

# Verify repository contents
ls -la
```

#### Step 2: Create Conda Environment
```bash
# Create environment from file (Apple Silicon)
conda env create -f environment-macos.yml

# For Intel Macs, use the standard environment
conda env create -f environment.yml

# Activate the environment
conda activate nexus
```

#### Step 3: Verify Installation
```bash
# Run integration tests
python test_integration.py

# Run basic demo
python scripts/demo.py

# Check environment functionality
python test_basic.py
```

Expected output:
```
🚀 PROJECT NEXUS - COMPREHENSIVE INTEGRATION TEST
===============================
✅ INTEGRATION SUCCESSFUL!
✅ All tests passed - system fully operational
🚀 System ready for Phase 3!
```

### Method 2: GPU-Enabled Installation (CUDA)

For systems with NVIDIA GPUs, use the GPU environment:

```bash
# Clone repository
git clone https://github.com/curtisreker-creator/project-nexus.git
cd project-nexus

# Create GPU environment
conda env create -f environment-gpu.yml
conda activate nexus

# Verify CUDA installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Method 3: Development Installation

For contributors and researchers who need the full development setup:

```bash
# Clone with development tools
git clone https://github.com/curtisreker-creator/project-nexus.git
cd project-nexus

# Create development environment
conda env create -f environment-macos.yml
conda activate nexus

# Install additional development tools
pip install pytest-cov black flake8 mypy pre-commit

# Set up pre-commit hooks
pre-commit install

# Run comprehensive tests
python -m pytest tests/ -v --cov=agents --cov=environment
```

## Environment Verification

### Quick Verification
```bash
conda activate nexus

# Test basic imports
python -c "
import torch
import numpy as np
from environment.grid_world import GridWorld
print('✅ Core imports successful')
"

# Test MPS (Apple Silicon)
python -c "
import torch
if torch.backends.mps.is_available():
    print('✅ MPS backend available')
    device = torch.device('mps')
    x = torch.randn(3, 3, device=device)
    print(f'✅ MPS tensor creation successful: {x.device}')
else:
    print('⚠️ MPS not available, using CPU')
"
```

### Comprehensive Verification
```bash
# Run all verification tests
python test_integration.py

# Test neural networks
python -c "
from agents.networks import create_standard_network
network = create_standard_network()
if network:
    param_count = sum(p.numel() for p in network.parameters())
    print(f'✅ Network created: {param_count:,} parameters')
"

# Test environment
python -c "
from environment import create_environment
env = create_environment()
obs, info = env.reset(seed=42)
print(f'✅ Environment created: {obs.shape} observation')
"
```

## Common Installation Issues

### Issue 1: Conda Environment Creation Fails
```bash
# Solution: Update conda first
conda update conda
conda clean --all

# Try creating environment again
conda env create -f environment-macos.yml
```

### Issue 2: PyTorch MPS Issues
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Should be 1.12.0 or later for MPS support
# If not, reinstall:
conda remove pytorch torchvision torchaudio
conda install pytorch torchvision torchaudio -c pytorch
```

### Issue 3: Import Errors for Neural Networks
```bash
# Verify Python path
python -c "
import sys
print('Python path:')
for path in sys.path:
    print(f'  {path}')
"

# Add current directory to path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue 4: Permission Errors
```bash
# Fix conda permissions
sudo chown -R $(whoami) /opt/anaconda3/  # Adjust path as needed

# Or use conda environment in user directory
conda config --add envs_dirs ~/.conda/envs
```

## Alternative Installation Options

### Docker Installation (Coming Soon)
```bash
# Pull Docker image (when available)
docker pull curtisreker/project-nexus:latest

# Run container
docker run -it --rm curtisreker/project-nexus:latest
```

### Cloud Installation (Colab/Kaggle)
```python
# For Google Colab
!git clone https://github.com/curtisreker-creator/project-nexus.git
%cd project-nexus
!pip install -r requirements.txt  # When available

# Import and test
from environment.grid_world import GridWorld
env = GridWorld()
print("✅ Installation successful in cloud environment")
```

## Environment Management

### Managing Multiple Environments
```bash
# List all conda environments
conda env list

# Create environment with specific name
conda env create -f environment-macos.yml -n nexus-dev

# Remove environment
conda env remove -n nexus-dev

# Export current environment
conda env export > my-nexus-env.yml
```

### Updating Environment
```bash
# Update existing environment
conda env update -f environment-macos.yml

# Update specific packages
conda activate nexus
conda update pytorch torchvision torchaudio
```

## Performance Optimization

### Apple Silicon Optimization
```bash
# Verify optimal settings
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CPU count: {torch.get_num_threads()}')

# Test performance
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
x = torch.randn(1000, 1000, device=device)
result = torch.mm(x, x.t())
print(f'✅ Performance test completed on {device}')
"
```

### Memory Optimization
```bash
# Set memory growth (if needed)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Monitor memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## Troubleshooting

### Getting Help
1. **Check Issues**: Visit [GitHub Issues](https://github.com/curtisreker-creator/project-nexus/issues)
2. **Run Diagnostics**: Use `python test_integration.py` for comprehensive testing
3. **Environment Info**: Use `conda info` and `conda list` for debugging
4. **System Info**: Use `python -m torch.utils.collect_env` for PyTorch diagnostics

### Clean Reinstallation
```bash
# Remove existing environment
conda env remove -n nexus

# Clean conda cache
conda clean --all

# Remove repository and re-clone
cd ..
rm -rf project-nexus
git clone https://github.com/curtisreker-creator/project-nexus.git
cd project-nexus

# Fresh installation
conda env create -f environment-macos.yml
conda activate nexus
python test_integration.py
```

## Next Steps

After successful installation:

1. **Read the Quick Start Guide**: `docs/QUICKSTART.md`
2. **Explore Examples**: Run `python scripts/demo.py`
3. **Review Architecture**: Read `docs/ARCHITECTURE.md`
4. **Start Development**: Follow `CONTRIBUTING.md`

---

**Need Help?** Open an issue on GitHub or check the [documentation wiki](https://github.com/curtisreker-creator/project-nexus/wiki).