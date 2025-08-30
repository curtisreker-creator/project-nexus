# Contributing to Project NEXUS

Thank you for your interest in contributing to Project NEXUS! This document provides guidelines and information for contributing to our multi-agent reinforcement learning research platform.

## 🚀 Getting Started

### Prerequisites
- **Development Environment**: macOS with Apple Silicon (recommended) or Intel Mac
- **Python**: 3.9+ with conda environment management
- **Git**: Familiarity with Git workflows and GitHub
- **Background**: Basic understanding of reinforcement learning and PyTorch

### Development Setup
```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/project-nexus.git
cd project-nexus

# 3. Create development environment
conda env create -f environment-macos.yml
conda activate nexus

# 4. Install development dependencies
pip install pytest-cov black flake8 mypy pre-commit

# 5. Set up pre-commit hooks
pre-commit install

# 6. Verify setup
python test_integration.py
```

## 🎯 How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **🐛 Bug Reports**: Issues with existing functionality
2. **✨ Feature Requests**: New capabilities and enhancements
3. **📝 Documentation**: Improvements to guides, comments, and examples
4. **🧪 Testing**: Additional test cases and test coverage improvements
5. **🔧 Code Contributions**: Bug fixes, features, and optimizations
6. **📊 Research**: Experimental results, benchmarks, and analysis
7. **🎨 Examples**: Demo scripts, tutorials, and use case implementations

### Priority Areas for Contributions

**High Priority:**
- **Training Pipeline**: PPO implementation and distributed training
- **Multi-Agent Communication**: Token-based coordination protocols
- **Performance Optimization**: Apple Silicon MPS optimizations
- **Documentation**: API docs and advanced tutorials

**Medium Priority:**
- **Environment Extensions**: New scenarios and mechanics
- **Neural Architecture**: Advanced attention mechanisms
- **Evaluation Metrics**: Comprehensive benchmarking tools
- **Visualization**: Training progress and agent behavior visualization

**Research Areas:**
- **Curriculum Learning**: Automated difficulty progression
- **Transfer Learning**: Cross-environment knowledge transfer
- **Emergent Behavior**: Complex coordination patterns analysis
- **Meta-Learning**: Few-shot adaptation capabilities

## 📋 Contribution Process

### 1. Planning Your Contribution

**Before starting work:**

1. **Check Existing Issues**: Review [open issues](https://github.com/curtisreker-creator/project-nexus/issues) to avoid duplication
2. **Discuss Major Changes**: For significant features, open a discussion or issue first
3. **Review Architecture**: Read `docs/ARCHITECTURE.md` to understand system design
4. **Check Project Phases**: Align contribution with current development phase

**For New Contributors:**
- Look for issues labeled `good-first-issue` or `help-wanted`
- Start with documentation improvements or test additions
- Join discussions to understand project direction

### 2. Development Workflow

#### Branch Strategy
```bash
# Create feature branch from main
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b bugfix/issue-description

# Or for documentation
git checkout -b docs/documentation-improvement
```

#### Development Process
```bash
# 1. Make your changes
# 2. Run tests frequently
python test_integration.py
python -m pytest tests/ -v

# 3. Check code quality
black . --check
flake8 .
mypy agents/

# 4. Run comprehensive tests
python -m pytest tests/ -v --cov=agents --cov=environment

# 5. Commit changes (pre-commit hooks will run)
git add .
git commit -m "feat: add spatial attention mechanism to CNN

- Implement SpatialAttention class with channel attention
- Add attention toggle to EnhancedSpatialCNN
- Update tests for attention functionality
- Benchmark shows 5% performance improvement"
```

### 3. Code Standards

#### Python Code Style
```python
# Use type hints for all function signatures
def process_observations(observations: torch.Tensor, 
                        agent_states: torch.Tensor,
                        device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process multi-agent observations for network input.
    
    Args:
        observations: Grid observations of shape (batch, 5, 15, 15)
        agent_states: Agent state tensor of shape (batch, 8)
        device: Target device for tensor operations
        
    Returns:
        Tuple of (processed_observations, normalized_states)
        
    Raises:
        ValueError: If input dimensions don't match expectations
    """
    # Input validation
    if observations.dim() != 4:
        raise ValueError(f"Expected 4D observations, got {observations.dim()}D")
    
    # Implementation with clear comments
    batch_size = observations.size(0)
    
    # Process observations
    processed_obs = observations.float()
    if device is not None:
        processed_obs = processed_obs.to(device)
    
    return processed_obs, agent_states
```

#### Documentation Standards
```python
class SpatialCNN(nn.Module):
    """
    Convolutional Neural Network for spatial feature extraction from grid observations.
    
    This network processes 5-channel grid observations representing:
    - Channel 0: Empty spaces and topology
    - Channel 1: Resource locations and types  
    - Channel 2: Agent positions
    - Channel 3: Building locations
    - Channel 4: Recent activity traces
    
    Args:
        input_channels: Number of input channels (default: 5)
        feature_dim: Output feature dimension (default: 256)
        
    Example:
        >>> cnn = SpatialCNN(input_channels=5, feature_dim=128)
        >>> observations = torch.randn(4, 5, 15, 15)
        >>> features = cnn(observations)
        >>> print(features.shape)  # torch.Size([4, 128])
    """
```

#### Testing Standards
```python
class TestSpatialCNN:
    """Comprehensive test suite for SpatialCNN module"""
    
    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape"""
        cnn = SpatialCNN(feature_dim=256)
        batch_size = 4
        test_input = torch.randn(batch_size, 5, 15, 15)
        
        output = cnn(test_input)
        
        assert output.shape == (batch_size, 256)
        assert output.dtype == torch.float32
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through network"""
        cnn = SpatialCNN()
        test_input = torch.randn(2, 5, 15, 15, requires_grad=True)
        
        output = cnn(test_input)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist and are non-zero
        assert test_input.grad is not None
        assert torch.sum(torch.abs(test_input.grad)) > 0
    
    def test_device_compatibility(self):
        """Test network works on different devices"""
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            cnn = SpatialCNN().to(device)
            test_input = torch.randn(2, 5, 15, 15, device=device)
            
            output = cnn(test_input)
            assert output.device == device
```

### 4. Commit Message Guidelines

We use conventional commits for clear project history:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

#### Commit Types:
- `feat`: New feature implementation
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or modifications
- `refactor`: Code refactoring without functionality changes
- `perf`: Performance improvements
- `style`: Code style/formatting changes
- `build`: Build system or dependency changes
- `ci`: Continuous integration changes
- `chore`: Maintenance tasks

#### Examples:
```bash
# Feature addition
git commit -m "feat(networks): add spatial attention mechanism

Implement SpatialAttention class for enhanced CNN with:
- Channel-wise attention computation
- Spatial feature weighting
- 5% performance improvement in benchmarks

Closes #42"

# Bug fix
git commit -m "fix(environment): correct resource respawn logic

Resources were not respawning correctly after depletion.
Fixed off-by-one error in respawn timer calculation.

Fixes #38"

# Documentation
git commit -m "docs(api): add comprehensive network factory documentation

- Document all factory methods with examples
- Add configuration parameter descriptions  
- Include performance benchmark results"

# Testing
git commit -m "test(integration): add multi-agent coordination tests

Add test cases for:
- 2-agent resource competition
- 4-agent building coordination
- Communication token passing

Increases test coverage to 94%"
```

## 🧪 Testing Requirements

### Test Categories

#### 1. Unit Tests
```bash
# Test individual components
python -m pytest tests/test_networks.py::TestSpatialCNN -v
python -m pytest tests/test_environment.py::TestGridWorld -v
```

#### 2. Integration Tests
```bash
# Test component interactions
python test_integration.py

# Specific integration scenarios
python -m pytest tests/test_training.py -v
```

#### 3. Performance Tests
```bash
# Benchmark network performance
python -c "
from agents.networks.network_factory import NetworkFactory
factory = NetworkFactory()
results = factory.benchmark_network(preset='standard')
print(results)
"
```

#### 4. Device Compatibility Tests
```bash
# Test MPS compatibility
python -c "
import torch
from agents.networks import create_standard_network

if torch.backends.mps.is_available():
    device = torch.device('mps')
    network = create_standard_network(device=device)
    print('✅ MPS compatibility verified')
"
```

### Adding New Tests

When adding functionality, include:
1. **Unit tests** for individual functions/classes
2. **Integration tests** for cross-component interactions
3. **Error handling tests** for edge cases
4. **Performance benchmarks** for optimization validation

```python
# Example: Adding tests for new feature
class TestNewFeature:
    """Test suite for newly added feature"""
    
    def test_basic_functionality(self):
        """Test basic operation works correctly"""
        pass
    
    def test_edge_cases(self):
        """Test behavior with edge cases"""
        pass
    
    def test_error_handling(self):
        """Test proper error handling"""
        pass
    
    def test_performance(self):
        """Benchmark performance characteristics"""
        pass
    
    def test_integration(self):
        """Test integration with existing components"""
        pass
```

## 📊 Performance Guidelines

### Optimization Priorities

1. **Memory Efficiency**: Minimize GPU/MPS memory usage
2. **Training Speed**: Optimize forward/backward pass performance
3. **Scalability**: Support multiple agents and environments
4. **Apple Silicon**: Leverage MPS backend capabilities

### Benchmarking New Code

```python
# Template for performance testing
import time
import torch
from memory_profiler import profile

@profile
def benchmark_new_feature():
    """Benchmark new feature implementation"""
    # Setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Create test data
    test_input = torch.randn(32, 5, 15, 15, device=device)
    
    # Warmup
    for _ in range(10):
        _ = your_new_function(test_input)
    
    # Benchmark
    torch.mps.synchronize() if device.type == 'mps' else None
    start_time = time.perf_counter()
    
    for _ in range(100):
        result = your_new_function(test_input)
    
    torch.mps.synchronize() if device.type == 'mps' else None
    end_time = time.perf_counter()
    
    # Report results
    avg_time = (end_time - start_time) / 100
    throughput = 32 / avg_time  # samples per second
    
    print(f"Average time: {avg_time*1000:.2f}ms")
    print(f"Throughput: {throughput:.1f} samples/sec")
    
    return avg_time, throughput
```

## 📝 Documentation Requirements

### Code Documentation
- **Docstrings**: All public functions, classes, and methods
- **Type hints**: Function signatures and complex variables
- **Comments**: Complex algorithms and business logic
- **Examples**: Usage examples in docstrings

### External Documentation
- **API Documentation**: For new modules and significant changes
- **Tutorials**: For new features requiring user guidance  
- **Architecture Changes**: Updates to technical documentation
- **Research Results**: Experimental findings and benchmarks

### Documentation Checklist
- [ ] Docstrings follow NumPy style
- [ ] Type hints are comprehensive
- [ ] Examples are included and tested
- [ ] README updates reflect changes
- [ ] API documentation is updated
- [ ] Architecture diagrams are current

## 🔄 Pull Request Process

### Before Submitting

1. **Rebase on Latest Main**:
   ```bash
   git checkout main
   git pull origin main
   git checkout your-branch
   git rebase main
   ```

2. **Run Full Test Suite**:
   ```bash
   python test_integration.py
   python -m pytest tests/ -v --cov=agents --cov=environment
   ```

3. **Code Quality Check**:
   ```bash
   black . --check
   flake8 .
   mypy agents/
   ```

### PR Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] New feature (non-breaking change adding functionality)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Changes Made
- List of specific changes
- Impact on existing functionality
- New dependencies or requirements

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Added tests for new functionality
- [ ] Benchmarked performance impact

## Screenshots/Benchmarks
Include relevant performance data or visual changes.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. **Automated Checks**: CI runs tests and style checks
2. **Code Review**: Maintainers review design and implementation
3. **Testing Validation**: Reviewers test functionality locally
4. **Documentation Review**: Ensure docs are complete and accurate
5. **Performance Validation**: Benchmark significant changes
6. **Final Approval**: Maintainer approval and merge

## 🎓 Research Contributions

### Experimental Work

For research contributions:

1. **Hypothesis Documentation**: Clear research questions and hypotheses
2. **Experimental Setup**: Detailed methodology and parameters
3. **Results Analysis**: Statistical significance and interpretation
4. **Reproducibility**: Code and data for result reproduction

### Research Paper Integration

Contributing research findings:

```python
# Example: Research experiment structure
class ExperimentRunner:
    """Framework for reproducible research experiments"""
    
    def __init__(self, config_path: str, seed: int = 42):
        self.config = load_config(config_path)
        self.seed = seed
        self.results = {}
    
    def run_experiment(self, name: str, iterations: int = 5):
        """Run experiment with multiple seeds for statistical validity"""
        results = []
        
        for i in range(iterations):
            # Set seed for reproducibility
            torch.manual_seed(self.seed + i)
            np.random.seed(self.seed + i)
            
            # Run single experiment instance
            result = self._single_run()
            results.append(result)
        
        # Statistical analysis
        mean_result = np.mean(results)
        std_result = np.std(results)
        
        self.results[name] = {
            'mean': mean_result,
            'std': std_result,
            'raw_results': results,
            'config': self.config
        }
        
        return self.results[name]
```

## 🏆 Recognition

### Contributor Recognition

We recognize contributions through:

- **GitHub Contributors**: All contributors listed in repository
- **Release Notes**: Significant contributions highlighted
- **Research Papers**: Academic collaborations acknowledged
- **Community Highlights**: Outstanding contributions featured

### Becoming a Maintainer

Path to maintainer status:
1. **Consistent Contributions**: Regular, high-quality contributions
2. **Community Engagement**: Helping other contributors and users  
3. **Technical Expertise**: Deep understanding of project architecture
4. **Review Participation**: Providing thoughtful code reviews
5. **Leadership**: Mentoring new contributors

## 🤝 Community Guidelines

### Communication

- **Respectful Discourse**: Professional and inclusive communication
- **Constructive Feedback**: Focus on improving code and ideas
- **Collaborative Spirit**: Work together toward common goals
- **Knowledge Sharing**: Help others learn and contribute

### Getting Help

- **GitHub Issues**: Technical problems and bug reports
- **GitHub Discussions**: General questions and ideas
- **Code Reviews**: Learning opportunities and feedback
- **Documentation**: Comprehensive guides and examples

### Reporting Issues

When reporting bugs or issues:

1. **Search Existing Issues**: Avoid duplicates
2. **Provide Reproduction Steps**: Clear instructions to reproduce
3. **Include Environment Info**: OS, Python version, dependencies
4. **Attach Logs**: Error messages and stack traces
5. **Suggest Solutions**: If you have ideas for fixes

---

Thank you for contributing to Project NEXUS! Your contributions help advance multi-agent reinforcement learning research and build a stronger AI community.

**Questions?** Open a [GitHub Discussion](https://github.com/curtisreker-creator/project-nexus/discussions) or contact the maintainers.