# File: agents/training/gae_computer.py
"""
Generalized Advantage Estimation (GAE) for PPO
Efficient computation of advantages and returns for policy gradient methods
"""
import numpy as np
import torch
from typing import Union, Tuple, List, Optional
import logging

def compute_gae(rewards: Union[np.ndarray, List[float]], 
                values: Union[np.ndarray, List[float]],
                dones: Union[np.ndarray, List[bool]],
                gamma: float = 0.99,
                gae_lambda: float = 0.95,
                last_value: float = 0.0,
                normalize_advantages: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE)
    
    GAE helps reduce variance in policy gradient estimates while maintaining
    low bias through the lambda parameter that interpolates between TD(1) and TD(∞).
    
    Args:
        rewards: Array of rewards for each timestep
        values: Array of value estimates for each timestep  
        dones: Array of done flags for each timestep
        gamma: Discount factor for future rewards
        gae_lambda: GAE lambda parameter (0=high bias/low variance, 1=low bias/high variance)
        last_value: Value estimate for the last state (for bootstrapping)
        normalize_advantages: Whether to normalize advantages to have zero mean and unit variance
        
    Returns:
        Tuple of (advantages, returns) as numpy arrays
        
    References:
        "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
        https://arxiv.org/abs/1506.02438
    """
    # Convert inputs to numpy arrays
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32) 
    dones = np.asarray(dones, dtype=np.bool_)
    
    # Validate input shapes
    assert len(rewards) == len(values) == len(dones), \
        f"Input lengths don't match: rewards={len(rewards)}, values={len(values)}, dones={len(dones)}"
    
    # Initialize arrays
    T = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    
    # Compute advantages using GAE
    gae = 0.0
    
    for t in reversed(range(T)):
        if t == T - 1:
            # Last timestep - use provided last_value for bootstrapping
            next_value = last_value
            next_non_terminal = 1.0 - float(dones[t])
        else:
            # Use next state's value
            next_value = values[t + 1]
            next_non_terminal = 1.0 - float(dones[t])
        
        # Temporal difference error
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        
        # GAE computation
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
    
    # Compute returns (advantages + values)
    returns = advantages + values
    
    # Normalize advantages if requested
    if normalize_advantages and len(advantages) > 1:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
    
    return advantages, returns


def compute_gae_torch(rewards: torch.Tensor,
                     values: torch.Tensor, 
                     dones: torch.Tensor,
                     gamma: float = 0.99,
                     gae_lambda: float = 0.95,
                     last_value: float = 0.0,
                     normalize_advantages: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch version of GAE computation for GPU acceleration
    
    Args:
        rewards: Tensor of rewards [T]
        values: Tensor of value estimates [T]
        dones: Tensor of done flags [T] 
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        last_value: Final state value for bootstrapping
        normalize_advantages: Whether to normalize advantages
        
    Returns:
        Tuple of (advantages, returns) tensors
    """
    device = rewards.device
    T = len(rewards)
    
    # Initialize tensors
    advantages = torch.zeros_like(rewards)
    
    # Convert last_value to tensor
    last_value = torch.tensor(last_value, device=device, dtype=rewards.dtype)
    
    # Compute advantages using GAE (backward pass)
    gae = torch.tensor(0.0, device=device, dtype=rewards.dtype)
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_non_terminal = 1.0 - dones[t].float()
        else:
            next_value = values[t + 1] 
            next_non_terminal = 1.0 - dones[t].float()
        
        # TD error
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        
        # GAE update
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
    
    # Compute returns
    returns = advantages + values
    
    # Normalize advantages
    if normalize_advantages and T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def compute_n_step_returns(rewards: np.ndarray,
                          values: np.ndarray,
                          dones: np.ndarray,
                          n_steps: int,
                          gamma: float = 0.99,
                          last_value: float = 0.0) -> np.ndarray:
    """
    Compute n-step returns for policy gradient methods
    
    Args:
        rewards: Rewards array
        values: Value estimates array
        dones: Done flags array
        n_steps: Number of steps for n-step return
        gamma: Discount factor
        last_value: Bootstrap value for final state
        
    Returns:
        Array of n-step returns
    """
    T = len(rewards)
    returns = np.zeros_like(rewards)
    
    for t in range(T):
        return_val = 0.0
        discount = 1.0
        
        # Compute n-step return
        for step in range(n_steps):
            if t + step >= T:
                # Use bootstrap value
                return_val += discount * last_value
                break
            
            return_val += discount * rewards[t + step]
            discount *= gamma
            
            if dones[t + step]:
                break
        else:
            # If we didn't break early, add bootstrapped value
            if t + n_steps < T:
                return_val += discount * values[t + n_steps]
        
        returns[t] = return_val
    
    return returns


class GAEComputer:
    """
    Class-based GAE computer with additional features and logging
    """
    
    def __init__(self,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 normalize_advantages: bool = True,
                 use_gpu: bool = True):
        """
        Initialize GAE computer
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            normalize_advantages: Whether to normalize advantages
            use_gpu: Whether to use GPU acceleration when available
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        self.logger = logging.getLogger('GAEComputer')
        self.logger.info(f"GAE Computer initialized: gamma={gamma}, lambda={gae_lambda}")
        
        # Statistics tracking
        self.computation_stats = {
            'num_computations': 0,
            'total_timesteps': 0,
            'avg_advantage': 0.0,
            'avg_return': 0.0
        }
    
    def compute(self,
                rewards: Union[np.ndarray, torch.Tensor, List[float]],
                values: Union[np.ndarray, torch.Tensor, List[float]], 
                dones: Union[np.ndarray, torch.Tensor, List[bool]],
                last_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE advantages and returns
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            dones: Episode termination flags
            last_value: Bootstrap value
            
        Returns:
            Tuple of (advantages, returns)
        """
        # Convert to appropriate format
        if isinstance(rewards, torch.Tensor) and self.use_gpu:
            # Use GPU computation
            values_tensor = values if isinstance(values, torch.Tensor) else torch.tensor(values)
            dones_tensor = dones if isinstance(dones, torch.Tensor) else torch.tensor(dones)
            
            advantages, returns = compute_gae_torch(
                rewards=rewards,
                values=values_tensor,
                dones=dones_tensor,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                last_value=last_value,
                normalize_advantages=self.normalize_advantages
            )
            
            # Convert back to numpy
            advantages = advantages.cpu().numpy()
            returns = returns.cpu().numpy()
        else:
            # Use CPU computation
            advantages, returns = compute_gae(
                rewards=rewards,
                values=values,
                dones=dones,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                last_value=last_value,
                normalize_advantages=self.normalize_advantages
            )
        
        # Update statistics
        self._update_stats(advantages, returns)
        
        return advantages, returns
    
    def _update_stats(self, advantages: np.ndarray, returns: np.ndarray):
        """Update computation statistics"""
        self.computation_stats['num_computations'] += 1
        self.computation_stats['total_timesteps'] += len(advantages)
        
        # Running averages
        alpha = 0.1  # Exponential moving average factor
        self.computation_stats['avg_advantage'] = (
            alpha * np.mean(advantages) + 
            (1 - alpha) * self.computation_stats['avg_advantage']
        )
        self.computation_stats['avg_return'] = (
            alpha * np.mean(returns) + 
            (1 - alpha) * self.computation_stats['avg_return']
        )
    
    def get_stats(self) -> dict:
        """Get computation statistics"""
        return self.computation_stats.copy()
    
    def reset_stats(self):
        """Reset computation statistics"""
        self.computation_stats = {
            'num_computations': 0,
            'total_timesteps': 0,
            'avg_advantage': 0.0,
            'avg_return': 0.0
        }


def validate_gae_inputs(rewards: np.ndarray, 
                       values: np.ndarray,
                       dones: np.ndarray) -> bool:
    """
    Validate inputs for GAE computation
    
    Args:
        rewards: Rewards array
        values: Values array
        dones: Dones array
        
    Returns:
        True if inputs are valid, False otherwise
    """
    try:
        # Check shapes
        if not (len(rewards) == len(values) == len(dones)):
            return False
        
        # Check for NaN or inf values
        if np.any(np.isnan(rewards)) or np.any(np.isinf(rewards)):
            return False
            
        if np.any(np.isnan(values)) or np.any(np.isinf(values)):
            return False
        
        # Check dtypes
        if not np.issubdtype(rewards.dtype, np.floating):
            return False
            
        if not np.issubdtype(values.dtype, np.floating):
            return False
        
        # Check dones is boolean-like
        if not np.issubdtype(dones.dtype, np.bool_) and not np.issubdtype(dones.dtype, np.integer):
            return False
        
        return True
        
    except Exception:
        return False


def compute_gae_with_validation(rewards: np.ndarray,
                               values: np.ndarray, 
                               dones: np.ndarray,
                               gamma: float = 0.99,
                               gae_lambda: float = 0.95,
                               last_value: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute GAE with input validation and error handling
    
    Args:
        rewards: Rewards array
        values: Values array
        dones: Dones array
        gamma: Discount factor
        gae_lambda: GAE lambda
        last_value: Bootstrap value
        
    Returns:
        Tuple of (advantages, returns) or (None, None) if computation fails
    """
    logger = logging.getLogger('GAE')
    
    try:
        # Validate inputs
        if not validate_gae_inputs(rewards, values, dones):
            logger.error("GAE input validation failed")
            return None, None
        
        # Validate hyperparameters
        if not (0.0 <= gamma <= 1.0):
            logger.error(f"Invalid gamma: {gamma}. Must be in [0, 1]")
            return None, None
            
        if not (0.0 <= gae_lambda <= 1.0):
            logger.error(f"Invalid gae_lambda: {gae_lambda}. Must be in [0, 1]")
            return None, None
        
        # Compute GAE
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
            last_value=last_value
        )
        
        # Validate outputs
        if np.any(np.isnan(advantages)) or np.any(np.isnan(returns)):
            logger.error("GAE computation produced NaN values")
            return None, None
            
        if np.any(np.isinf(advantages)) or np.any(np.isinf(returns)):
            logger.error("GAE computation produced infinite values")
            return None, None
        
        return advantages, returns
        
    except Exception as e:
        logger.error(f"GAE computation failed: {e}")
        return None, None


if __name__ == "__main__":
    # Test GAE computation functionality
    print("Testing GAE Computer...")
    
    try:
        # Create test data
        T = 100
        rewards = np.random.randn(T) * 0.1  # Small rewards
        values = np.random.randn(T) * 0.5   # Value estimates
        dones = np.random.random(T) < 0.05  # 5% episode endings
        
        # Test basic GAE computation
        advantages, returns = compute_gae(rewards, values, dones)
        
        print(f"✅ Basic GAE: advantages shape={advantages.shape}, returns shape={returns.shape}")
        print(f"   Mean advantage: {np.mean(advantages):.6f}")
        print(f"   Mean return: {np.mean(returns):.6f}")
        
        # Test GAE computer class
        gae_computer = GAEComputer(gamma=0.99, gae_lambda=0.95)
        advantages2, returns2 = gae_computer.compute(rewards, values, dones)
        
        # Should be identical to basic computation
        np.testing.assert_allclose(advantages, advantages2, rtol=1e-6)
        np.testing.assert_allclose(returns, returns2, rtol=1e-6)
        print("✅ GAE Computer class matches basic computation")
        
        # Test input validation
        valid = validate_gae_inputs(rewards, values, dones)
        print(f"✅ Input validation: {valid}")
        
        # Test edge cases
        small_rewards = np.array([1.0, 0.0, -1.0])
        small_values = np.array([0.5, 0.3, 0.1])  
        small_dones = np.array([False, False, True])
        
        small_adv, small_ret = compute_gae(small_rewards, small_values, small_dones)
        print(f"✅ Small example: advantages={small_adv}, returns={small_ret}")
        
        # Test GPU computation if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            rewards_gpu = torch.tensor(rewards, device=device)
            values_gpu = torch.tensor(values, device=device)
            dones_gpu = torch.tensor(dones, device=device)
            
            adv_gpu, ret_gpu = compute_gae_torch(rewards_gpu, values_gpu, dones_gpu)
            
            # Compare with CPU computation
            np.testing.assert_allclose(advantages, adv_gpu.cpu().numpy(), rtol=1e-6)
            print("✅ GPU computation matches CPU")
        
        print("✅ All GAE tests passed!")
        
    except Exception as e:
        print(f"❌ GAE test failed: {e}")
        import traceback
        traceback.print_exc()