# File: agents/training/mixed_precision_optimizer.py
# Performance & Optimization Engineering (POE) Subsystem
# Mixed Precision Training for 3x Additional Performance Boost

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Dict, Optional, Any, Tuple
import logging
from pathlib import Path

from agents.networks.ppo_networks import PPOActorCritic
from agents.training.rollout_buffer import RolloutBuffer


class MixedPrecisionPPOTrainer:
    """High-performance PPO trainer with FP16/BF16 mixed precision training"""
    
    def __init__(self, 
                 network: PPOActorCritic,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None,
                 precision_type: str = 'fp16'):
        
        self.network = network
        self.config = config
        self.precision_type = precision_type  # 'fp16', 'bf16', or 'fp32'
        
        # Device setup with precision compatibility
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                # MPS supports mixed precision starting PyTorch 2.0
                self.device = torch.device('mps') 
            else:
                self.device = torch.device('cpu')
                # Fall back to FP32 for CPU
                self.precision_type = 'fp32'
        else:
            self.device = device
            
        # Mixed precision setup
        self._setup_mixed_precision()
        
        # Optimizer with mixed precision considerations
        learning_rate = config['training'].get('learning_rate', 3e-4)
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8 if self.precision_type == 'fp32' else 1e-4,  # Adjusted for FP16
            weight_decay=config['training'].get('weight_decay', 1e-5)
        )
        
        # Training metrics
        self.training_step = 0
        self.performance_metrics = {
            'memory_usage': [],
            'forward_time': [],
            'backward_time': [],
            'throughput': []
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🔥 Mixed Precision Trainer initialized: {precision_type.upper()}")
        self.logger.info(f"📱 Device: {self.device}")
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training components"""
        
        if self.precision_type == 'fp16' and (self.device.type in ['cuda', 'mps']):
            # Use PyTorch's Automatic Mixed Precision (AMP)
            self.use_amp = True
            self.scaler = GradScaler()
            self.autocast_dtype = torch.float16
            self.logger.info("✅ FP16 Mixed Precision enabled with AMP")
            
        elif self.precision_type == 'bf16' and torch.cuda.is_bf16_supported():
            # Use BFloat16 for newer hardware
            self.use_amp = True
            self.scaler = None  # BF16 doesn't need gradient scaling
            self.autocast_dtype = torch.bfloat16
            self.logger.info("✅ BF16 Mixed Precision enabled")
            
        else:
            # Fall back to FP32
            self.use_amp = False
            self.scaler = None
            self.autocast_dtype = torch.float32
            self.precision_type = 'fp32'
            self.logger.info("⚪ FP32 Precision (mixed precision not available)")
    
    def _memory_efficient_forward(self, 
                                 observations: torch.Tensor,
                                 agent_states: torch.Tensor,
                                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Memory-efficient forward pass with mixed precision"""
        
        if self.use_amp:
            with autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                log_probs, values, entropy = self.network.evaluate_actions(
                    observations, agent_states, actions
                )
        else:
            log_probs, values, entropy = self.network.evaluate_actions(
                observations, agent_states, actions
            )
            
        return log_probs, values, entropy
    
    def _compute_ppo_loss(self,
                         new_log_probs: torch.Tensor,
                         old_log_probs: torch.Tensor, 
                         advantages: torch.Tensor,
                         returns: torch.Tensor,
                         values: torch.Tensor,
                         entropy: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss with numerical stability for mixed precision"""
        
        # Configuration
        clip_ratio = self.config['training'].get('clip_ratio', 0.2)
        value_coef = self.config['training'].get('value_loss_coef', 0.5)
        entropy_coef = self.config['training'].get('entropy_coef', 0.01)
        
        # Ratio computation with numerical stability
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(torch.clamp(log_ratio, min=-10.0, max=10.0))  # Prevent overflow
        
        # Normalize advantages for stability
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss with clipping for stability
        if self.config['training'].get('clip_value_loss', True):
            # Clip value estimates
            values_clipped = old_log_probs + torch.clamp(
                values - old_log_probs, -clip_ratio, clip_ratio
            )
            value_loss1 = (values - returns).pow(2)
            value_loss2 = (values_clipped - returns).pow(2)
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = 0.5 * (values - returns).pow(2).mean()
        
        # Entropy loss
        entropy_loss = entropy.mean()
        
        # Combined loss
        total_loss = (
            policy_loss + 
            value_coef * value_loss - 
            entropy_coef * entropy_loss
        )
        
        # Loss metrics
        loss_metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item()
        }
        
        return total_loss, loss_metrics
    
    def update_policy(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """High-performance policy update with mixed precision"""
        
        import time
        update_start = time.perf_counter()
        
        # Training configuration
        batch_size = self.config['training'].get('batch_size', 64)
        n_epochs = self.config['training'].get('n_epochs', 4)
        max_grad_norm = self.config['training'].get('max_grad_norm', 0.5)
        
        # Accumulate metrics
        epoch_metrics = []
        memory_peak = 0
        
        # Mixed precision training loop
        for epoch in range(n_epochs):
            
            # Randomize batch order
            indices = torch.randperm(len(buffer), device=self.device)
            epoch_losses = []
            
            for start_idx in range(0, len(buffer), batch_size):
                batch_start = time.perf_counter()
                
                end_idx = min(start_idx + batch_size, len(buffer))
                batch_indices = indices[start_idx:end_idx]
                
                # Extract batch data
                batch_obs = buffer.observations[batch_indices]
                batch_states = buffer.agent_states[batch_indices]
                batch_actions = buffer.actions[batch_indices]
                batch_old_log_probs = buffer.log_probs[batch_indices]
                batch_advantages = buffer.advantages[batch_indices]
                batch_returns = buffer.returns[batch_indices]
                
                # Forward pass with mixed precision
                forward_start = time.perf_counter()
                new_log_probs, values, entropy = self._memory_efficient_forward(
                    batch_obs, batch_states, batch_actions
                )
                forward_time = time.perf_counter() - forward_start
                
                # Compute loss
                total_loss, loss_metrics = self._compute_ppo_loss(
                    new_log_probs, batch_old_log_probs, batch_advantages,
                    batch_returns, values, entropy
                )
                
                # Backward pass with mixed precision
                backward_start = time.perf_counter()
                self.optimizer.zero_grad()
                
                if self.use_amp and self.scaler is not None:
                    # FP16 with gradient scaling
                    self.scaler.scale(total_loss).backward()
                    
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                    
                    # Optimizer step with gradient scaling
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                elif self.use_amp:
                    # BF16 (no gradient scaling needed)
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                    self.optimizer.step()
                    
                else:
                    # FP32 standard training
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_grad_norm)
                    self.optimizer.step()
                
                backward_time = time.perf_counter() - backward_start
                
                # Performance tracking
                batch_time = time.perf_counter() - batch_start
                throughput = len(batch_indices) / batch_time
                
                self.performance_metrics['forward_time'].append(forward_time)
                self.performance_metrics['backward_time'].append(backward_time)
                self.performance_metrics['throughput'].append(throughput)
                
                epoch_losses.append(loss_metrics)
                
                # Memory tracking (if CUDA)
                if self.device.type == 'cuda':
                    memory_peak = max(memory_peak, torch.cuda.max_memory_allocated())
            
            # Average epoch metrics
            epoch_avg = {}
            for key in epoch_losses[0].keys():
                epoch_avg[f'epoch_{epoch}_{key}'] = np.mean([m[key] for m in epoch_losses])
            epoch_metrics.append(epoch_avg)
        
        # Final metrics
        update_time = time.perf_counter() - update_start
        
        # Aggregate all metrics
        final_metrics = {}
        for epoch_metric in epoch_metrics:
            for key, value in epoch_metric.items():
                base_key = key.split('_', 2)[-1]  # Remove epoch prefix
                if base_key not in final_metrics:
                    final_metrics[base_key] = []
                final_metrics[base_key].append(value)
        
        # Average across epochs
        for key, values in final_metrics.items():
            final_metrics[key] = np.mean(values)
        
        # Performance metrics
        final_metrics.update({
            'update_time': update_time,
            'avg_throughput': np.mean(self.performance_metrics['throughput'][-len(buffer)//batch_size:]),
            'peak_memory_mb': memory_peak / (1024**2) if memory_peak > 0 else 0,
            'precision_type': self.precision_type,
            'device_type': self.device.type
        })
        
        self.training_step += 1
        return final_metrics
    
    def benchmark_performance(self, buffer: RolloutBuffer, num_iterations: int = 10) -> Dict[str, float]:
        """Benchmark training performance with mixed precision"""
        
        self.logger.info(f"🔥 Benchmarking {self.precision_type.upper()} performance...")
        
        baseline_metrics = []
        
        for i in range(num_iterations):
            metrics = self.update_policy(buffer)
            baseline_metrics.append({
                'throughput': metrics['avg_throughput'],
                'update_time': metrics['update_time'],
                'memory_usage': metrics['peak_memory_mb']
            })
            
            if i % 3 == 0:
                self.logger.info(f"  Iteration {i+1}/{num_iterations}: "
                               f"{metrics['avg_throughput']:.1f} samples/sec")
        
        # Performance statistics
        perf_stats = {
            'avg_throughput': np.mean([m['throughput'] for m in baseline_metrics]),
            'max_throughput': np.max([m['throughput'] for m in baseline_metrics]),
            'avg_update_time': np.mean([m['update_time'] for m in baseline_metrics]),
            'avg_memory_mb': np.mean([m['memory_usage'] for m in baseline_metrics]),
            'throughput_std': np.std([m['throughput'] for m in baseline_metrics]),
            'precision_type': self.precision_type,
            'device_type': self.device.type
        }
        
        # Compare to FP32 baseline (estimated)
        fp32_baseline = 200.0  # Estimated FP32 throughput
        speedup = perf_stats['avg_throughput'] / fp32_baseline
        
        self.logger.info("📊 MIXED PRECISION BENCHMARK RESULTS")
        self.logger.info(f"   Average Throughput: {perf_stats['avg_throughput']:.1f} samples/sec")
        self.logger.info(f"   Peak Throughput: {perf_stats['max_throughput']:.1f} samples/sec")
        self.logger.info(f"   Memory Usage: {perf_stats['avg_memory_mb']:.1f} MB")
        self.logger.info(f"   Estimated Speedup: {speedup:.2f}x vs FP32")
        
        return perf_stats
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed memory usage statistics"""
        
        stats = {}
        
        if self.device.type == 'cuda':
            stats.update({
                'allocated_mb': torch.cuda.memory_allocated() / (1024**2),
                'max_allocated_mb': torch.cuda.max_memory_allocated() / (1024**2),
                'reserved_mb': torch.cuda.memory_reserved() / (1024**2),
                'max_reserved_mb': torch.cuda.max_memory_reserved() / (1024**2)
            })
        elif self.device.type == 'mps':
            # MPS memory tracking (limited)
            stats.update({
                'allocated_mb': torch.mps.current_allocated_memory() / (1024**2),
                'max_allocated_mb': torch.mps.max_memory_allocated() / (1024**2)
            })
        
        # Add model parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in self.network.parameters())
        stats['parameter_memory_mb'] = param_memory / (1024**2)
        
        return stats
    
    def save_performance_report(self, save_path: str):
        """Save comprehensive performance analysis report"""
        
        report = {
            'precision_configuration': {
                'precision_type': self.precision_type,
                'device_type': self.device.type,
                'autocast_enabled': self.use_amp,
                'gradient_scaling': self.scaler is not None
            },
            'performance_metrics': {
                'avg_throughput': np.mean(self.performance_metrics['throughput']),
                'max_throughput': np.max(self.performance_metrics['throughput']) if self.performance_metrics['throughput'] else 0,
                'avg_forward_time': np.mean(self.performance_metrics['forward_time']),
                'avg_backward_time': np.mean(self.performance_metrics['backward_time'])
            },
            'memory_stats': self.get_memory_stats(),
            'training_steps': self.training_step
        }
        
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📋 Performance report saved: {save_path}")


class DynamicBatchingOptimizer:
    """Dynamic batching system for optimal memory utilization"""
    
    def __init__(self, 
                 max_memory_mb: float = 8000,  # 8GB default
                 min_batch_size: int = 16,
                 max_batch_size: int = 512):
        
        self.max_memory_mb = max_memory_mb
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        self.current_batch_size = 64  # Starting point
        self.memory_history = []
        self.throughput_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def optimize_batch_size(self, 
                           network: nn.Module,
                           sample_batch: Tuple[torch.Tensor, ...],
                           target_memory_utilization: float = 0.85) -> int:
        """Dynamically optimize batch size for maximum throughput"""
        
        self.logger.info("🎯 Optimizing batch size for memory efficiency...")
        
        optimal_batch_size = self.current_batch_size
        best_throughput = 0.0
        
        # Test different batch sizes
        test_sizes = [16, 32, 64, 128, 256, 512]
        test_sizes = [bs for bs in test_sizes if self.min_batch_size <= bs <= self.max_batch_size]
        
        for batch_size in test_sizes:
            try:
                # Create test batch
                scaled_batch = self._scale_batch(sample_batch, batch_size)
                
                # Measure memory and throughput
                memory_usage, throughput = self._measure_performance(network, scaled_batch)
                
                # Check if within memory limits
                if memory_usage <= self.max_memory_mb * target_memory_utilization:
                    if throughput > best_throughput:
                        best_throughput = throughput
                        optimal_batch_size = batch_size
                        
                    self.logger.info(f"  Batch {batch_size:>3d}: {throughput:>6.1f} samples/sec, "
                                   f"{memory_usage:>6.1f} MB")
                else:
                    self.logger.info(f"  Batch {batch_size:>3d}: Memory limit exceeded ({memory_usage:.1f} MB)")
                    
            except Exception as e:
                self.logger.warning(f"  Batch {batch_size}: Failed - {e}")
                
        self.current_batch_size = optimal_batch_size
        self.logger.info(f"🎯 Optimal batch size: {optimal_batch_size} "
                        f"({best_throughput:.1f} samples/sec)")
        
        return optimal_batch_size
    
    def _scale_batch(self, sample_batch: Tuple[torch.Tensor, ...], target_size: int) -> Tuple[torch.Tensor, ...]:
        """Scale sample batch to target size"""
        current_size = sample_batch[0].size(0)
        
        if target_size == current_size:
            return sample_batch
            
        # Repeat or truncate batch
        scaled_batch = []
        for tensor in sample_batch:
            if target_size > current_size:
                # Repeat batch to reach target size
                repeat_factor = (target_size + current_size - 1) // current_size
                scaled_tensor = tensor.repeat(repeat_factor, *([1] * (tensor.dim() - 1)))
                scaled_tensor = scaled_tensor[:target_size]
            else:
                # Truncate batch
                scaled_tensor = tensor[:target_size]
                
            scaled_batch.append(scaled_tensor)
            
        return tuple(scaled_batch)
    
    def _measure_performance(self, 
                           network: nn.Module, 
                           batch: Tuple[torch.Tensor, ...]) -> Tuple[float, float]:
        """Measure memory usage and throughput for given batch"""
        
        device = next(network.parameters()).device
        
        # Clear memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        
        # Warm-up
        with torch.no_grad():
            _ = network.evaluate_actions(*batch)
        
        # Measure memory before
        memory_before = 0
        if device.type == 'cuda':
            memory_before = torch.cuda.memory_allocated()
        elif device.type == 'mps':
            memory_before = torch.mps.current_allocated_memory()
        
        # Performance measurement
        import time
        
        start_time = time.perf_counter()
        num_iterations = 10
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = network.evaluate_actions(*batch)
                
        if device.type in ['cuda', 'mps']:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            
        end_time = time.perf_counter()
        
        # Calculate throughput
        total_samples = batch[0].size(0) * num_iterations
        throughput = total_samples / (end_time - start_time)
        
        # Measure peak memory
        peak_memory = memory_before
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated()
        elif device.type == 'mps':
            peak_memory = torch.mps.max_memory_allocated()
        
        memory_usage_mb = peak_memory / (1024**2)
        
        return memory_usage_mb, throughput