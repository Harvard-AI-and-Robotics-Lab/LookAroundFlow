import torch
from typing import Union, Dict, Tuple
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

class AFloPS_SD35(FlowMatchEulerDiscreteScheduler):
    """
    A-FloPS scheduler for Stable Diffusion 3.5 (native flow matching model).
    Implements adaptive velocity decomposition from the A-FloPS paper.
    """
    
    def __init__(self, *args, **kwargs):
        self.lambda_min = kwargs.pop("lambda_min", -1.0)
        self.lambda_max = kwargs.pop("lambda_max", 1.0)
        super().__init__(*args, **kwargs)
        self._reset_adaptive_state()
    
    def _reset_adaptive_state(self):
        """Reset memory between generations"""
        self._prev_sample = None
        self._prev_velocity = None
        self._prev_residual = None
        self._prev_timestep = None
    
    def compute_adaptive_lambda(
        self,
        current_sample: torch.FloatTensor,
        current_velocity: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute λ^(n) = ⟨Δv, Δx⟩ / ||Δx||² (Eq. 12)"""
        if self._prev_sample is None:
            return torch.zeros(
                current_sample.shape[0], 
                device=current_sample.device, 
                dtype=current_sample.dtype  # ← CRITICAL: Match input dtype
            )
        
        delta_x = current_sample - self._prev_sample
        delta_v = current_velocity - self._prev_velocity
        
        batch_size = delta_x.shape[0]
        delta_x_flat = delta_x.reshape(batch_size, -1)
        delta_v_flat = delta_v.reshape(batch_size, -1)
        
        numerator = torch.sum(delta_v_flat * delta_x_flat, dim=1)
        denominator = torch.sum(delta_x_flat * delta_x_flat, dim=1)
        denominator = torch.clamp(denominator, min=1e-6)
        
        lambda_n = numerator / denominator
        return torch.clamp(lambda_n, self.lambda_min, self.lambda_max).to(dtype=current_sample.dtype)
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.FloatTensor,
        sample: torch.FloatTensor,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Dict, Tuple]:
        """
        Perform A-FloPS integration step with adaptive decomposition.
        model_output: velocity prediction v(x_t, t) from SD3.5
        """
        dtype = sample.dtype
        device = sample.device
        
        t_norm = timestep.to(dtype=dtype) / self.config.num_train_timesteps
        
        v_t = model_output  # Should already be correct dtype
        
        if self._prev_sample is not None:
            lambda_n = self.compute_adaptive_lambda(sample, v_t)  # Now returns correct dtype
            lambda_expanded = lambda_n.reshape(-1, 1, 1, 1).to(dtype=dtype)  # Explicit cast
            
            # Residual term: h_t = v_t - λ_t * x_t
            h_t = v_t - lambda_expanded * sample
            
            # FIX: Cast timestep arithmetic to correct dtype
            dt = (self._prev_timestep.to(dtype=dtype) - timestep.to(dtype=dtype))
            dt = dt / self.config.num_train_timesteps
            dt = dt.reshape(-1, 1, 1, 1)
            
            # Integration coefficients (Eq. 15) - all now in correct dtype
            exp_term = torch.exp(lambda_expanded * dt)
            a = (1 - exp_term) / (lambda_expanded + 1e-8)
            b = (1 - (1 + lambda_expanded * dt) * exp_term) / (lambda_expanded**2 + 1e-8)
            
            # Derivative of residual (Eq. 16)
            dh_dt = (h_t - self._prev_residual) / dt
            
            # Next sample (Eq. 14)
            prev_sample = exp_term * sample + a * h_t + b * dh_dt
        else:
            # First step: Euler integration - FIX dtype
            dt = t_norm.reshape(-1, 1, 1, 1)
            prev_sample = sample + v_t * dt
            h_t = v_t
        
        # Update state
        self._prev_residual = h_t
        self._prev_sample = sample
        self._prev_velocity = v_t
        self._prev_timestep = timestep
        
        return {"prev_sample": prev_sample} if return_dict else (prev_sample,)