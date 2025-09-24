import torch
from torch import nn
import math


# --- Standard Functions ---

def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    Calculates the learning rate for a given step `t` using a cosine schedule with warmup.
    
    Args:
        t: The current training step.
        alpha_max: The maximum learning rate after warmup.
        alpha_min: The minimum learning rate after annealing.
        T_w: The number of warmup steps.
        T_c: The step at which annealing finishes.
    """
    # Warm-up phase: Linear increase from 0 to alpha_max
    if t< T_w:
        return (t * alpha_max) / T_w
    
    # Cosine annealing phase: Smooth decay from alpha_max to alpha_min
    elif t <= T_c:
        # Calculate how far along we are in the annealing phase (from 1 to 0)
        progress = (t - T_w) / (T_c - T_w)
        # Calculate the cosine component, which goes from 1 to 0
        cosine_component = 0.5 *(1+math.cos(math.pi * progress))
        
        return alpha_min + cosine_component*(alpha_max - alpha_min)
    
    # Post-annealing phase: Constant minimum learning rate 
    else:
        return alpha_min

def gradient_clipping(parameters: list[nn.Parameter], max_norm: float, eps = 1e-6):
    """
    Clips the gradients of a list of parameters in-place.
    """
    # Calculate the total L2 norm of all gradients combioned
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            # Calculate the L2 Norm (Eulidean norm) of this parameter's gradient
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            
    # Take the square root to get the final total L2 norm.
    total_norm = total_norm**0.5
    
    # Check if the total norm exceeds the threshold
    if total_norm > max_norm:
        # Calculate the clipping ratio and scale all gradients
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                # Multiply each gradient by the scaling factor in-place.
                # The `_` at the end of `mul_` signifies an in-place operation.
                p.grad.data.mul_(clip_coef)
            

# --- Classes ---

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-2):
        defaults = dict(lr= lr, betas = betas, eps= eps, weight_decay = weight_decay)
        super().__init__(params, defaults)
        
    def step(self):
        # Loop through each parameter group (e.g., one for weights, one for biases)
        for group in self.param_groups:
            # get the hyperparameters for this specific group
            lr = group['lr']
            beta_1, beta_2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            # Loop through each parameter in the group
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Initialize the state for this parameter if it's the first step
                if len(state) == 0:
                    state['step'] =0
                    state['m'] = torch.zeros_like(p.data)   # 1st moment vector
                    state['v'] = torch.zeros_like(p.data)   # 2nd moment vector
                    
                m, v = state['m'], state['v']
                state['step']+=1
                t = state['step']
                
                # --- The AdamW Algorithm Steps ---
                
                # Update moment estimates (m & v)
                m.mul_(beta_1).add_(grad, alpha = 1- beta_1)
                v.mul_(beta_2).addcmul_(grad, grad, value = 1 -beta_2)
                
                # Compute bias-corrected learning rate for this step
                bias_correction1 = 1 - beta_1 ** t
                bias_correction2 = 1 - beta_2 ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # Update the parameter's data IN-PLACE
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value = -step_size)
                
                # Apply decoupled weight decay IN-PLACE
                if weight_decay != 0:
                    p.data.add_(p.data, alpha = -lr * weight_decay)
                
            
        # beta_1 = self.defaults['betas'][0]
        # beta_2 = self.defaults['betas'][1]
        # for p in self.params:
        #     t = self.state['t']
        #     alpha =  self.state['alpha']
        #     alpha_t = alpha* torch.sqrt(1- beta_2**t) / ( 1 - beta_1**t)
        #     self.state['alpha_t'] = alpha_t
        #     self.state['t']  =t+1
        #     g = p.grad
        #     m = beta_1*self.state[p].m + (1-beta_1) *g
        #     v = beta_2*self.state[p].v + (1-beta_2) *(g**2)
        #     self.state[p].g = g
        #     self.state[p].m = m
        #     self.state[p].v = v
        #     p = p - alpha_t *m / (torch.sqrt(v) + self.defaluts['eps'])
        #     p = p - alpha* self.defaults['weight_decay']* p
            
