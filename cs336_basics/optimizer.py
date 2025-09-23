import torch
import math

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
            
            