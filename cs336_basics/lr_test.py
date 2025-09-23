import torch 
from torch import nn

# A minimal SGD optimizer for this experiment
class SimpleSGD(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, dict(lr= lr))
        
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is not None:
                    # This is the SGD update rule: w = w -lr * grad
                    p.data.add_(p.grad.data, alpha = -lr)
                    
def run_sgd_experiment(learning_rate: float):
    print(f"\n --- Testing Learning Rate: {learning_rate} --- ")
    weights = nn.Parameter(5* torch.randn((10, 10)))
    optimizer = SimpleSGD([weights], lr = learning_rate)
    
    # Run for 10 steps as specified in the problem 
    for t in range(10):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        print(f"Step {t}: Loss = {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        
        
# --- Run the experiments ---
run_sgd_experiment(learning_rate=1.0)
run_sgd_experiment(learning_rate=0.1)
run_sgd_experiment(learning_rate=0.01)
run_sgd_experiment(learning_rate=0.001)

