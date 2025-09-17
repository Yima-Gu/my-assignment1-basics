import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device =None, dtype = None):
        super().__init__()
        
        # 1. Create an empty tensor with the correct shape
        weight_tensor = torch.empty(in_features, out_features, device= device, dtype= dtype)
        
        # 2. Calculate the standard deviation from the rule in the assignment
        variance = 2 /(in_features + out_features)
        std = variance**0.5
        
        # 3. Initialize the tensor's value in-place 
        torch.nn.init.trunc_normal_(weight_tensor, mean = 0.0, std = std, a=-30, b =30)
        
        # 4. Wrap up the initialized tensor in nn.Parameter
        self.W = nn.Parameter(weight_tensor)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return x @ self.W
    
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,  device = None, dtype = None):
        super().__init__()
        # 1. Create the embedding matrix tensor with the correct shape (num_embeddings*embedding_dim)
        embedding_matrix = torch.empty(num_embeddings, embedding_dim, device = device, dtype = dtype)
        
        # 2. Initialize it using the rule from the assignment
        torch.nn.init.trunc_normal_(embedding_matrix, mean = 0.0, std = 1.0, a = -3.0, b =3.0)
        
        # 3. Register it as a learnable parameter.
        self.embedding_matrix = nn.Parameter(embedding_matrix)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Pytorch automatically looks up the vector for each ID
        # and return a new tensor with the results.
        return self.embedding_matrix[token_ids]