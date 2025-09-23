import torch
import math
from torch import nn
import torch.nn.functional as F
from einops import einsum
from einops import rearrange

# --- Standalone Functions ---

# softmax
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Subtract the maximum value from all the elements in the tensor to maintains tability
    #   `keepdim = True` is crucial for broadcasting to work correctly later.
    #   `torch.max` returns (values, indices), so we take the .values.
    max_val = torch.max(x, dim=dim, keepdim = True).values
    
    # Subtract the max value. Broadcasting handles the shape automatically.
    x_subtracted = x - max_val
    
    # Exponentiate.
    x_exp = torch.exp(x_subtracted)
    
    # Sum along the specified dimension to get the denominator.
    denominator = x_exp.sum(dim = dim, keepdim = True)
    
    # Normalize and return the result. 
    return x_exp/ denominator
    
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    # In the code we use the Q@K^T as the torch uses the row-vector
    # get d_k from the key tensor's shape
    d_k = query.size(-1)
    
    # Calculate scores using sinsum for clarity
    # This multiplies the query ('... q d') and the key ('... k d') along the 'd' dimension
    scores = einsum(query, key, "... q d, ... k d -> ... q k")
    
    # Scale the scores
    scaled_scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply the mask (if exists)
    if mask is not None:
        # Where the mark is False, we replace the score with negative infinity
        # This will make the softmax output for that position zero.
        scaled_scores = scaled_scores.masked_fill(mask == False, -torch.inf)
        
    # Apply the softmax to get probabilities
    attention_probs = softmax(scaled_scores, dim = -1)
    
    # Apply probabilities to the value vectors using einsum
    output = einsum(attention_probs, value, "... q k, ... k v -> ... q v")
    
    return output

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss using the log-sum-exp trick for stability.
    """
    # Step 1: Stabilize the logits by subtracting maximum logit value
    # from each row (each last dim finds one). This prevents overflow when we expoentiate.
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stable_logits = logits - max_logits

    # Step 2: Calculate the log-sum-exp term of the formula.
    # This is the logarithm of the denominatoe of the softmax function.
    log_sum_exp = max_logits.squeeze(-1) + torch.log(torch.exp(stable_logits).sum(dim=-1))
    # log_sum_exp = torch.log(torch.exp(stable_logits).sum(dim=-1, keepdim=True))

    # Step 3：Select the logit scores corresponding to the target tokens.
    # We need to add a dimension to `targets` to use it with `gather`.

    target_logits = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)

    # Step 4: Calculate the loss for each token using the stable formula
    loss_per_token = log_sum_exp - target_logits

    # return the average loss across all tokens
    return loss_per_token.mean()


# --- nn.Module Classes ---

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
    
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype = None):
        super().__init__()
        
        self.eps = eps
        
        # Create a tensor of ones with the shape (d_model,) which can be broadcast
        gain_tensor = torch.ones(d_model, device=device, dtype = dtype)
        
        # Register it as a learnable parameter
        self.g = nn.Parameter(gain_tensor)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remember the original data type
        in_dtype = x.dtype
        
        # Upcast to float32 for numerical stability
        x = x.to(torch.float32)
        
        normalized_x = x/ torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Apply the learnable gain
        output = normalized_x * self.g
        
        # Downcast back to the original data type
        return output.to(in_dtype)
    
# Position-Wise FFN, used to add computational depth to the model
# A small neural network that is applied to each token's vector representation individually
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device= None, dtype = None):
        super().__init__()

        # Transform from d_model -> d_ff
        self.W_1 = Linear(d_model, d_ff, device = device, dtype = dtype)
        self.W_3 = Linear(d_model, d_ff, device = device, dtype = dtype)

        # Transform from d_ff -> d_model
        self.W_2 = Linear(d_ff, d_model, device = device, dtype = dtype)


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Pass the input through the first and third linear layers
        x1 = self.W_1(x)
        x3 = self.W_3(x)

        # Apply the SiLU activation function to the output of W1
        activated_x1 = x1 * F.sigmoid(x1)

        # Element-wise multiply the activated gate with the output of W_3
        gated_output = activated_x1 * x3

        # Pass the result through the final linear layer
        output = self.W_2(gated_output)

        return output

# Relative Positional Embeddings: used to "help" model understand the relative positions of tokens
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype =None):
        super().__init__()
        
        # Create a tensor for the position 'i' from 0 to max_seq_len -1
        # represent the diffreent position of the token
        positions = torch.arange(max_seq_len, device = device)
        
        # Create a tensor for the even dimension indices (0,2,4,...)
        # represent the different hidden_dim of the token
        dim_indices = torch.arange(0, d_k, 2, device = device)
        
        # Calculate the dimension term from the formula
        inv_freq = 1.0 /(theta**(dim_indices.float() / d_k))
        
        # This efficiently combines the position and dimension terns to
        # full grid of theta angles with shape (max_seq_len, d_k /2)
        # torch.outer is used to calculate the outer product of two vectors
        thetas = torch.outer(positions, inv_freq)
        
        # Calculate the sine and cosine values
        sin_cache = torch.sin(thetas)
        cos_cache = torch.cos(thetas)
        
        # This stores a tensor as part of the module's state, but not as a
        # learnable parameter. This is perfect for pre-computed values.
        self.register_buffer('sin_cache', sin_cache.to(dtype))
        self.register_buffer('cos_cache', cos_cache.to(dtype))
        
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        # x has the shape of (..., seq_len, d_k)
        # token_position has shape (..., seq_len)
        
        # 1. Get the correct sin/cos values from your cache for the given positions 
        # Use the lookup tables to find the sin and cos of the corresponding position
        # These will have shape (..., seq_len, d_k/2)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        
        # 2. Split the correct sin/cos values from your cache for the givem positions 
        x_even = x[..., ::2] # Takes all even indices (0,2,4,...)
        x_odd = x[..., 1::2] # Takes all odd indices (1,3,5,...)
        
        # This is equivalent to the matrix multiplication but much faster
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        
        # 4. Combine the rotated halves back into a single tensor
        x_rotated = torch.empty_like(x)
        x_rotated[..., ::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        
        return x_rotated

 
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: nn.Module | None = None, device= None, dtype= None):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of each head
        
        # Create one large projection layer for each of Q, K, and V
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # The final output projection layer
        self.out_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope
             
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        # Project input to get Q, K and V for all heads
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        
        _ , seq_len, _ = x.shape
        device = x.device
        
        # Reshape the Q, K, V to split into nultiple heads
        query = rearrange(query, "b s (h d_k) -> b h s d_k", h= self.num_heads)
        key = rearrange(key, "b s (h d_k) -> b h s d_k", h= self.num_heads)
        value = rearrange(value, "b s (h d_k) -> b h s d_k", h= self.num_heads)
        # Now query has shape: (batch, num_heads, seq_len, d_k)
        
        if self.rope is not None:
            if token_positions is None:
                raise ValueError("token_positions must be provided when RoPE is enabled.")
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)
            
        # --- Create a Causal Mask if one isn't provided --- 
        if mask is None:
            # This creates a square matrix with True on the lower triangle and False on the upper
            # It ensures a token can only attend to itself and previous tokens
            # Use a triangular mask to prevent the front query seeing the latter key
            # That is why the upper part of the tensor is all zeros (we use the row-vector in coding) 
            mask = torch.tril(torch.ones(seq_len, seq_len,device = device, dtype = torch.bool))
        
        # Apply scaled-dot product 
        attention_output  = scaled_dot_product_attention(
            query=query, key=key, value=value, mask=mask)
        
        concatenated_output  = rearrange(attention_output , "b h s d -> b s (h d) ")
        
        final_output = self.out_proj(concatenated_output)
        
        return final_output   
        
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: nn.Module, device = None, dtype = None):
        super().__init__()
        
        self.rmsnorm_mha = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rmsnorm_ffn = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.mha_layer = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads , rope=rope, device=device, dtype=dtype)
        self.ffn_layer = PositionWiseFeedForward(
            d_model=d_model, d_ff=d_ff, device=device, dtype= dtype
        )
        
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # --- Attention Sub-layer (Norm -> MHA -> Add) ---
        
        # The first residual connection starts from the original input `x`
        residual = x
        # Apply RMSNorm 
        norm_x = self.rmsnorm_mha(x)
        # Apply Multi-Head Attention 
        attn_output = self.mha_layer(norm_x, token_positions = token_positions)
        # Add the residual connection 
        h = residual + attn_output
        
        # --- FFN Sub-layer  (Norm -> FFN -> Add) ---
        
        # The second residual connection starts from the output of the first sub-layer, `h`
        residual = h
        # Apply RMSNorm
        norm_h = self.rmsnorm_ffn(h)
        # Apply the Feed-Forward Network
        ffn_output = self.ffn_layer(norm_h)
        # Add the residual connection
        output = residual + ffn_output
        
        return output
    
class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layer: int,
            num_heads: int,
            d_ff: int,
            theta: float,
            device= None,
            dtype= None
    ):
        super().__init__()
        # Create the token embedding layer
        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device= device, dtype= dtype)
        
        d_k = d_model // num_heads
        rope = RoPE(theta= theta, d_k= d_k, max_seq_len= context_length, device= device, dtype = dtype)
        # Create the stack of transformer blocks using nn.ModuleList
        # This creates `num_layers` identical blocks and stores them in a list

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model= d_model, num_heads= num_heads, d_ff= d_ff, 
                                 rope= rope, device= device, dtype= dtype)
                                 for _ in range(num_layer)
            ]
        )

        self.norm = RMSNorm(d_model=d_model, device= device, dtype= dtype)
        # The size of final layer will transform the `d_model` -> `vocab_size`
        self.linear = Linear(d_model, vocab_size, device= device, dtype= dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Get the sequence length from the input shape
        seq_len = token_ids.shape[1]

        # Create the token positions tensor [[0,1,2,...]]
        token_positions = torch.arange(seq_len, device=token_ids.device)

        # Get token embeddings 
        x = self.embedding(token_ids)
        
        # Pass through the stack of Transformer Blocks
        for block in self.blocks:
            # Pass both x and token_positions to each block
            x = block(x, token_positions = token_positions)
        
        # Apply final normalization
        x = self.norm(x)

        logits = self.linear(x)

        return logits
    
