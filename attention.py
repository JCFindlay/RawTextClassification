import torch
import numpy as np
import math

def scaled_dot_product_attention(query: torch.tensor, key: torch.tensor, value: torch.tensor):
    
    #matmul 
    q_k_result = torch.matmul(query, torch.transpose(key, -2, -1))
    # these above should both be tensors
    
    
    scaled = q_k_result/math.sqrt(query.shape[-1])
    
    # Softmax
    
    sm = torch.nn.Softmax(dim=-1)
    attention_matrix = sm(scaled)
    
    # Last matmul
    
    s_d_p_attention = torch.matmul(attention_matrix, value)
    
    return s_d_p_attention 
    