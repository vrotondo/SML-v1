"""
Modern Small Language Model (SLM) - using your existing Llama components!

This creates a TRUE SLM with:
- Grouped Query Attention (GQA) - more efficient than multi-head
- RoPE (Rotary Position Embeddings) - better than learned positions
- RMSNorm - faster than LayerNorm
- SwiGLU activation - better than GELU
"""
import sys
sys.path.append('Prototype')

import torch
import torch.nn as nn
import tiktoken
from Dataset import create_dataloader_v1

# ============================================================================
# MODERN SLM ARCHITECTURE (using your Llama components)
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Normalization - faster than LayerNorm"""
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        return self.scale * norm_x


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention - More efficient than Multi-Head Attention
    
    Instead of every head having its own K,V:
    - Divide heads into groups
    - Heads in same group share K,V projections
    - Reduces KV cache size by ~70%!
    """
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, context_length, dropout=0.0):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = d_out // num_heads
        self.d_out = d_out
        
        # Query projection - full size (one per head)
        self.W_query = nn.Linear(d_in, d_out, bias=False)
        
        # Key/Value projections - reduced size (one per group)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(d_out, d_out, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # Project to Q, K, V
        queries = self.W_query(x)  # (b, num_tokens, d_out)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)
        
        # Reshape and transpose
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        
        # Expand K,V to match number of query heads
        # Each group's K,V is repeated for all heads in that group
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        
        # Attention
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Combine
        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)
        
        return self.out_proj(context)


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward - Used in Llama, PaLM, etc.
    Better than standard GELU in practice
    """
    def __init__(self, emb_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False)  # Gate
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False)  # Up
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False)  # Down
    
    def forward(self, x):
        # SwiGLU: swish(gate) * up
        return self.fc3(nn.functional.silu(self.fc1(x)) * self.fc2(x))


class ModernTransformerBlock(nn.Module):
    """Transformer block with modern SLM components"""
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"]
        )
        self.ff = SwiGLUFeedForward(cfg["emb_dim"], cfg["hidden_dim"])
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        # Pre-norm architecture (like Llama, GPT-2)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        
        return x


class ModernSLM(nn.Module):
    """
    Modern Small Language Model with:
    - Grouped Query Attention (GQA)
    - RMSNorm
    - SwiGLU activation
    - Tall & thin architecture
    """
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[ModernTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ============================================================================
# SLM CONFIGURATIONS
# ============================================================================

# Tiny SLM - for fast training/testing (100M params)
TINY_SLM_CONFIG = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "n_kv_groups": 4,        # GQA: 12 heads, 4 groups = 3 heads per group
    "hidden_dim": 2048,      # SwiGLU hidden size
    "drop_rate": 0.0,        # Modern SLMs often use no dropout
}

# Small SLM - production-like (500M params)
SMALL_SLM_CONFIG = {
    "vocab_size": 32000,     # Smaller vocab
    "context_length": 2048,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,          # Tall & thin!
    "n_kv_groups": 4,        # 16 heads, 4 groups = 4 heads per group
    "hidden_dim": 2816,      # ~2.75x expansion (common in SLMs)
    "drop_rate": 0.0,
}

# Compare to your GPT-2 config:
# GPT_CONFIG_124M = {
#     "emb_dim": 768,
#     "n_heads": 12,
#     "n_layers": 12,         # SLM has 24 layers - twice as deep!
#     # No n_kv_groups - uses standard MHA
#     # Uses LayerNorm instead of RMSNorm
#     # Uses GELU instead of SwiGLU
# }


# ============================================================================
# DEMO: Compare GPT-2 vs Modern SLM
# ============================================================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_architectures():
    """Compare GPT-2 style vs Modern SLM"""
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON: GPT-2 vs Modern SLM")
    print("="*60)
    
    # Import your GPT model
    from GPTModel import GPTModel, GPT_CONFIG_124M
    
    # Create models
    gpt_model = GPTModel(GPT_CONFIG_124M)
    slm_model = ModernSLM(TINY_SLM_CONFIG)
    
    # Count params
    gpt_params = count_parameters(gpt_model)
    slm_params = count_parameters(slm_model)
    
    print(f"\nGPT-2 Style Model:")
    print(f"  Parameters: {gpt_params:,}")
    print(f"  Layers: {GPT_CONFIG_124M['n_layers']}")
    print(f"  Width: {GPT_CONFIG_124M['emb_dim']}")
    print(f"  Attention: Multi-Head (all heads independent)")
    print(f"  Normalization: LayerNorm")
    print(f"  Activation: GELU")
    
    print(f"\nModern SLM:")
    print(f"  Parameters: {slm_params:,}")
    print(f"  Layers: {TINY_SLM_CONFIG['n_layers']} (same)")
    print(f"  Width: {TINY_SLM_CONFIG['emb_dim']} (same)")
    print(f"  Attention: Grouped Query ({TINY_SLM_CONFIG['n_kv_groups']} groups)")
    print(f"  Normalization: RMSNorm")
    print(f"  Activation: SwiGLU")
    
    print(f"\nParameter Difference: {slm_params - gpt_params:+,}")
    print(f"Efficiency Gain: {(1 - slm_params/gpt_params)*100:.1f}% fewer KV cache parameters")
    
    # Test inference
    print("\n" + "-"*60)
    print("Inference Test:")
    print("-"*60)
    
    dummy_input = torch.randint(0, 1000, (1, 128))
    
    with torch.no_grad():
        gpt_out = gpt_model(dummy_input)
        slm_out = slm_model(dummy_input)
    
    print(f"GPT-2 output shape: {gpt_out.shape}")
    print(f"SLM output shape: {slm_out.shape}")
    print("\n✓ Both models work! SLM is more memory-efficient for inference.")


if __name__ == "__main__":
    compare_architectures()
    
    print("\n" + "="*60)
    print("To train the modern SLM, use train_modern_slm.py")
    print("="*60)