"""
Training script for Modern SLM

Trains a Small Language Model using:
- Grouped Query Attention (GQA)
- RMSNorm
- SwiGLU activation
"""
import sys
sys.path.append('Prototype')

import torch
import tiktoken
from modern_slm import ModernSLM, TINY_SLM_CONFIG
from Dataset import create_dataloader_v1

def calc_loss_batch(input_batch, target_batch, model, device):
    """Calculate loss for a single batch"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """Calculate average loss over data loader"""
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """Simple greedy text generation"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate_and_print_sample(model, tokenizer, device, start_context):
    """Generate and print a sample from the model"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded_tensor,
            max_new_tokens=50,
            context_size=context_size
        )
    
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(f"\n{decoded_text}\n")
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, 
                      num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    """Training loop"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Generate sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def main():
    print("\n" + "="*60)
    print("Training Modern SLM with GQA + RMSNorm + SwiGLU")
    print("="*60 + "\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Create dataloaders
    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_loader = create_dataloader_v1(
        raw_text[:split_idx],
        batch_size=2,
        max_length=TINY_SLM_CONFIG["context_length"],
        stride=TINY_SLM_CONFIG["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        raw_text[split_idx:],
        batch_size=2,
        max_length=TINY_SLM_CONFIG["context_length"],
        stride=TINY_SLM_CONFIG["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    # Initialize model
    torch.manual_seed(123)
    model = ModernSLM(TINY_SLM_CONFIG)
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Print config
    print("\nModel Configuration:")
    print(f"  Layers: {TINY_SLM_CONFIG['n_layers']}")
    print(f"  Embedding dim: {TINY_SLM_CONFIG['emb_dim']}")
    print(f"  Attention heads: {TINY_SLM_CONFIG['n_heads']}")
    print(f"  KV groups: {TINY_SLM_CONFIG['n_kv_groups']}")
    print(f"  Hidden dim: {TINY_SLM_CONFIG['hidden_dim']}")
    print(f"  Context length: {TINY_SLM_CONFIG['context_length']}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    
    # Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Train
    print("\nStarting training...\n")
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': TINY_SLM_CONFIG
    }, 'modern_slm.pth')
    print("\n✓ Model saved to modern_slm.pth")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

if __name__ == "__main__":
    main()