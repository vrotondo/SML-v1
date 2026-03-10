"""
Interactive Chatbot for Modern SLM

Usage:
    python chatbot_slm.py

Make sure you've trained the model first with:
    python train_modern_slm.py
"""
import sys
sys.path.append('Prototype')

import torch
import tiktoken
from modern_slm import ModernSLM

def generate_text(model, idx, max_new_tokens, context_size, 
                 temperature=0.8, top_k=50, top_p=0.9):
    """
    Generate text with temperature, top-k, and top-p sampling
    """
    model.eval()
    
    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx[:, -context_size:]
        
        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus on last time step
        logits = logits[:, -1, :]
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1
            )
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')
        
        # Convert to probabilities and sample
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

def load_model(model_path, device):
    """Load trained SLM from checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        
        model = ModernSLM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        return model, config
    except FileNotFoundError:
        print(f"\n❌ Error: Model file '{model_path}' not found.")
        print("\n📝 Please train the model first:")
        print("   python train_modern_slm.py")
        print("\nThis will create the 'modern_slm.pth' file.\n")
        sys.exit(1)

def chat_with_slm(model, tokenizer, config, device):
    """Interactive chat loop"""
    print("\n" + "="*60)
    print("🤖 Modern SLM Chatbot (GQA + RMSNorm + SwiGLU)")
    print("="*60)
    print("\nFeatures:")
    print("  • Grouped Query Attention (67% less KV cache)")
    print("  • RMSNorm (10-15% faster)")
    print("  • SwiGLU activation")
    print("\nType 'quit' to exit, 'temp X' to set temperature")
    print("="*60 + "\n")
    
    context_size = config["context_length"]
    temperature = 0.8
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!\n")
            break
        
        # Temperature adjustment
        if user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                temperature = max(0.1, min(2.0, temperature))
                print(f"🌡️  Temperature set to {temperature}")
                continue
            except:
                print("❌ Invalid temperature. Use: temp 0.7")
                continue
        
        # Convert to tokens
        token_ids = tokenizer.encode(user_input)
        token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)
        
        # Generate response
        print("\n🤔 Thinking...", end='\r')
        with torch.no_grad():
            output_ids = generate_text(
                model=model,
                idx=token_ids_tensor,
                max_new_tokens=100,
                context_size=context_size,
                temperature=temperature,
                top_k=50,
                top_p=0.9
            )
        
        # Convert back to text
        response = tokenizer.decode(output_ids.squeeze(0).tolist())
        
        # Extract just the generated part
        response_only = response[len(user_input):].strip()
        
        print(f"Bot: {response_only}\n")

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\n🚀 Loading Modern SLM...")
    print(f"   Device: {device}")
    
    # Load model
    model, config = load_model('modern_slm.pth', device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {total_params:,}")
    print(f"   Context length: {config['context_length']}")
    print(f"   KV groups: {config['n_kv_groups']}")
    print("   ✓ Model loaded successfully!")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Start chat
    chat_with_slm(model, tokenizer, config, device)

if __name__ == "__main__":
    main()