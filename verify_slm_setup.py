"""
SLM Setup Verification & Quick Start

This script checks if you have everything needed to run the Modern SLM
and provides a quick demo.

Run this first to verify your setup!
"""
import sys
import os

def check_file(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  [{status}] {description}: {filepath}")
    return exists

def check_import(module_name, package_name=None):
    """Check if a Python module can be imported"""
    try:
        __import__(module_name)
        print(f"  [✓] {module_name}")
        return True
    except ImportError:
        pkg = package_name or module_name
        print(f"  [✗] {module_name} - Install with: pip install {pkg} --break-system-packages")
        return False

def main():
    print("\n" + "="*70)
    print("MODERN SLM - SETUP VERIFICATION")
    print("="*70)
    
    all_good = True
    
    # 1. Check Python packages
    print("\n1️⃣  Checking Python packages...")
    packages = [
        ("torch", "torch"),
        ("tiktoken", "tiktoken"),
    ]
    for module, package in packages:
        if not check_import(module, package):
            all_good = False
    
    # 2. Check required files
    print("\n2️⃣  Checking required files...")
    
    # SLM files
    files_slm = [
        ("modern_slm.py", "Modern SLM architecture"),
        ("train_modern_slm.py", "Training script"),
        ("chatbot_slm.py", "Chatbot interface"),
    ]
    
    for filepath, desc in files_slm:
        if not check_file(filepath, desc):
            all_good = False
    
    # Prototype dependencies
    print("\n3️⃣  Checking Prototype dependencies...")
    prototype_files = [
        ("Prototype/Dataset.py", "Dataset loader"),
        ("the-verdict.txt", "Training data"),
    ]
    
    for filepath, desc in prototype_files:
        if not check_file(filepath, desc):
            all_good = False
    
    # 4. Optional: Check if model is trained
    print("\n4️⃣  Checking trained models (optional)...")
    model_exists = check_file("modern_slm.pth", "Trained SLM model")
    
    # Summary
    print("\n" + "="*70)
    if all_good:
        print("✅ ALL REQUIRED FILES AND PACKAGES PRESENT!")
        print("="*70)
        
        if not model_exists:
            print("\n📝 Next step: Train your model")
            print("   python train_modern_slm.py")
        else:
            print("\n🎉 Everything ready! You can:")
            print("   python chatbot_slm.py    # Chat with your SLM")
            print("   python modern_slm.py      # Compare GPT vs SLM architectures")
        
        # Run quick demo
        print("\n" + "-"*70)
        print("QUICK ARCHITECTURE DEMO")
        print("-"*70)
        
        try:
            sys.path.append('Prototype')
            import torch
            from modern_slm import ModernSLM, TINY_SLM_CONFIG
            
            print("\nCreating Modern SLM with configuration:")
            for key, value in TINY_SLM_CONFIG.items():
                print(f"  {key:20s}: {value}")
            
            model = ModernSLM(TINY_SLM_CONFIG)
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"\n✓ Model created successfully!")
            print(f"  Total parameters: {total_params:,}")
            print(f"  KV groups: {TINY_SLM_CONFIG['n_kv_groups']}")
            print(f"  Architecture: Grouped Query Attention + RMSNorm + SwiGLU")
            
            # Test forward pass
            dummy_input = torch.randint(0, 1000, (1, 64))
            with torch.no_grad():
                output = model(dummy_input)
            print(f"  Test forward pass: ✓ (output shape: {output.shape})")
            
            print("\n🎯 The SLM is working correctly!")
            
        except Exception as e:
            print(f"\n⚠️  Could not run demo: {e}")
    
    else:
        print("❌ MISSING REQUIRED FILES OR PACKAGES")
        print("="*70)
        print("\nPlease install missing packages and ensure files are present.")
    
    print()

if __name__ == "__main__":
    main()