import torch
import sys

print("=" * 60)
print("CUDA Verification Report")
print("=" * 60)

# PyTorch version
print(f"\n✓ PyTorch Version: {torch.__version__}")

# CUDA availability
cuda_available = torch.cuda.is_available()
print(f"✓ CUDA Available: {cuda_available}")

if cuda_available:
    # CUDA version
    print(f"✓ CUDA Version: {torch.version.cuda}")
    
    # Device count
    device_count = torch.cuda.device_count()
    print(f"✓ Number of GPUs: {device_count}")
    
    # Device information
    for i in range(device_count):
        print(f"\n--- GPU {i} ---")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_capability(i)}")
    
    # Test CUDA tensor operations
    print(f"\n✓ Testing CUDA tensor operations...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = x @ y
        print(f"  Matrix multiplication successful!")
        print(f"  Result tensor device: {z.device}")
    except Exception as e:
        print(f"  ERROR: {e}")
else:
    print("\n❌ CUDA is NOT available!")
    print("\nPossible reasons:")
    print("1. PyTorch was not installed with CUDA support")
    print("2. NVIDIA driver is not installed")
    print("3. GPU is not detected")
    sys.exit(1)

print("\n" + "=" * 60)
print("CUDA is ready for training! 🚀")
print("=" * 60)
