import torch
import sys
import subprocess
import platform

def check_nvidia_smi():
    """Run nvidia-smi and capture output"""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.stdout
    except FileNotFoundError:
        return "nvidia-smi command not found. NVIDIA drivers may not be installed."
    except Exception as e:
        return f"Error running nvidia-smi: {e}"

def main():
    print("=" * 50)
    print("GPU DIAGNOSTIC REPORT")
    print("=" * 50)
    
    print("\nPYTHON VERSION:")
    print(sys.version)
    
    print("\nPLATFORM INFO:")
    print(platform.platform())
    
    print("\nPYTORCH VERSION:")
    print(f"PyTorch: {torch.__version__}")
    
    print("\nCUDA AVAILABILITY:")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    # Check if CPU-only version
    if '+cpu' in torch.__version__:
        print("You have the CPU-only version of PyTorch installed.")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Try a simple CUDA operation
        print("\nTesting CUDA operation:")
        try:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = x + x
            print(f"  Success! Result: {y}")
            print(f"  Device: {y.device}")
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\nNVIDIA-SMI OUTPUT:")
    print(check_nvidia_smi())
    
    print("\nTROUBLESHOOTING RECOMMENDATIONS:")
    if not torch.cuda.is_available():
        if '+cpu' in torch.__version__:
            print("- Your PyTorch was not built with CUDA support (CPU-only version).")
            print("- Reinstall PyTorch with CUDA support using instructions from pytorch.org")
        else:
            print("- PyTorch can't find your GPU.")
            print("- Check if NVIDIA drivers are installed properly")
            print("- Make sure your GPU is supported by your CUDA version")
            print("- Try updating NVIDIA drivers")
    
    print("\nRECOMMENDED PYTORCH INSTALLATION COMMAND:")
    print("conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
    print("# or for pip:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()