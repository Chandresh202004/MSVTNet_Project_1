import torch
import argparse
import numpy as np
from msvtnet import MSVTNet
from dataloader import get_dataloader

def find_optimal_batch_size(device, max_batch_size=512, start_batch_size=16):
    """Find the largest batch size that fits in GPU memory"""
    print(f"Finding optimal batch size for {device}...")
    
    # Create a sample model
    model = MSVTNet(in_channels=1, num_classes=4, hidden_size=64).to(device)
    
    # Create criterion
    criterion = torch.nn.CrossEntropyLoss()
    
    # Try different batch sizes
    batch_size = start_batch_size
    while batch_size <= max_batch_size:
        try:
            # Create random inputs and labels
            inputs = torch.randn(batch_size, 1, 22, 875, device=device)
            labels = torch.randint(0, 4, (batch_size,), device=device)
            
            # Forward and backward pass
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                main_out = outputs[0]
            else:
                main_out = outputs
                
            loss = criterion(main_out, labels)
            loss.backward()
            
            print(f"Batch size {batch_size} fits in memory.")
            batch_size *= 2
            
            # Clean up
            del inputs, labels, outputs, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Return the last successful batch size
                optimal_batch_size = batch_size // 2
                print(f"Out of memory with batch size {batch_size}.")
                print(f"Optimal batch size: {optimal_batch_size}")
                return optimal_batch_size
            else:
                raise e
    
    print(f"All tested batch sizes fit in memory. Max tested: {batch_size//2}")
    return batch_size // 2

def main():
    parser = argparse.ArgumentParser(description='Find optimal batch size for GPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--max_batch_size', type=int, default=512, help='Maximum batch size to test')
    parser.add_argument('--start_batch_size', type=int, default=16, help='Starting batch size')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
        
        # Print memory information
        print(f"Total GPU memory: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3:.2f} GB")
        print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        
        optimal_batch_size = find_optimal_batch_size(device, args.max_batch_size, args.start_batch_size)
        
        # Save optimal batch size to a file
        with open('optimal_batch_size.txt', 'w') as f:
            f.write(str(optimal_batch_size))
        
        print(f"Optimal batch size saved to optimal_batch_size.txt")
    else:
        print("CUDA is not available. Cannot determine optimal batch size for GPU.")

if __name__ == "__main__":
    main()