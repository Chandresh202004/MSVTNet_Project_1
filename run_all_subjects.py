import os
import argparse
import subprocess
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import signal

def parse_args():
    # Your existing argument parsing code
    parser = argparse.ArgumentParser(description='Run MSVTNet training for all subjects')
    # ... (keep all your existing arguments)
    return parser.parse_args()

def verify_train_py_works():
    """Verify that train.py can be executed with a simple test"""
    print("Verifying train.py...")
    try:
        # Set a longer timeout for verification
        result = subprocess.run(['python', 'train.py', '--subject', '1', '--epochs', '1'], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, timeout=30)
        if result.returncode == 0:
            print("train.py verification passed.")
            return True
        else:
            print(f"train.py verification failed. Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("train.py verification timed out. This might indicate an import or initialization issue.")
        return False
    except Exception as e:
        print(f"Error verifying train.py: {e}")
        return False

def run_training(subject, args):
    """Run training for a single subject with timeout handling"""
    # Your existing run_training function with an added timeout mechanism
    
    # Use a more realistic timeout (e.g., 3600 seconds = 1 hour per subject)
    timeout = 3600  
    
    cmd = [
        'python', 'train.py',
        f'--subject={subject}',
        f'--data_path={args.data_path}',
        f'--save_dir={args.save_dir}',
        f'--results_dir={args.results_dir}',
        f'--epochs={args.epochs}',
        f'--batch_size={args.batch_size}',
        f'--lr={args.lr}',
        f'--hidden_size={args.hidden_size}',
        f'--dropout={args.dropout}',
        f'--weight_decay={args.weight_decay}',
        f'--scheduler={args.scheduler}',
        f'--early_stopping={args.early_stopping}',
        f'--gpu_id={args.gpu_id}',
        f'--num_workers={args.num_workers}'
    ]
    
    if args.session_dependent:
        cmd.append('--session_dependent')
        cmd.append(f'--train_session={args.train_session}')
        cmd.append(f'--test_session={args.test_session}')
    
    if args.mixed_precision:
        cmd.append('--mixed_precision')
    
    if args.augment:
        cmd.append('--augment')
    
    try:
        print(f"Running command: {' '.join(cmd)}", flush=True)
        
        # Run with timeout
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                               text=True, timeout=timeout)
        
        # Process the output
        output = process.stdout
        
        # Look for accuracy info in the output
        best_acc = 0.0
        final_acc = 0.0
        
        for line in output.splitlines():
            if "Best test accuracy:" in line:
                try:
                    acc_part = line.split("Best test accuracy:")[1].strip()
                    best_acc = float(acc_part.split()[0])
                except:
                    pass
            
            if "Final test accuracy:" in line:
                try:
                    acc_part = line.split("Final test accuracy:")[1].strip()
                    final_acc = float(acc_part.split()[0])
                except:
                    pass
        
        if process.returncode != 0:
            print(f"Training for Subject {subject} failed with return code {process.returncode}")
            return (subject, 0.0, 0.0, process.returncode)
        
        return (subject, best_acc, final_acc, 0)
        
    except subprocess.TimeoutExpired:
        print(f"Training for Subject {subject} timed out after {timeout} seconds")
        return (subject, 0.0, 0.0, -2)  # -2 return code = timeout
    except Exception as e:
        print(f"Error running training for Subject {subject}: {e}")
        return (subject, 0.0, 0.0, -1)  # -1 return code = error

def main():
    args = parse_args()
    print(f"Starting batch training with arguments: {args}")
    
    # Modified check: Just try to import and run a minimal version
    try:
        print("Testing if modules can be imported...")
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Try importing your custom modules
        try:
            from dataloader import get_dataloader
            print("Successfully imported dataloader")
        except Exception as e:
            print(f"Error importing dataloader: {e}")
        
        try:
            from msvtnet import MSVTNet
            print("Successfully imported MSVTNet")
        except Exception as e:
            print(f"Error importing MSVTNet: {e}")
            
    except Exception as e:
        print(f"Error during module import test: {e}")
    
    # Run with a single subject first as a test
    print("Running test training with subject 1...")
    test_args = argparse.Namespace(**vars(args))
    test_args.epochs = 5  # Just a few epochs for testing
    test_result = run_training(1, test_args)
    print(f"Test training result: {test_result}")
    
    user_input = input("Continue with training all subjects? (y/n): ")
    if user_input.lower() != 'y':
        print("Exiting...")
        sys.exit(0)
    
    # Your existing code for running all subjects
    # ...

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()