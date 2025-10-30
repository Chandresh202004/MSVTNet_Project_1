import os
import numpy as np
import matplotlib.pyplot as plt

def check_and_fix_labels(input_path, output_path=None):
    """Check and optionally fix label encoding issues in preprocessed data."""
    if output_path is None:
        output_path = input_path  # Overwrite existing files
    
    # Check if output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = np.load(input_path)
    X = data['data']
    y = data['labels']
    
    print(f"File: {input_path}")
    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Check label distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Unique labels: {unique_labels}")
    print(f"Label counts: {counts}")
    
    # Visualize a few trials
    plt.figure(figsize=(15, 10))
    for i in range(min(5, len(X))):
        plt.subplot(5, 1, i+1)
        plt.plot(X[i, 0, :])  # Plot first channel
        plt.title(f"Trial {i}, Label: {y[i]}")
    plt.tight_layout()
    plt.savefig(f"{os.path.splitext(input_path)[0]}_samples.png")
    plt.close()
    
    # Check if labels need fixing (if they're not in 0-3 range)
    fixed = False
    if np.min(y) > 0 or np.max(y) > 3:
        print("Labels need fixing (not in 0-3 range)")
        # Map labels to 0-3 range
        if np.min(y) >= 769:  # Raw GDF event codes
            # Map 769, 770, 771, 772 to 0, 1, 2, 3
            new_y = np.array([y_val - 769 for y_val in y])
        elif np.min(y) >= 1 and np.max(y) <= 4:
            # Map 1, 2, 3, 4 to 0, 1, 2, 3
            new_y = y - 1
        else:
            print(f"Unexpected label range: min={np.min(y)}, max={np.max(y)}")
            return False
        
        # Check fixed labels
        unique_new, counts_new = np.unique(new_y, return_counts=True)
        print(f"Fixed labels: {unique_new}")
        print(f"Fixed counts: {counts_new}")
        
        # Save fixed data
        np.savez(output_path, data=X, labels=new_y)
        print(f"Saved fixed data to {output_path}")
        fixed = True
    else:
        print("Labels are already in correct range (0-3)")
    
    # Check for NaN or infinity values
    if np.isnan(X).any() or np.isinf(X).any():
        print("WARNING: Data contains NaN or infinity values!")
        # Count NaN and infinity values
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        print(f"NaN count: {nan_count}")
        print(f"Infinity count: {inf_count}")
        
        # Replace with zeros (optional)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print("Replaced NaN and infinity values with zeros")
        
        # Save fixed data
        np.savez(output_path, data=X, labels=new_y if fixed else y)
        print(f"Saved fixed data to {output_path}")
        fixed = True
    
    # Check data range
    data_min = X.min()
    data_max = X.max()
    data_mean = X.mean()
    data_std = X.std()
    print(f"Data range: min={data_min:.4f}, max={data_max:.4f}, mean={data_mean:.4f}, std={data_std:.4f}")
    
    # If the data range is very small or very large, it might need normalization
    if data_std < 1e-6 or data_std > 1e6:
        print("WARNING: Data has unusual standard deviation!")
    
    return fixed

def process_all_files(data_path):
    """Process all preprocessed files in the given directory."""
    fixed_any = False
    
    # Loop through all subjects and sessions
    for subject in range(1, 10):
        for session in [1, 2]:
            file_path = os.path.join(data_path, f"S{subject:02d}_session_{session}_preprocessed.npz")
            if os.path.exists(file_path):
                print(f"\nChecking Subject {subject}, Session {session}...")
                fixed = check_and_fix_labels(file_path)
                fixed_any = fixed_any or fixed
    
    return fixed_any

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check and fix preprocessed data')
    parser.add_argument('--data_path', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed',
                       help='Path to preprocessed data')
    
    args = parser.parse_args()
    
    fixed = process_all_files(args.data_path)
    
    if fixed:
        print("\nFixed issues in one or more files. Please run training again.")
    else:
        print("\nNo issues found or fixed in the data files.")