import os
import numpy as np
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='Check BCI Competition IV 2a data files')
    parser.add_argument('--data_dir', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed', help='Path to the preprocessed dataset')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory does not exist: {args.data_dir}")
        return
    
    files = glob.glob(os.path.join(args.data_dir, "*.npz"))
    print(f"Found {len(files)} .npz files in {args.data_dir}")
    
    for f in files:
        print(f"File: {os.path.basename(f)}")
    
    if files:
        print(f"\nTrying to load the first file: {files[0]}")
        try:
            data = np.load(files[0])
            print(f"Success! Available keys in the file: {list(data.keys())}")
            
            # Try to access each array in the file
            for key in data.keys():
                print(f"  - {key}: shape {data[key].shape}, dtype {data[key].dtype}")
            
        except Exception as e:
            print(f"Error loading file: {e}")

if __name__ == "__main__":
    main()
