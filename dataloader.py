import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob

class BCIDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def augment_eeg(eeg_data, labels):
    """Apply data augmentation to EEG signals"""
    augmented_data = []
    augmented_labels = []
    
    for i in range(len(eeg_data)):
        signal = eeg_data[i]
        label = labels[i]
        
        # Original data
        augmented_data.append(signal)
        augmented_labels.append(label)
        
        # Augmentation 1: Gaussian noise
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.01, 0.05)
            noisy_signal = signal + noise_level * np.random.normal(0, 1, size=signal.shape)
            augmented_data.append(noisy_signal)
            augmented_labels.append(label)
        
        # Augmentation 2: Temporal shifting
        if np.random.random() < 0.5:
            shift = np.random.randint(-10, 10)
            shifted_signal = np.roll(signal, shift, axis=1)
            augmented_data.append(shifted_signal)
            augmented_labels.append(label)
    
    return np.array(augmented_data), np.array(augmented_labels)

def get_dataloader(data_dir, subject, batch_size=32, dataset_type="2a", session_dependent=True, 
                  train_session=1, test_session=2, num_workers=0, augment=False):
    """Create data loaders for BCI Competition IV 2a or 2b datasets
    
    Parameters:
    -----------
    data_dir: str
        Directory containing preprocessed data
    subject: int
        Subject ID (1-9)
    batch_size: int
        Batch size for dataloaders
    dataset_type: str
        Either "2a" (4-class motor imagery) or "2b" (2-class motor imagery)
    session_dependent: bool
        Whether to use session-dependent training
    train_session: int
        Session to use for training in session-dependent mode
    test_session: int
        Session to use for testing in session-dependent mode
    num_workers: int
        Number of worker processes for data loading
    augment: bool
        Whether to apply data augmentation
    """
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory does not exist: {data_dir}")
        print("Please check the path and create the directory if needed.")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Look for existing files to determine naming pattern
    print(f"Looking for {dataset_type} data files in: {data_dir}")
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    print(f"Found {len(files)} .npz files")
    
    if len(files) == 0:
        print("ERROR: No .npz files found in the data directory.")
        raise FileNotFoundError("No .npz files found in the data directory")
    
    # Handle different file naming conventions for 2a and 2b datasets
    if dataset_type == "2b":
        # Format for 2b dataset, typically has 5 sessions per subject
        if train_session > 3 or test_session > 3:
            print("WARNING: BCI IV 2b typically uses sessions 1-3, check your session numbers")
        
        # For subjects 1-9, add leading zero
        if subject < 10:
            session1_file = os.path.join(data_dir, f'B0{subject}0{train_session}_preprocessed.npz')
            session2_file = os.path.join(data_dir, f'B0{subject}0{test_session}_preprocessed.npz')
        else:
            session1_file = os.path.join(data_dir, f'B{subject}0{train_session}_preprocessed.npz')
            session2_file = os.path.join(data_dir, f'B{subject}0{test_session}_preprocessed.npz')
    else:  # Default to 2a format
        # For subjects 1-9, add leading zero
        if subject < 10:
            session1_file = os.path.join(data_dir, f'S0{subject}_session_1_preprocessed.npz')
            session2_file = os.path.join(data_dir, f'S0{subject}_session_2_preprocessed.npz')
        else:
            session1_file = os.path.join(data_dir, f'S{subject}_session_1_preprocessed.npz')
            session2_file = os.path.join(data_dir, f'S{subject}_session_2_preprocessed.npz')
    
    print(f"Looking for session files with {dataset_type} format:")
    print(f"  - Training session file: {session1_file}")
    print(f"  - Testing session file: {session2_file}")
    
    # Try alternative naming patterns if files not found
    if not os.path.exists(session1_file) or not os.path.exists(session2_file):
        print("WARNING: Files not found with standard naming. Trying alternative formats...")
        
        # Try common alternative naming patterns
        if dataset_type == "2b":
            alternatives = [
                (f'B0{subject}0{train_session}_preprocessed.npz', f'B0{subject}0{test_session}_preprocessed.npz'),
                (f'subject_{subject}_session_{train_session}_preprocessed.npz', f'subject_{subject}_session_{test_session}_preprocessed.npz'),
                (f'sub_{subject}_sess_{train_session}.npz', f'sub_{subject}_sess_{test_session}.npz')
            ]
        else:  # 2a
            alternatives = [
                (f'A0{subject}0{train_session}_preprocessed.npz', f'A0{subject}0{test_session}_preprocessed.npz'),
                (f'subject_{subject}_session_{train_session}_preprocessed.npz', f'subject_{subject}_session_{test_session}_preprocessed.npz'),
                (f'sub_{subject}_sess_{train_session}.npz', f'sub_{subject}_sess_{test_session}.npz')
            ]
        
        # Try each alternative pattern
        for train_pattern, test_pattern in alternatives:
            alt_train = os.path.join(data_dir, train_pattern)
            alt_test = os.path.join(data_dir, test_pattern)
            
            if os.path.exists(alt_train) and os.path.exists(alt_test):
                print(f"Found files with alternative naming pattern:")
                print(f"  - {alt_train}")
                print(f"  - {alt_test}")
                session1_file = alt_train
                session2_file = alt_test
                break
        
        # If we still haven't found the files, list available files and raise error
        if not os.path.exists(session1_file) or not os.path.exists(session2_file):
            print("ERROR: Could not find the required data files.")
            print("Available files:")
            for f in files[:10]:  # Print first 10 files
                print(f"  - {os.path.basename(f)}")
            raise FileNotFoundError("Required data files not found")
    
    # Load .npz files (which contain multiple arrays)
    session1_data = np.load(session1_file)
    session2_data = np.load(session2_file)
    
    # Print available keys in the .npz files
    print(f"Session {train_session} file contains keys: {list(session1_data.keys())}")
    print(f"Session {test_session} file contains keys: {list(session2_data.keys())}")
    
    # Extract data and labels - adjust these keys based on your actual .npz structure
    try:
        # Attempt to extract using common key patterns
        if 'data' in session1_data.keys() and 'labels' in session1_data.keys():
            session1_X = session1_data['data']
            session1_y = session1_data['labels']
            session2_X = session2_data['data']
            session2_y = session2_data['labels']
        elif 'X' in session1_data.keys() and 'y' in session1_data.keys():
            session1_X = session1_data['X']
            session1_y = session1_data['y']
            session2_X = session2_data['X']
            session2_y = session2_data['y']
        elif 'signals' in session1_data.keys() and 'labels' in session1_data.keys():
            session1_X = session1_data['signals']
            session1_y = session1_data['labels']
            session2_X = session2_data['signals']
            session2_y = session2_data['labels']
        else:
            # If standard keys not found, use the first two arrays
            keys = list(session1_data.keys())
            session1_X = session1_data[keys[0]]
            session1_y = session1_data[keys[1]]
            session2_X = session2_data[keys[0]]
            session2_y = session2_data[keys[1]]
    except Exception as e:
        print(f"ERROR extracting data from .npz files: {e}")
        print("Please provide the correct keys for data and labels in your .npz files")
        raise
    
    # Print data shapes for debugging
    print(f"Session {train_session} data shape: {session1_X.shape}")
    print(f"Session {train_session} labels shape: {session1_y.shape}")
    print(f"Session {test_session} data shape: {session2_X.shape}")
    print(f"Session {test_session} labels shape: {session2_y.shape}")
    
    # Verify class labels are appropriate for the dataset type
    num_classes = 4 if dataset_type == "2a" else 2
    unique_labels = np.unique(np.concatenate([session1_y, session2_y]))
    print(f"Unique class labels found: {unique_labels}")
    if max(unique_labels) >= num_classes:
        print(f"WARNING: Found labels up to {max(unique_labels)} but dataset {dataset_type} should have {num_classes} classes")
        print("Class labels might need remapping")
    
    # For 2b dataset, ensure labels are 0 and 1
    if dataset_type == "2b" and (0 not in unique_labels or 1 not in unique_labels):
        print("Remapping 2b dataset labels to [0,1]...")
        # Map lowest value to 0 and highest to 1
        label_map = {min(unique_labels): 0, max(unique_labels): 1}
        session1_y = np.array([label_map[y] for y in session1_y])
        session2_y = np.array([label_map[y] for y in session2_y])
    
    # Ensure labels are integers
    session1_y = session1_y.astype(np.int64)
    session2_y = session2_y.astype(np.int64)
    
    if session_dependent:
        # Session-dependent: train on one session, test on another
        train_data, train_labels = session1_X, session1_y
        test_data, test_labels = session2_X, session2_y
        
        # Apply augmentation if requested
        if augment:
            print("Applying data augmentation to training data...")
            train_data, train_labels = augment_eeg(train_data, train_labels)
    else:
        # Session-independent: combine data from both sessions and use cross-validation
        all_data = np.concatenate([session1_X, session2_X], axis=0)
        all_labels = np.concatenate([session1_y, session2_y], axis=0)
        
        # Use random 80-20 split for training and testing
        np.random.seed(42)  # For reproducibility
        indices = np.random.permutation(len(all_labels))
        split = int(0.8 * len(indices))
        
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        train_data = all_data[train_indices]
        train_labels = all_labels[train_indices]
        test_data = all_data[test_indices]
        test_labels = all_labels[test_indices]
        
        # Apply augmentation if requested
        if augment:
            print("Applying data augmentation to training data...")
            train_data, train_labels = augment_eeg(train_data, train_labels)
    
    print(f"Final training data shape: {train_data.shape}")
    print(f"Final testing data shape: {test_data.shape}")
    
    # Convert to torch tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Create datasets
    train_dataset = BCIDataset(train_data, train_labels)
    test_dataset = BCIDataset(test_data, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader