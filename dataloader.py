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
    augmented_data = []
    augmented_labels = []
    
    for i in range(len(eeg_data)):
        signal = eeg_data[i]
        label = labels[i]
        
        augmented_data.append(signal)
        augmented_labels.append(label)
        
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.01, 0.05)
            noisy_signal = signal + noise_level * np.random.normal(0, 1, size=signal.shape)
            augmented_data.append(noisy_signal)
            augmented_labels.append(label)
        
        if np.random.random() < 0.5:
            shift = np.random.randint(-10, 10)
            shifted_signal = np.roll(signal, shift, axis=1)
            augmented_data.append(shifted_signal)
            augmented_labels.append(label)
    
    return np.array(augmented_data), np.array(augmented_labels)

def get_dataloader(
    data_dir,
    subject,
    batch_size=32,
    dataset_type="2a",
    session_dependent=True,
    train_session=1,
    test_session=2,
    num_workers=0,
    augment=False
):

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    if len(files) == 0:
        raise FileNotFoundError("No .npz files found in the data directory")
    
    if dataset_type == "2b":
        if subject < 10:
            session1_file = os.path.join(
                data_dir,
                f'B0{subject}0{train_session}_preprocessed.npz'
            )
            session2_file = os.path.join(
                data_dir,
                f'B0{subject}0{test_session}_preprocessed.npz'
            )
        else:
            session1_file = os.path.join(
                data_dir,
                f'B{subject}0{train_session}_preprocessed.npz'
            )
            session2_file = os.path.join(
                data_dir,
                f'B{subject}0{test_session}_preprocessed.npz'
            )
    else:
        if subject < 10:
            session1_file = os.path.join(
                data_dir,
                f'S0{subject}_session_1_preprocessed.npz'
            )
            session2_file = os.path.join(
                data_dir,
                f'S0{subject}_session_2_preprocessed.npz'
            )
        else:
            session1_file = os.path.join(
                data_dir,
                f'S{subject}_session_1_preprocessed.npz'
            )
            session2_file = os.path.join(
                data_dir,
                f'S{subject}_session_2_preprocessed.npz'
            )
    
    if not os.path.exists(session1_file) or not os.path.exists(session2_file):

        if dataset_type == "2b":
            alternatives = [
                (
                    f'B0{subject}0{train_session}_preprocessed.npz',
                    f'B0{subject}0{test_session}_preprocessed.npz'
                ),
                (
                    f'subject_{subject}_session_{train_session}_preprocessed.npz',
                    f'subject_{subject}_session_{test_session}_preprocessed.npz'
                ),
                (
                    f'sub_{subject}_sess_{train_session}.npz',
                    f'sub_{subject}_sess_{test_session}.npz'
                )
            ]
        else:
            alternatives = [
                (
                    f'A0{subject}0{train_session}_preprocessed.npz',
                    f'A0{subject}0{test_session}_preprocessed.npz'
                ),
                (
                    f'subject_{subject}_session_{train_session}_preprocessed.npz',
                    f'subject_{subject}_session_{test_session}_preprocessed.npz'
                ),
                (
                    f'sub_{subject}_sess_{train_session}.npz',
                    f'sub_{subject}_sess_{test_session}.npz'
                )
            ]
        
        for train_pattern, test_pattern in alternatives:
            alt_train = os.path.join(data_dir, train_pattern)
            alt_test = os.path.join(data_dir, test_pattern)
            
            if os.path.exists(alt_train) and os.path.exists(alt_test):
                session1_file = alt_train
                session2_file = alt_test
                break
        
        if not os.path.exists(session1_file) or not os.path.exists(session2_file):
            raise FileNotFoundError("Required data files not found")
    
    session1_data = np.load(session1_file)
    session2_data = np.load(session2_file)
    
    try:
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
            keys = list(session1_data.keys())

            session1_X = session1_data[keys[0]]
            session1_y = session1_data[keys[1]]

            session2_X = session2_data[keys[0]]
            session2_y = session2_data[keys[1]]

    except Exception as e:
        raise RuntimeError(f"Error extracting data from .npz files: {e}")
    
    num_classes = 4 if dataset_type == "2a" else 2

    unique_labels = np.unique(
        np.concatenate([session1_y, session2_y])
    )

    if max(unique_labels) >= num_classes:
        pass
    
    if dataset_type == "2b" and (0 not in unique_labels or 1 not in unique_labels):
        label_map = {
            min(unique_labels): 0,
            max(unique_labels): 1
        }

        session1_y = np.array([label_map[y] for y in session1_y])
        session2_y = np.array([label_map[y] for y in session2_y])
    
    session1_y = session1_y.astype(np.int64)
    session2_y = session2_y.astype(np.int64)
    
    if session_dependent:

        train_data = session1_X
        train_labels = session1_y

        test_data = session2_X
        test_labels = session2_y
        
        if augment:
            train_data, train_labels = augment_eeg(
                train_data,
                train_labels
            )

    else:
        all_data = np.concatenate(
            [session1_X, session2_X],
            axis=0
        )

        all_labels = np.concatenate(
            [session1_y, session2_y],
            axis=0
        )
        
        np.random.seed(42)

        indices = np.random.permutation(len(all_labels))

        split = int(0.8 * len(indices))
        
        train_indices = indices[:split]
        test_indices = indices[split:]
        
        train_data = all_data[train_indices]
        train_labels = all_labels[train_indices]

        test_data = all_data[test_indices]
        test_labels = all_labels[test_indices]
        
        if augment:
            train_data, train_labels = augment_eeg(
                train_data,
                train_labels
            )
    
    train_data = torch.tensor(
        train_data,
        dtype=torch.float32
    )

    train_labels = torch.tensor(
        train_labels,
        dtype=torch.long
    )

    test_data = torch.tensor(
        test_data,
        dtype=torch.float32
    )

    test_labels = torch.tensor(
        test_labels,
        dtype=torch.long
    )
    
    train_dataset = BCIDataset(
        train_data,
        train_labels
    )

    test_dataset = BCIDataset(
        test_data,
        test_labels
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader
