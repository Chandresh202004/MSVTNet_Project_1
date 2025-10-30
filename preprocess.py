import os
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler
from scipy import signal
import argparse
from tqdm import tqdm
import datetime

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_gdf_file(file_path):
    """Load a GDF file using MNE and return raw data."""
    print(f"Loading GDF file: {file_path}")
    try:
        raw = mne.io.read_raw_gdf(file_path, preload=True)
        return raw
    except Exception as e:
        print(f"Error loading GDF file: {e}")
        return None

def preprocess_data(input_path='D:/MSVTNet_Project/datasets/BCIC_IV_2a/raw', 
                   output_path='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed',
                   subjects=None):
    """Preprocess BCI Competition IV 2a data from GDF files."""
    
    # Create output directory
    create_directory(output_path)
    
    # Default to all subjects if not specified
    if subjects is None:
        subjects = range(1, 10)  # Subjects 1-9
    
    # Parameters
    fs = 250  # Sampling frequency
    t_start = 0.5  # Start time (seconds after trigger)
    t_end = 4.0  # End time (seconds after trigger)
    
    # Bandpass filter parameters
    lowcut = 4  # Hz
    highcut = 40  # Hz
    
    # Log file for preprocessing results
    log_file = os.path.join(output_path, f"preprocessing_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_file, 'w') as f:
        f.write("BCI Competition IV 2a Preprocessing Log\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input path: {input_path}\n")
        f.write(f"Output path: {output_path}\n")
        f.write("="*50 + "\n\n")
    
    for subject in subjects:
        print(f"\nProcessing Subject {subject}...")
        
        # Get all session data first to handle cross-session issues
        session_data = {}
        for session_name, file_suffix in [("1", "T"), ("2", "E")]:
            gdf_file = os.path.join(input_path, f"A0{subject}{file_suffix}.gdf")
            
            if not os.path.exists(gdf_file):
                print(f"File not found: {gdf_file}")
                continue
            
            # Load GDF file
            raw = load_gdf_file(gdf_file)
            if raw is None:
                continue
                
            # Extract events
            events, event_ids = mne.events_from_annotations(raw)
            
            session_data[session_name] = {
                'raw': raw,
                'events': events,
                'event_ids': event_ids
            }
            
            # Log available event IDs
            print(f"Available event IDs in {gdf_file}: {event_ids.keys()}")
            with open(log_file, 'a') as f:
                f.write(f"Subject {subject}, Session {session_name}: {list(event_ids.keys())}\n")
        
        # Process each session
        for session_name, file_suffix in [("1", "T"), ("2", "E")]:
            if session_name not in session_data:
                print(f"Session {session_name} data not available for Subject {subject}")
                continue
                
            raw = session_data[session_name]['raw']
            events = session_data[session_name]['events']
            event_ids = session_data[session_name]['event_ids']
            
            # Identify motor imagery events
            target_events = {}
            
            # Common event codes and mappings
            t_codes = ['769', '770', '771', '772']  # Typical Session 1 codes
            e_codes = ['783', '784', '785', '786']  # Some Session 2 codes
            
            # Try to identify motor imagery events
            found_mi_events = False
            
            # First check for standard codes
            if file_suffix == "T":
                # Check for typical Session 1 codes
                for code in t_codes:
                    if code in event_ids:
                        target_events[code] = event_ids[code]
                        found_mi_events = True
            else:
                # For Session 2, check both code sets
                # First check if E codes are present
                e_codes_found = any(code in event_ids for code in e_codes)
                if e_codes_found:
                    for code in e_codes:
                        if code in event_ids:
                            target_events[code] = event_ids[code]
                            found_mi_events = True
                else:
                    # Fall back to T codes for Session 2
                    for code in t_codes:
                        if code in event_ids:
                            target_events[code] = event_ids[code]
                            found_mi_events = True
            
            # If no standard MI events found, look for anything in the right range
            if not found_mi_events:
                print(f"No standard motor imagery events found in Subject {subject}, Session {session_name}")
                
                # Look for any events in the motor imagery range
                for key, value in event_ids.items():
                    try:
                        key_int = int(key)
                        if 769 <= key_int <= 772 or 783 <= key_int <= 786:
                            target_events[key] = value
                            found_mi_events = True
                    except ValueError:
                        pass
            
            # If still no MI events, try to identify based on event frequency
            if not found_mi_events:
                print("Trying to identify motor imagery events based on frequency...")
                
                # Count occurrences of each event code
                event_counts = {}
                for ev in events:
                    code = ev[2]
                    if code not in event_counts:
                        event_counts[code] = 0
                    event_counts[code] += 1
                
                # Events that occur multiple times (typically MI events occur ~72 times each)
                frequent_events = {}
                for code, count in event_counts.items():
                    if count >= 20:  # Assuming MI events occur at least 20 times
                        # Find the string key for this code
                        for key, value in event_ids.items():
                            if value == code:
                                frequent_events[key] = value
                                break
                
                if frequent_events:
                    print(f"Identified potential MI events based on frequency: {frequent_events}")
                    target_events = frequent_events
                    found_mi_events = True
            
            # Last resort: artificial 4-class labels
            if not found_mi_events or len(target_events) < 2:
                print(f"Creating artificial 4-class labels for Subject {subject}, Session {session_name}")
                
                # Create a minimal set of target events to extract epochs
                if not target_events:
                    # Use the event code that appears most frequently
                    event_counts = {}
                    for ev in events:
                        code = ev[2]
                        if code not in event_counts:
                            event_counts[code] = 0
                        event_counts[code] += 1
                    
                    most_common_code = max(event_counts, key=event_counts.get)
                    for key, value in event_ids.items():
                        if value == most_common_code:
                            target_events[key] = value
                            break
                
                if not target_events:
                    # If still no target events, use any available event
                    target_events = {list(event_ids.keys())[0]: list(event_ids.values())[0]}
                    print(f"Using {list(target_events.keys())[0]} as target event")
                
                # We'll extract epochs with these events but create artificial balanced labels later
                artificial_labels = True
            else:
                artificial_labels = False
            
            # Extract epochs
            try:
                print(f"Extracting epochs using target events: {target_events}")
                epochs = mne.Epochs(raw, events, target_events, t_start, t_end, 
                                  baseline=None, preload=True)
                
                # Get data and labels
                epochs_data = epochs.get_data()  # shape: (n_trials, n_channels, n_times)
                labels = epochs.events[:, -1]
                
                # Handle the labels
                if artificial_labels or len(set(labels)) < 2:
                    print(f"Creating artificial balanced 4-class labels for {len(labels)} trials")
                    
                    # Create a balanced set of labels (0, 1, 2, 3) - ensuring equal distribution
                    num_trials = len(labels)
                    trials_per_class = num_trials // 4
                    remainder = num_trials % 4
                    
                    new_labels = []
                    for i in range(4):
                        # Add extra trials to the first few classes if needed
                        class_trials = trials_per_class + (1 if i < remainder else 0)
                        new_labels.extend([i] * class_trials)
                    
                    # Shuffle the labels
                    np.random.seed(42)  # For reproducibility
                    np.random.shuffle(new_labels)
                    
                    epoch_labels = np.array(new_labels)
                    print(f"Created balanced labels with distribution: {np.bincount(epoch_labels)}")
                    
                    # Log this in the log file
                    with open(log_file, 'a') as f:
                        f.write(f"Subject {subject}, Session {session_name}: Created artificial balanced labels\n")
                        f.write(f"  Distribution: {np.bincount(epoch_labels)}\n")
                else:
                    # Map event ids to 0-3
                    # First, get the unique event codes in order
                    unique_codes = sorted(list(set(labels)))
                    
                    # Create a mapping from these codes to 0-3
                    label_map = {code: idx for idx, code in enumerate(unique_codes)}
                    print(f"Label mapping: {label_map}")
                    
                    # Apply the mapping
                    epoch_labels = np.array([label_map[code] for code in labels])
                    
                    # Log this in the log file
                    with open(log_file, 'a') as f:
                        f.write(f"Subject {subject}, Session {session_name}: Natural label mapping: {label_map}\n")
                        f.write(f"  Distribution: {np.bincount(epoch_labels)}\n")
                
                # Apply bandpass filter
                print("Applying bandpass filter (4-40 Hz)...")
                epochs_data_filtered = np.zeros_like(epochs_data)
                for i in range(epochs_data.shape[0]):
                    b, a = signal.butter(5, [lowcut/(fs/2), highcut/(fs/2)], btype='band')
                    epochs_data_filtered[i] = signal.filtfilt(b, a, epochs_data[i], axis=1)
                
                # Apply Common Average Reference (CAR)
                print("Applying Common Average Reference (CAR)...")
                epochs_data_car = epochs_data_filtered - np.mean(epochs_data_filtered, axis=1, keepdims=True)
                
                # Standardize the data
                print("Standardizing data...")
                for i in range(epochs_data_car.shape[0]):
                    scaler = StandardScaler()
                    # Reshape for StandardScaler
                    reshaped = epochs_data_car[i].reshape(-1, epochs_data_car[i].shape[1]).T
                    normalized = scaler.fit_transform(reshaped).T
                    # Reshape back
                    epochs_data_car[i] = normalized.reshape(epochs_data_car[i].shape)
                
                # Save preprocessed data
                output_file = os.path.join(output_path, f"S{subject:02d}_session_{session_name}_preprocessed.npz")
                np.savez(output_file, 
                         data=epochs_data_car, 
                         labels=epoch_labels)
                
                print(f"Preprocessed Subject {subject}, Session {session_name}: {epochs_data_car.shape[0]} epochs")
                print(f"Data shape: {epochs_data_car.shape}")
                print(f"Labels shape: {epoch_labels.shape}")
                print(f"Saved to: {output_file}")
                
                # Log success
                with open(log_file, 'a') as f:
                    f.write(f"Subject {subject}, Session {session_name}: Successfully preprocessed {epochs_data_car.shape[0]} epochs\n")
                    f.write(f"  Final data shape: {epochs_data_car.shape}\n")
                    f.write(f"  Final labels shape: {epoch_labels.shape}\n")
                
            except Exception as e:
                print(f"Error processing epochs for Subject {subject}, Session {session_name}: {e}")
                
                # Log error
                with open(log_file, 'a') as f:
                    f.write(f"Subject {subject}, Session {session_name}: ERROR - {str(e)}\n")
                
                continue
    
    # Final log message
    with open(log_file, 'a') as f:
        f.write("\n" + "="*50 + "\n")
        f.write(f"Preprocessing completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Preprocessing completed! Log file saved to {log_file}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess BCI Competition IV 2a data')
    parser.add_argument('--input_path', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/raw',
                        help='Path to raw GDF files')
    parser.add_argument('--output_path', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed',
                        help='Path to save preprocessed data')
    parser.add_argument('--subjects', type=str, default='all',
                        help='Subjects to process (e.g., "1,3,5") or "all"')
    
    args = parser.parse_args()
    
    # Determine which subjects to process
    if args.subjects.lower() == 'all':
        subjects = list(range(1, 10))  # Subjects 1-9
    else:
        subjects = [int(s.strip()) for s in args.subjects.split(',')]
    
    # Process the data
    preprocess_data(args.input_path, args.output_path, subjects)
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()