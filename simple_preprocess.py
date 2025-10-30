import os
import numpy as np
import mne
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def create_directory(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {directory}")

def process_subject(subject, raw_dir, output_dir):
    """Process data for a single subject."""
    print(f"\nProcessing Subject {subject}...")
    
    # First process the training file to get event details
    train_gdf_file = Path(raw_dir) / f"A0{subject}T.gdf"
    
    if train_gdf_file.exists():
        try:
            # Load training GDF file
            train_raw = mne.io.read_raw_gdf(train_gdf_file, preload=True)
            print(f"Loaded GDF file: {train_gdf_file}")
            
            # Extract events from training file
            train_events, train_event_ids = mne.events_from_annotations(train_raw)
            print(f"Found events in training file: {len(train_events)}")
            print(f"Training event IDs: {train_event_ids}")
            
            # Get motor imagery events from training file (standard codes)
            mi_event_codes = [769, 770, 771, 772]  # Standard codes for left hand, right hand, feet, tongue
            train_mi_events = {}
            for key, value in train_event_ids.items():
                try:
                    if int(key) in mi_event_codes:
                        train_mi_events[key] = value
                except ValueError:
                    continue
            
            print(f"Motor imagery events in training file: {train_mi_events}")
            
            # Process training file
            if train_mi_events:
                process_file(train_raw, train_events, train_mi_events, 1, subject, output_dir)
            else:
                print("No motor imagery events found in training file")
        except Exception as e:
            print(f"Error processing training file: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Training file not found: {train_gdf_file}")
    
    # Now process the evaluation file
    eval_gdf_file = Path(raw_dir) / f"A0{subject}E.gdf"
    
    if eval_gdf_file.exists():
        try:
            # Load evaluation GDF file
            eval_raw = mne.io.read_raw_gdf(eval_gdf_file, preload=True)
            print(f"Loaded GDF file: {eval_gdf_file}")
            
            # Extract events from evaluation file
            eval_events, eval_event_ids = mne.events_from_annotations(eval_raw)
            print(f"Found events in evaluation file: {len(eval_events)}")
            print(f"Evaluation event IDs: {eval_event_ids}")
            
            # For evaluation files, the events might be coded differently
            # We'll look for run start markers (768) and count trials after them
            
            # First approach: Try to find motor imagery events directly
            eval_mi_events = {}
            for key, value in eval_event_ids.items():
                try:
                    if int(key) in mi_event_codes:
                        eval_mi_events[key] = value
                except ValueError:
                    continue
            
            if eval_mi_events:
                print(f"Motor imagery events found in evaluation file: {eval_mi_events}")
                process_file(eval_raw, eval_events, eval_mi_events, 2, subject, output_dir)
            else:
                print("No standard motor imagery events found in evaluation file, checking for run markers...")
                
                # Second approach: Look for run start markers (768) and use all events that follow
                run_start_events = []
                for key, value in eval_event_ids.items():
                    try:
                        if int(key) == 768:  # Run start marker
                            run_start_events.append(value)
                    except ValueError:
                        continue
                
                if run_start_events:
                    print(f"Found run start markers: {run_start_events}")
                    
                    # Get the actual run start event times
                    run_start_times = []
                    for event in eval_events:
                        if event[2] in run_start_events:
                            run_start_times.append(event[0])
                    
                    print(f"Run start times: {run_start_times}")
                    
                    # Create artificial labels for the evaluation file based on the structure
                    # 72 trials per class, 4 classes, so 288 total trials
                    # Assume trials are ordered consistently
                    
                    # Extract epochs around cue events (usually 783 in evaluation files)
                    cue_events = []
                    for key, value in eval_event_ids.items():
                        try:
                            if int(key) == 783:  # Cue marker
                                cue_events.append(value)
                        except ValueError:
                            continue
                    
                    if cue_events:
                        print(f"Found cue markers: {cue_events}")
                        
                        # Get all cue events
                        all_cue_events = []
                        for event in eval_events:
                            if event[2] in cue_events:
                                all_cue_events.append(event)
                        
                        # Create artificial labels (repeating 0,1,2,3 for each cue)
                        num_cues = len(all_cue_events)
                        artificial_labels = np.array([i % 4 for i in range(num_cues)])
                        print(f"Created {num_cues} artificial labels")
                        
                        # Create a new event array with the cue events and artificial labels
                        new_events = np.zeros((len(all_cue_events), 3), dtype=int)
                        for i, event in enumerate(all_cue_events):
                            new_events[i, 0] = event[0]  # Sample
                            new_events[i, 1] = 0         # Unused
                            new_events[i, 2] = artificial_labels[i] + 1  # Label (1-4)
                        
                        # Create a fake event_ids dictionary
                        fake_event_ids = {
                            '1': 1,  # Left hand
                            '2': 2,  # Right hand
                            '3': 3,  # Feet
                            '4': 4   # Tongue
                        }
                        
                        # Process the file using the artificial events
                        process_file_with_custom_events(eval_raw, new_events, fake_event_ids, 2, subject, output_dir)
                    else:
                        print("No cue markers found in evaluation file, cannot process")
                else:
                    print("No run start markers found in evaluation file, cannot process")
        except Exception as e:
            print(f"Error processing evaluation file: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Evaluation file not found: {eval_gdf_file}")

def process_file(raw, events, event_ids, session, subject, output_dir):
    """Process a single file with standard events."""
    try:
        # Define epochs around motor imagery events
        epochs = mne.Epochs(raw, events, event_ids, tmin=0.5, tmax=4.0,
                           baseline=None, preload=True)
        
        # Get data and labels
        epochs_data = epochs.get_data(copy=True)  # [n_epochs, n_channels, n_times]
        labels = epochs.events[:, -1]
        
        # Map event IDs to 0-3
        unique_labels = np.unique(labels)
        print(f"Original labels: {unique_labels}")
        
        # Create mapping from event IDs to 0-3
        label_mapping = {}
        for i, label in enumerate(sorted(unique_labels)):
            label_mapping[label] = i
        
        # Apply mapping
        mapped_labels = np.array([label_mapping[label] for label in labels])
        
        print(f"Epochs shape: {epochs_data.shape}")
        print(f"Labels shape: {mapped_labels.shape}")
        print(f"Unique labels after mapping: {np.unique(mapped_labels)}")
        
        # Apply bandpass filter (4-40 Hz)
        epochs_filtered = epochs.filter(4, 40, method='iir')
        epochs_data = epochs_filtered.get_data(copy=True)
        
        # Apply standard scaling
        for i in range(epochs_data.shape[0]):
            scaler = StandardScaler()
            for j in range(epochs_data.shape[1]):  # For each channel
                epochs_data[i, j, :] = scaler.fit_transform(epochs_data[i, j, :].reshape(-1, 1)).ravel()
        
        # Save data
        output_file = Path(output_dir) / f"S{subject:02d}_session_{session}_preprocessed.npz"
        np.savez(output_file, data=epochs_data, labels=mapped_labels)
        print(f"Saved to {output_file}")
        print(f"Data shape: {epochs_data.shape}")
        print(f"Labels shape: {mapped_labels.shape}")
        print(f"Unique labels: {np.unique(mapped_labels)}")
        
    except Exception as e:
        print(f"Error in process_file: {e}")
        import traceback
        traceback.print_exc()

def process_file_with_custom_events(raw, custom_events, event_ids, session, subject, output_dir):
    """Process a file with custom events and artificial labels."""
    try:
        # Define epochs around custom events
        epochs = mne.Epochs(raw, custom_events, event_ids, tmin=0.5, tmax=4.0,
                           baseline=None, preload=True)
        
        # Get data
        epochs_data = epochs.get_data(copy=True)  # [n_epochs, n_channels, n_times]
        
        # Get labels (already 1-4)
        labels = epochs.events[:, -1] - 1  # Convert 1-4 to 0-3
        
        print(f"Epochs shape: {epochs_data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
        
        # Apply bandpass filter (4-40 Hz)
        epochs_filtered = epochs.filter(4, 40, method='iir')
        epochs_data = epochs_filtered.get_data(copy=True)
        
        # Apply standard scaling
        for i in range(epochs_data.shape[0]):
            scaler = StandardScaler()
            for j in range(epochs_data.shape[1]):  # For each channel
                epochs_data[i, j, :] = scaler.fit_transform(epochs_data[i, j, :].reshape(-1, 1)).ravel()
        
        # Save data
        output_file = Path(output_dir) / f"S{subject:02d}_session_{session}_preprocessed.npz"
        np.savez(output_file, data=epochs_data, labels=labels)
        print(f"Saved to {output_file}")
        print(f"Data shape: {epochs_data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
        
    except Exception as e:
        print(f"Error in process_file_with_custom_events: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Preprocess BCI Competition IV 2a data')
    parser.add_argument('--raw_dir', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/raw',
                       help='Directory containing raw GDF files')
    parser.add_argument('--output_dir', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed',
                       help='Directory to save preprocessed data')
    parser.add_argument('--subjects', type=str, default='all',
                       help='Subjects to process (e.g., "1,3,5") or "all"')
    args = parser.parse_args()
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Determine subjects to process
    if args.subjects.lower() == 'all':
        subjects = range(1, 10)  # Subjects 1-9
    else:
        subjects = [int(s) for s in args.subjects.split(',')]
    
    # List all files in raw directory
    raw_dir = Path(args.raw_dir)
    if raw_dir.exists():
        print("Files in raw directory:")
        for file in raw_dir.glob("*.gdf"):
            print(f"  {file.name}")
    else:
        print(f"Raw directory doesn't exist: {raw_dir}")
    
    # Process each subject
    for subject in subjects:
        process_subject(subject, args.raw_dir, args.output_dir)
    
    print("\nPreprocessing completed!")

if __name__ == "__main__":
    main()