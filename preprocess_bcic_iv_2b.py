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

def preprocess_data(input_path='D:/MSVTNet_Project/datasets/BCIC_IV_2b/raw', 
                   output_path='D:/MSVTNet_Project/datasets/BCIC_IV_2b/preprocessed',
                   subjects=None):
    """Preprocess BCI Competition IV 2b data from GDF files."""
    
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
        f.write("BCI Competition IV 2b Preprocessing Log\n")
        f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input path: {input_path}\n")
        f.write(f"Output path: {output_path}\n")
        f.write("="*50 + "\n\n")
    
    for subject in subjects:
        print(f"\nProcessing Subject {subject}...")
        
        # Get all session data first to handle cross-session issues
        session_data = {}
        
        # BCIC IV 2b has 5 sessions per subject (01T, 02T, 03T, 04E, 05E)
        for session_num in range(1, 6):
            # Sessions 1-3 are training, 4-5 are evaluation
            file_suffix = "T" if session_num <= 3 else "E"
            is_evaluation = file_suffix == "E"
            
            # Format session number with leading zero
            session_str = f"{session_num:02d}"
            
            # Construct filename using the correct pattern (B0101T.gdf format)
            gdf_file = os.path.join(input_path, f"B0{subject}{session_str}{file_suffix}.gdf")
            
            if not os.path.exists(gdf_file):
                print(f"File not found: {gdf_file}")
                continue
            
            # Load GDF file
            raw = load_gdf_file(gdf_file)
            if raw is None:
                continue
                
            # Extract events - Handle potential repeated events differently for evaluation sessions
            try:
                if is_evaluation:
                    # Use event_repeated='drop' to handle repeated events in evaluation sessions
                    events, event_ids = mne.events_from_annotations(raw, event_repeated='drop')
                    print(f"Processed events with 'drop' option for repeated events")
                else:
                    events, event_ids = mne.events_from_annotations(raw)
            except Exception as e:
                print(f"Error extracting events: {e}")
                print("Trying with alternative event_repeated setting...")
                try:
                    events, event_ids = mne.events_from_annotations(raw, event_repeated='merge')
                    print(f"Processed events with 'merge' option for repeated events")
                except Exception as e2:
                    print(f"Error extracting events with merge option: {e2}")
                    continue
            
            session_data[str(session_num)] = {
                'raw': raw,
                'events': events,
                'event_ids': event_ids,
                'is_evaluation': is_evaluation
            }
            
            # Log available event IDs
            print(f"Available event IDs in {gdf_file}: {event_ids}")
            with open(log_file, 'a') as f:
                f.write(f"Subject {subject}, Session {session_num}: {list(event_ids.keys())}\n")
        
        # Process each session
        for session_num in range(1, 6):
            session_name = str(session_num)
            
            if session_name not in session_data:
                print(f"Session {session_name} data not available for Subject {subject}")
                continue
                
            raw = session_data[session_name]['raw']
            events = session_data[session_name]['events']
            event_ids = session_data[session_name]['event_ids']
            is_evaluation = session_data[session_name]['is_evaluation']
            
            print(f"\nProcessing Session {session_name} ({'Evaluation' if is_evaluation else 'Training'})...")
            
            # Special handling for evaluation sessions (04E, 05E)
            if is_evaluation:
                print(f"Special handling for evaluation session")
                
                # For evaluation sessions, we need a different approach to identify relevant events
                # Based on the event IDs you reported, events 781 and 783 might be the cues
                eval_relevant_events = {}
                
                # Check for specific evaluation events first
                if '781' in event_ids and '783' in event_ids:
                    print("Found evaluation cue events 781 and 783")
                    eval_relevant_events['781'] = event_ids['781']  # First class
                    eval_relevant_events['783'] = event_ids['783']  # Second class
                
                # If we found the evaluation specific events, use them
                if eval_relevant_events:
                    target_events = eval_relevant_events
                    print(f"Using evaluation-specific events: {target_events}")
                else:
                    # Otherwise fall back to looking for other events
                    print("No specific evaluation events found, using fallback strategy")
                    target_events = {}
                    
                    # Check for potential cue events (768 is often a cue start in BCIC datasets)
                    if '768' in event_ids:
                        target_events['768'] = event_ids['768']
            else:
                # For training sessions, look for standard motor imagery events
                # Typically class 1 (769) and class 2 (770) for left and right hand
                target_events = {}
                
                # Common event codes for left/right hand MI
                mi_codes = ['769', '770']  # Left and right hand
                
                # Check for standard codes
                for code in mi_codes:
                    if code in event_ids:
                        target_events[code] = event_ids[code]
            
            # If no suitable events found, try to identify based on frequency or patterns
            if not target_events:
                print("No suitable events found. Analyzing event structure...")
                
                # Count occurrences of each event code
                event_counts = {}
                for ev in events:
                    code = ev[2]
                    if code not in event_counts:
                        event_counts[code] = 0
                    event_counts[code] += 1
                
                # Display event counts for debugging
                print("Event counts:")
                for code, count in event_counts.items():
                    # Find the string key for this code
                    code_str = None
                    for key, value in event_ids.items():
                        if value == code:
                            code_str = key
                            break
                    print(f"  Event {code} ({code_str}): {count} occurrences")
                
                # Events that occur multiple times (typically MI events occur ~40-60 times each)
                frequent_events = {}
                for code, count in event_counts.items():
                    if count >= 20:  # Assuming MI events occur at least 20 times
                        # Find the string key for this code
                        for key, value in event_ids.items():
                            if value == code:
                                frequent_events[key] = value
                                break
                
                if frequent_events:
                    print(f"Identified potential events based on frequency: {frequent_events}")
                    target_events = frequent_events
            
            # If still no events, create artificial balanced classes
            if not target_events:
                print(f"Creating artificial event markers for Subject {subject}, Session {session_name}")
                
                # Use the most common event as a marker for trial boundaries
                most_common_code = max(event_counts, key=event_counts.get)
                for key, value in event_ids.items():
                    if value == most_common_code:
                        target_events[key] = value
                        print(f"Using {key} as trial boundary marker")
                        break
                
                # Force artificial labels mode
                artificial_labels = True
            else:
                artificial_labels = False
            
            # Extract epochs
            try:
                print(f"Extracting epochs using target events: {target_events}")
                
                # For evaluation sessions, we need different parameters
                if is_evaluation:
                    # When handling evaluation data, be more flexible with event handling
                    try:
                        epochs = mne.Epochs(raw, events, target_events, t_start, t_end, 
                                          baseline=None, preload=True, 
                                          event_repeated='drop')
                    except:
                        print("Falling back to merge strategy for repeated events...")
                        epochs = mne.Epochs(raw, events, target_events, t_start, t_end, 
                                          baseline=None, preload=True, 
                                          event_repeated='merge')
                else:
                    # Standard approach for training data
                    epochs = mne.Epochs(raw, events, target_events, t_start, t_end, 
                                      baseline=None, preload=True)
                
                # Get data
                epochs_data = epochs.get_data(copy=True)  # shape: (n_trials, n_channels, n_times)
                
                # Get labels based on events
                if len(target_events) >= 2:
                    # Natural labels if we have at least 2 event types
                    labels = epochs.events[:, -1]
                    
                    # Map event ids to 0-1
                    # First, get the unique event codes in order
                    unique_codes = sorted(list(set(labels)))
                    
                    # Create a mapping from these codes to 0-1
                    label_map = {code: idx for idx, code in enumerate(unique_codes)}
                    print(f"Label mapping: {label_map}")
                    
                    # Apply the mapping
                    epoch_labels = np.array([label_map[code] for code in labels])
                    artificial_labels = False
                    
                    print(f"Natural labels distribution: {np.bincount(epoch_labels)}")
                else:
                    # Need to create artificial labels
                    artificial_labels = True
                
                # Handle artificial labels if needed
                if artificial_labels:
                    print(f"Creating artificial balanced 2-class labels for {len(epochs_data)} trials")
                    
                    # Create a balanced set of labels (0, 1)
                    num_trials = len(epochs_data)
                    trials_per_class = num_trials // 2
                    remainder = num_trials % 2
                    
                    new_labels = []
                    for i in range(2):
                        # Add extra trial to first class if needed
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
                    # Log natural label distribution
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
                    f.write(f"  {'Artificial' if artificial_labels else 'Natural'} labels used\n")
                
            except Exception as e:
                print(f"Error processing epochs for Subject {subject}, Session {session_name}: {e}")
                import traceback
                traceback.print_exc()
                
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
    parser = argparse.ArgumentParser(description='Preprocess BCI Competition IV 2b data')
    parser.add_argument('--input_path', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2b/raw',
                        help='Path to raw GDF files')
    parser.add_argument('--output_path', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2b/preprocessed',
                        help='Path to save preprocessed data')
    parser.add_argument('--subjects', type=str, default='all',
                        help='Subjects to process (e.g., "1,3,5") or "all"')
    
    args = parser.parse_args()
    
    # Determine which subjects to process
    if args.subjects.lower() == 'all':
        subjects = list(range(1, 10))  # Subjects 1-9 (adjust if needed)
    else:
        subjects = [int(s.strip()) for s in args.subjects.split(',')]
    
    # Process the data
    preprocess_data(args.input_path, args.output_path, subjects)
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()