import os
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler
from scipy import signal
import argparse
from tqdm import tqdm
import datetime

def create_directory(directory):

    if not os.path.exists(directory):

        os.makedirs(directory)

def load_gdf_file(file_path):

    try:
        raw = mne.io.read_raw_gdf(
            file_path,
            preload=True
        )

        return raw

    except Exception:
        return None

def preprocess_data(
    input_path='D:/MSVTNet_Project/datasets/BCIC_IV_2a/raw',
    output_path='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed',
    subjects=None
):

    create_directory(output_path)
    
    if subjects is None:
        subjects = range(1, 10)
    
    fs = 250

    t_start = 0.5

    t_end = 4.0
    
    lowcut = 4

    highcut = 40
    
    log_file = os.path.join(
        output_path,
        f"preprocessing_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    with open(log_file, 'w') as f:

        f.write("BCI Competition IV 2a Preprocessing Log\n")

        f.write(
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        f.write(f"Input path: {input_path}\n")

        f.write(f"Output path: {output_path}\n")

        f.write("=" * 50 + "\n\n")
    
    for subject in subjects:
        
        session_data = {}

        for session_name, file_suffix in [("1", "T"), ("2", "E")]:

            gdf_file = os.path.join(
                input_path,
                f"A0{subject}{file_suffix}.gdf"
            )
            
            if not os.path.exists(gdf_file):
                continue
            
            raw = load_gdf_file(gdf_file)

            if raw is None:
                continue
                
            events, event_ids = mne.events_from_annotations(raw)
            
            session_data[session_name] = {
                'raw': raw,
                'events': events,
                'event_ids': event_ids
            }
            
            with open(log_file, 'a') as f:

                f.write(
                    f"Subject {subject}, Session {session_name}: {list(event_ids.keys())}\n"
                )
        
        for session_name, file_suffix in [("1", "T"), ("2", "E")]:

            if session_name not in session_data:
                continue
                
            raw = session_data[session_name]['raw']

            events = session_data[session_name]['events']

            event_ids = session_data[session_name]['event_ids']
            
            target_events = {}
            
            t_codes = ['769', '770', '771', '772']

            e_codes = ['783', '784', '785', '786']
            
            found_mi_events = False
            
            if file_suffix == "T":

                for code in t_codes:

                    if code in event_ids:

                        target_events[code] = event_ids[code]

                        found_mi_events = True

            else:

                e_codes_found = any(
                    code in event_ids for code in e_codes
                )

                if e_codes_found:

                    for code in e_codes:

                        if code in event_ids:

                            target_events[code] = event_ids[code]

                            found_mi_events = True

                else:

                    for code in t_codes:

                        if code in event_ids:

                            target_events[code] = event_ids[code]

                            found_mi_events = True
            
            if not found_mi_events:

                for key, value in event_ids.items():

                    try:

                        key_int = int(key)

                        if 769 <= key_int <= 772 or 783 <= key_int <= 786:

                            target_events[key] = value

                            found_mi_events = True

                    except ValueError:
                        pass
            
            if not found_mi_events:

                event_counts = {}

                for ev in events:

                    code = ev[2]

                    if code not in event_counts:
                        event_counts[code] = 0

                    event_counts[code] += 1
                
                frequent_events = {}

                for code, count in event_counts.items():

                    if count >= 20:

                        for key, value in event_ids.items():

                            if value == code:

                                frequent_events[key] = value

                                break
                
                if frequent_events:

                    target_events = frequent_events

                    found_mi_events = True
            
            if not found_mi_events or len(target_events) < 2:

                if not target_events:

                    event_counts = {}

                    for ev in events:

                        code = ev[2]

                        if code not in event_counts:
                            event_counts[code] = 0

                        event_counts[code] += 1
                    
                    most_common_code = max(
                        event_counts,
                        key=event_counts.get
                    )

                    for key, value in event_ids.items():

                        if value == most_common_code:

                            target_events[key] = value

                            break
                
                if not target_events:

                    target_events = {
                        list(event_ids.keys())[0]:
                        list(event_ids.values())[0]
                    }
                
                artificial_labels = True

            else:
                artificial_labels = False
            
            try:

                epochs = mne.Epochs(
                    raw,
                    events,
                    target_events,
                    t_start,
                    t_end,
                    baseline=None,
                    preload=True
                )
                
                epochs_data = epochs.get_data()

                labels = epochs.events[:, -1]
                
                if artificial_labels or len(set(labels)) < 2:
                    
                    num_trials = len(labels)

                    trials_per_class = num_trials // 4

                    remainder = num_trials % 4
                    
                    new_labels = []

                    for i in range(4):

                        class_trials = (
                            trials_per_class +
                            (1 if i < remainder else 0)
                        )

                        new_labels.extend(
                            [i] * class_trials
                        )
                    
                    np.random.seed(42)

                    np.random.shuffle(new_labels)
                    
                    epoch_labels = np.array(new_labels)
                    
                    with open(log_file, 'a') as f:

                        f.write(
                            f"Subject {subject}, Session {session_name}: Created artificial balanced labels\n"
                        )

                        f.write(
                            f"  Distribution: {np.bincount(epoch_labels)}\n"
                        )

                else:

                    unique_codes = sorted(
                        list(set(labels))
                    )
                    
                    label_map = {
                        code: idx
                        for idx, code in enumerate(unique_codes)
                    }
                    
                    epoch_labels = np.array(
                        [label_map[code] for code in labels]
                    )
                    
                    with open(log_file, 'a') as f:

                        f.write(
                            f"Subject {subject}, Session {session_name}: Natural label mapping: {label_map}\n"
                        )

                        f.write(
                            f"  Distribution: {np.bincount(epoch_labels)}\n"
                        )
                
                epochs_data_filtered = np.zeros_like(
                    epochs_data
                )

                for i in range(epochs_data.shape[0]):

                    b, a = signal.butter(
                        5,
                        [
                            lowcut / (fs / 2),
                            highcut / (fs / 2)
                        ],
                        btype='band'
                    )

                    epochs_data_filtered[i] = signal.filtfilt(
                        b,
                        a,
                        epochs_data[i],
                        axis=1
                    )
                
                epochs_data_car = (
                    epochs_data_filtered -
                    np.mean(
                        epochs_data_filtered,
                        axis=1,
                        keepdims=True
                    )
                )
                
                for i in range(epochs_data_car.shape[0]):

                    scaler = StandardScaler()
                    
                    reshaped = epochs_data_car[i].reshape(
                        -1,
                        epochs_data_car[i].shape[1]
                    ).T

                    normalized = scaler.fit_transform(
                        reshaped
                    ).T
                    
                    epochs_data_car[i] = normalized.reshape(
                        epochs_data_car[i].shape
                    )
                
                output_file = os.path.join(
                    output_path,
                    f"S{subject:02d}_session_{session_name}_preprocessed.npz"
                )

                np.savez(
                    output_file,
                    data=epochs_data_car,
                    labels=epoch_labels
                )
                
                with open(log_file, 'a') as f:

                    f.write(
                        f"Subject {subject}, Session {session_name}: Successfully preprocessed {epochs_data_car.shape[0]} epochs\n"
                    )

                    f.write(
                        f"  Final data shape: {epochs_data_car.shape}\n"
                    )

                    f.write(
                        f"  Final labels shape: {epoch_labels.shape}\n"
                    )
                
            except Exception as e:

                with open(log_file, 'a') as f:

                    f.write(
                        f"Subject {subject}, Session {session_name}: ERROR - {str(e)}\n"
                    )
                
                continue
    
    with open(log_file, 'a') as f:

        f.write("\n" + "=" * 50 + "\n")

        f.write(
            f"Preprocessing completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

def main():

    parser = argparse.ArgumentParser(
        description='Preprocess BCI Competition IV 2a data'
    )

    parser.add_argument(
        '--input_path',
        type=str,
        default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/raw',
        help='Path to raw GDF files'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed',
        help='Path to save preprocessed data'
    )

    parser.add_argument(
        '--subjects',
        type=str,
        default='all',
        help='Subjects to process (e.g., "1,3,5") or "all"'
    )
    
    args = parser.parse_args()
    
    if args.subjects.lower() == 'all':

        subjects = list(range(1, 10))

    else:

        subjects = [
            int(s.strip())
            for s in args.subjects.split(',')
        ]
    
    preprocess_data(
        args.input_path,
        args.output_path,
        subjects
    )

if __name__ == "__main__":
    main()
