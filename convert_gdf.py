import os
import mne
import numpy as np
import h5py

def convert_gdf_to_h5():
    data_path = 'D:/MSVTNet_Project/datasets/BCIC_IV_2a/raw'
    output_path = 'D:/MSVTNet_Project/datasets/BCIC_IV_2a/processed'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    subjects = range(1, 10) 
    
    for subject in subjects:
        for session in [1, 2]:  
            file_name = f"A{subject:02d}{'T' if session==1 else 'E'}.gdf"
            file_path = os.path.join(data_path, file_name)
            
            raw = mne.io.read_raw_gdf(file_path, preload=True)
            
            events, event_ids = mne.events_from_annotations(raw)
            
            ch_names = raw.ch_names
            data = raw.get_data()
            
            labels = []
            if session == 1:
                for event in events:
                    if event[2] in [1, 2, 3, 4]: 
                        labels.append(event[2])
            else:
                label_file = os.path.join(data_path, f"A{subject:02d}E_labels.txt")
                if os.path.exists(label_file):
                    labels = np.loadtxt(label_file)
                else:
                    print(f"Warning: Label file {label_file} not found.")
            
            output_file = os.path.join(output_path, f"S{subject:02d}_session_{session}.h5")
            
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('eeg_data', data=data)
                f.create_dataset('events', data=events)
                if labels:
                    f.create_dataset('labels', data=np.array(labels))
                for i, name in enumerate(ch_names):
                    f['eeg_data'].attrs[f'ch_name_{i}'] = name
                
            print(f"Processed Subject {subject}, Session {session}")

if __name__ == "__main__":
    convert_gdf_to_h5()
