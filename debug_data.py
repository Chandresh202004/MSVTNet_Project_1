import os
import numpy as np
import matplotlib.pyplot as plt

def check_and_fix_labels(input_path, output_path=None):

    if output_path is None:
        output_path = input_path
    
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )
    
    data = np.load(input_path)

    X = data['data']
    y = data['labels']
    
    unique_labels, counts = np.unique(
        y,
        return_counts=True
    )
    
    plt.figure(figsize=(15, 10))

    for i in range(min(5, len(X))):

        plt.subplot(5, 1, i + 1)

        plt.plot(X[i, 0, :])

        plt.title(
            f"Trial {i}, Label: {y[i]}"
        )

    plt.tight_layout()

    plt.savefig(
        f"{os.path.splitext(input_path)[0]}_samples.png"
    )

    plt.close()
    
    fixed = False

    if np.min(y) > 0 or np.max(y) > 3:

        if np.min(y) >= 769:

            new_y = np.array(
                [y_val - 769 for y_val in y]
            )

        elif np.min(y) >= 1 and np.max(y) <= 4:

            new_y = y - 1

        else:
            return False
        
        unique_new, counts_new = np.unique(
            new_y,
            return_counts=True
        )
        
        np.savez(
            output_path,
            data=X,
            labels=new_y
        )

        fixed = True

    if np.isnan(X).any() or np.isinf(X).any():

        nan_count = np.isnan(X).sum()

        inf_count = np.isinf(X).sum()
        
        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        
        np.savez(
            output_path,
            data=X,
            labels=new_y if fixed else y
        )

        fixed = True
    
    data_min = X.min()

    data_max = X.max()

    data_mean = X.mean()

    data_std = X.std()
    
    if data_std < 1e-6 or data_std > 1e6:
        pass
    
    return fixed

def process_all_files(data_path):

    fixed_any = False
    
    for subject in range(1, 10):

        for session in [1, 2]:

            file_path = os.path.join(
                data_path,
                f"S{subject:02d}_session_{session}_preprocessed.npz"
            )

            if os.path.exists(file_path):

                fixed = check_and_fix_labels(
                    file_path
                )

                fixed_any = fixed_any or fixed
    
    return fixed_any

if __name__ == "__main__":

    import argparse
    
    parser = argparse.ArgumentParser(
        description='Check and fix preprocessed data'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed',
        help='Path to preprocessed data'
    )
    
    args = parser.parse_args()
    
    fixed = process_all_files(
        args.data_path
    )
    
    if fixed:
        print(
            "\nFixed issues in one or more files. Please run training again."
        )

    else:
        print(
            "\nNo issues found or fixed in the data files."
        )
