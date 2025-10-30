import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MSVTNet Training for BCI Competition IV 2a')
    parser.add_argument('--subject', type=int, default=1,
                        help='Subject number (1-9)')
    parser.add_argument('--data_path', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed',
                        help='Path to preprocessed data')
    parser.add_argument('--save_dir', type=str, default='D:/MSVTNet_Project/models',
                        help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='D:/MSVTNet_Project/results',
                        help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size of the model')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['plateau', 'cosine', 'step'],
                        help='Learning rate scheduler type')
    parser.add_argument('--early_stopping', type=int, default=30,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--session_dependent', action='store_true',
                        help='Use session-dependent approach')
    parser.add_argument('--train_session', type=int, default=1,
                        help='Session for training in session-dependent mode')
    parser.add_argument('--test_session', type=int, default=2,
                        help='Session for testing in session-dependent mode')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use if multiple GPUs are available')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of worker processes for data loading')
    
    return parser.parse_args()

def main():
    args = parse_args()
    print("Arguments parsed successfully")
    print(f"Subject: {args.subject}")
    print(f"Data path: {args.data_path}")
    print("Verification complete")

if __name__ == "__main__":
    main()