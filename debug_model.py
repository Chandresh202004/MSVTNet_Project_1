import torch
import numpy as np
from msvtnet import MSVTNet
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse
from tqdm import tqdm

def debug_model(data_loader, device=None):
    """Debug MSVTNet model implementation"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Get a batch
    inputs, labels = next(iter(data_loader))
    print(f"Input shape: {inputs.shape}, Labels: {labels}")
    
    # Create model
    model = MSVTNet(
        num_channels=inputs.shape[1],
        num_classes=4,
        input_time_length=inputs.shape[2],
        dropout_rate=0.5
    )
    model.to(device)
    
    inputs = inputs.to(device)
    
    # Test forward pass
    print("Testing forward pass...")
    logits, aux_logits = model(inputs)
    print(f"Output logits shape: {logits.shape}")
    print(f"Number of auxiliary outputs: {len(aux_logits)}")
    
    # Check feature extraction
    print("Testing feature extraction...")
    features = model.extract_features(inputs)
    print(f"Features shape: {features.shape}")
    print(f"Feature norm: {torch.norm(features).item()}")
    
    # Check gradient flow
    print("Testing gradient flow...")
    labels = labels.to(device)
    loss, cls_loss, aux_loss, _ = model.compute_loss(logits, aux_logits, labels)
    loss.backward()
    
    # Check gradients
    print("Checking gradients in each component:")
    component_grads = {
        'msst_blocks': [],
        'csgt_encoder': [],
        'classifier': []
    }
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            if 'msst_blocks' in name:
                component_grads['msst_blocks'].append(grad_norm)
            elif 'csgt_encoder' in name:
                component_grads['csgt_encoder'].append(grad_norm)
            elif 'classifier' in name:
                component_grads['classifier'].append(grad_norm)
                
            if grad_norm > 10.0:
                print(f"Warning: {name} has high gradient norm {grad_norm:.4f}")
            elif grad_norm < 1e-5:
                print(f"Warning: {name} has very low gradient norm {grad_norm:.8f}")
    
    for component, grads in component_grads.items():
        if grads:
            print(f"{component} - Avg grad norm: {np.mean(grads):.6f}, Max: {np.max(grads):.6f}")
    
    print("Model debug complete!")

def check_data_distribution(data_loader):
    """Check the class distribution in a dataloader"""
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    sample_means = []
    sample_stds = []
    
    for inputs, labels in data_loader:
        for label in labels:
            class_counts[label.item()] += 1
        
        # Check data statistics
        sample_means.append(inputs.mean().item())
        sample_stds.append(inputs.std().item())
    
    print(f"Class distribution: {class_counts}")
    print(f"Data mean: {np.mean(sample_means):.6f}, std: {np.mean(sample_stds):.6f}")
    
    # Check if classes are balanced
    total = sum(class_counts.values())
    expected_per_class = total / len(class_counts)
    imbalance = max(abs(count - expected_per_class)/expected_per_class for count in class_counts.values())
    
    if imbalance > 0.1:  # More than 10% imbalance
        print(f"Warning: Dataset is imbalanced (max deviation: {imbalance*100:.1f}%)")
    else:
        print("Dataset is well-balanced")

def visualize_features(model, data_loader, device, save_path):
    """Visualize features extracted by the model"""
    model.eval()
    features = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Extracting features"):
            inputs = inputs.to(device)
            feats = model.extract_features(inputs)
            features.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
    
    features = np.concatenate(features)
    labels_list = np.concatenate(labels_list)
    
    # Use t-SNE for dimensionality reduction
    print("Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'y']
    markers = ['o', 's', '^', 'D']
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    
    for i in range(4):
        mask = labels_list == i
        plt.scatter(
            features_2d[mask, 0], 
            features_2d[mask, 1], 
            c=colors[i], 
            marker=markers[i],
            label=class_names[i],
            alpha=0.7
        )
    
    plt.legend()
    plt.title('Feature Space Visualization (t-SNE)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature visualization saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Debug MSVTNet model')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed', help='Path to the preprocessed dataset')
    parser.add_argument('--subject', type=int, default=1, choices=range(1, 10), help='Subject ID (1-9)')
    
    # Session parameters
    parser.add_argument('--session_dependent', action='store_true', help='Use session-dependent training')
    parser.add_argument('--train_session', type=int, default=1, choices=[1, 2], help='Session ID for training')
    parser.add_argument('--test_session', type=int, default=2, choices=[1, 2], help='Session ID for testing')
    
    # Output parameters
    parser.add_argument('--results_dir', type=str, default='D:/MSVTNet_Project/debug_results', help='Directory to save debug results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import dataloader here to avoid circular imports
    from dataloader import get_dataloader
    
    # Load data
    print(f"Loading data for subject {args.subject}...")
    
    if args.session_dependent:
        print(f"Session-dependent mode: Training on session {args.train_session}, testing on session {args.test_session}")
        train_loader, test_loader = get_dataloader(
            args.data_dir, args.subject, args.batch_size, 
            session_dependent=True,
            train_session=args.train_session, 
            test_session=args.test_session,
            num_workers=0
        )
    else:
        print("Session-independent mode: Using cross-validation across sessions")
        train_loader, test_loader = get_dataloader(
            args.data_dir, args.subject, args.batch_size, 
            session_dependent=False,
            num_workers=0
        )
    
    # Debug model
    print("\nDebugging model...")
    debug_model(train_loader, device)
    
    # Check data distribution
    print("\nChecking training data distribution...")
    check_data_distribution(train_loader)
    print("\nChecking test data distribution...")
    check_data_distribution(test_loader)
    
    # Create a model for feature visualization
    sample_input, _ = next(iter(train_loader))
    num_channels = sample_input.shape[1]
    input_time_length = sample_input.shape[2]
    
    model = MSVTNet(
        num_channels=num_channels,
        num_classes=4,
        input_time_length=input_time_length,
        dropout_rate=0.5
    )
    model.to(device)
    
    # Train for a few mini-batches to learn some features
    print("\nTraining model for a few mini-batches...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    for _ in range(5):
        for inputs, labels in tqdm(train_loader, desc="Training mini-batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, aux_logits = model(inputs)
            loss, _, _, _ = model.compute_loss(logits, aux_logits, labels)
            loss.backward()
            optimizer.step()
    
    # Visualize features
    print("\nVisualizing features...")
    visualize_features(model, test_loader, device, os.path.join(args.results_dir, f'subject_{args.subject}_features.png'))

if __name__ == "__main__":
    main()