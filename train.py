#!/usr/bin/env python
# MSVTNet Training Script for BCI Competition IV 2a and 2b
# Author: Chandresh202004
# Date: 2025-07-13

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import argparse
import datetime
from pathlib import Path

# Import your model and dataloader
from msvtnet import MSVTNet
from dataloader import get_dataloader

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mixup_data(x, y, alpha=0.2):
    """Applies mixup augmentation to the batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Calculates loss for mixup-augmented data."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def focal_loss(outputs, targets, alpha=0.25, gamma=2.0):
    """Compute focal loss for dealing with class imbalance."""
    ce_loss = nn.functional.cross_entropy(outputs, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    loss = alpha * (1-pt)**gamma * ce_loss
    return loss.mean()

def train_one_epoch(model, train_loader, optimizer, device, scaler=None, use_mixup=True, use_focal_loss=False):
    """Train model for one epoch with advanced techniques."""
    model.train()
    running_loss = AverageMeter()
    running_main_loss = AverageMeter()
    running_aux_loss = AverageMeter()
    all_preds = []
    all_labels = []
    
    # Class weights for balancing (computed dynamically if needed)
    class_weights = None
    
    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc=f'Training', leave=False)
    
    for batch_idx, (inputs, labels) in enumerate(pbar):
        # Move inputs and labels to device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply mixup if enabled
        if use_mixup and np.random.random() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
            mixup_applied = True
        else:
            targets_a = labels
            mixup_applied = False
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass with optional mixed precision
        if scaler is not None:
            with torch.amp.autocast('cuda'):  # Updated syntax
                logits, aux_logits = model(inputs)
                
                if mixup_applied:
                    # Custom mixup loss calculation
                    cls_loss = mixup_criterion(
                        lambda p, y: nn.functional.cross_entropy(p, y, weight=class_weights),
                        logits, targets_a, targets_b, lam
                    )
                    
                    # Auxiliary losses with mixup
                    aux_loss = 0
                    for aux_out in aux_logits:
                        aux_loss += mixup_criterion(
                            lambda p, y: nn.functional.cross_entropy(p, y, weight=class_weights),
                            aux_out, targets_a, targets_b, lam
                        )
                    aux_loss /= len(aux_logits) if aux_logits else 1
                    
                    # Combined loss with dynamic weighting
                    if model.training_step < 1000:
                        alpha = 0.8
                    else:
                        alpha = 0.3
                    total_loss = (1 - alpha) * cls_loss + alpha * aux_loss
                else:
                    # Standard loss computation
                    if use_focal_loss:
                        cls_loss = focal_loss(logits, labels)
                        aux_loss = sum(focal_loss(aux, labels) for aux in aux_logits) / len(aux_logits)
                        total_loss = cls_loss + 0.5 * aux_loss
                    else:
                        total_loss, cls_loss, aux_loss, _ = model.compute_loss(logits, aux_logits, labels, class_weights)
            
            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            
            # Apply gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward and backward pass
            logits, aux_logits = model(inputs)
            
            if mixup_applied:
                # Handle mixup similar to above
                cls_loss = mixup_criterion(
                    lambda p, y: nn.functional.cross_entropy(p, y, weight=class_weights),
                    logits, targets_a, targets_b, lam
                )
                
                aux_loss = 0
                for aux_out in aux_logits:
                    aux_loss += mixup_criterion(
                        lambda p, y: nn.functional.cross_entropy(p, y, weight=class_weights),
                        aux_out, targets_a, targets_b, lam
                    )
                aux_loss /= len(aux_logits) if aux_logits else 1
                
                if model.training_step < 1000:
                    alpha = 0.8
                else:
                    alpha = 0.3
                total_loss = (1 - alpha) * cls_loss + alpha * aux_loss
            else:
                if use_focal_loss:
                    cls_loss = focal_loss(logits, labels)
                    aux_loss = sum(focal_loss(aux, labels) for aux in aux_logits) / len(aux_logits)
                    total_loss = cls_loss + 0.5 * aux_loss
                else:
                    total_loss, cls_loss, aux_loss, _ = model.compute_loss(logits, aux_logits, labels, class_weights)
            
            total_loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Track metrics
        running_loss.update(total_loss.item(), inputs.size(0))
        running_main_loss.update(cls_loss.item(), inputs.size(0))
        running_aux_loss.update(aux_loss.item(), inputs.size(0))
        
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss.avg:.4f}',
            'main_loss': f'{running_main_loss.avg:.4f}',
            'aux_loss': f'{running_aux_loss.avg:.4f}'
        })
    
    # Calculate epoch statistics
    epoch_loss = running_loss.avg
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def validate(model, val_loader, device):
    """Validate the model on validation data."""
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = AverageMeter()
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, aux_logits = model(inputs)
            
            # Compute loss
            loss, _, _, _ = model.compute_loss(logits, aux_logits, labels)
            val_loss.update(loss.item(), inputs.size(0))
            
            # Get predictions
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    val_acc = accuracy_score(all_labels, all_preds)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return val_loss.avg, val_acc, cm, all_preds, all_labels

def plot_training_curve(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training and validation curves."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calibrate_for_subject(model, subject_data, device):
    """Fine-tune the model's batch normalization layers for a specific subject"""
    model.train()  # Set to train mode to update BN statistics
    
    with torch.no_grad():  # No gradient updates, just BN statistics
        for data, _ in subject_data:
            data = data.to(device)
            _ = model(data)  # Forward pass updates BN statistics
    
    return model

def check_data_distribution(data_loader, num_classes):
    """Check the class distribution in a dataloader"""
    class_counts = {i: 0 for i in range(num_classes)}
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

def get_class_names(dataset_type):
    """Return class names based on dataset type"""
    if dataset_type == "2a":
        return ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    elif dataset_type == "2b":
        return ['Left Hand', 'Right Hand']
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def train_with_curriculum(model, train_loader, test_loader, optimizer, device, scheduler, 
                          epochs=150, scaler=None, early_stopping=30, save_dir=None, 
                          results_dir=None, subject=1, dataset_type="2a", mode=""):
    """Train model with curriculum learning approach."""
    best_val_acc = 0.0
    epochs_no_improve = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    # Print the correct dataset type and class names
    print(f"Using dataset type {dataset_type} in curriculum training with {len(get_class_names(dataset_type))} classes")
    class_names = get_class_names(dataset_type)
    
    # Phase 1: Train only MSST blocks (freeze CSGT)
    print("Phase 1: Training MSST blocks...")
    for param in model.csgt_encoder.parameters():
        param.requires_grad = False
    
    for epoch in range(30):
        # Train
        epoch_loss, epoch_acc, _, _ = train_one_epoch(model, train_loader, optimizer, device, scaler, use_mixup=True)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validate
        val_loss, val_acc, cm, _, _ = validate(model, test_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{30} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Phase 2: Train CSGT with frozen MSST
    print("Phase 2: Training CSGT encoder...")
    for param in model.msst_blocks.parameters():
        param.requires_grad = False
    for param in model.csgt_encoder.parameters():
        param.requires_grad = True
    
    for epoch in range(30):
        # Train
        epoch_loss, epoch_acc, _, _ = train_one_epoch(model, train_loader, optimizer, device, scaler, use_mixup=True)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validate
        val_loss, val_acc, cm, _, _ = validate(model, test_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{30} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Phase 3: Fine-tune everything
    print("Phase 3: Fine-tuning all components...")
    for param in model.parameters():
        param.requires_grad = True
    
    # Use lower learning rate for fine-tuning
    for g in optimizer.param_groups:
        g['lr'] = g['lr'] * 0.1
    
    remaining_epochs = epochs - 60
    for epoch in range(remaining_epochs):
        # Train
        epoch_loss, epoch_acc, _, _ = train_one_epoch(model, train_loader, optimizer, device, scaler, use_mixup=True)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validate
        val_loss, val_acc, cm, val_preds, val_labels = validate(model, test_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1+60}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            
            # Save best model
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, f'subject_{subject}_best_model.pth'))
                
                # Save confusion matrix for best model
                if results_dir:
                    os.makedirs(results_dir, exist_ok=True)
                    plot_confusion_matrix(cm, class_names, os.path.join(results_dir, f'subject_{subject}_confusion_matrix.png'))
                    
                    # Save classification report
                    clf_report = classification_report(val_labels, val_preds, target_names=class_names)
                    with open(os.path.join(results_dir, f'subject_{subject}_classification_report.txt'), 'w') as f:
                        f.write(clf_report)
                    
                    # Make sure we save history.json at each improvement
                    temp_history = {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accs': train_accs,
                        'val_accs': val_accs,
                        'best_acc': best_val_acc,
                        'training_time': "In progress"
                    }
                    
                    with open(os.path.join(results_dir, f'subject_{subject}_history.json'), 'w') as f:
                        for k, v in temp_history.items():
                            if isinstance(v, np.ndarray):
                                temp_history[k] = v.tolist()
                        json.dump(temp_history, f, indent=2)
        else:
            epochs_no_improve += 1
            
        # Early stopping
        if epochs_no_improve >= early_stopping:
            print(f"Early stopping triggered after {epoch+1+60} epochs")
            break
            
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Save final model and results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f'subject_{subject}_final_model.pth'))
        
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        plot_training_curve(train_losses, val_losses, train_accs, val_accs, os.path.join(results_dir, f'subject_{subject}_learning_curve.png'))
    
    return best_val_acc, train_losses, val_losses, train_accs, val_accs

def determine_dataset_properties(args):
    """Determine the correct number of classes and class names based on dataset"""
    if "2b" in args.data_dir:
        # If we're using BCIC IV 2b data, it's always 2 classes
        print("Detected BCIC IV 2b dataset - using 2 classes")
        return 2, ['Left Hand', 'Right Hand']
    else:
        # Default to 2a dataset with 4 classes
        print(f"Using dataset type: BCIC IV {args.dataset_type}")
        if args.dataset_type == "2a":
            return 4, ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
        else:
            return 2, ['Left Hand', 'Right Hand']

def main():
    parser = argparse.ArgumentParser(description='Train MSVTNet model on BCI Competition IV datasets')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed', 
                        help='Path to the preprocessed dataset')
    parser.add_argument('--dataset_type', type=str, choices=['2a', '2b'], default='2a',
                        help='Dataset type: 2a (4-class) or 2b (2-class)')
    parser.add_argument('--subject', type=int, default=1, choices=range(1, 10), 
                        help='Subject ID (1-9)')
    
    # Session parameters
    parser.add_argument('--session_dependent', action='store_true', 
                       help='Use session-dependent training (train on one session, test on another)')
    parser.add_argument('--train_session', type=int, default=1, choices=[1, 2, 3], 
                       help='Session ID for training (1-3, depending on dataset)')
    parser.add_argument('--test_session', type=int, default=2, choices=[1, 2, 3], 
                       help='Session ID for testing (1-3, depending on dataset)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--early_stopping', type=int, default=30, help='Early stopping patience')
    
    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=32, help='Embedding dimension for MSVTNet')
    
    # Optimization options
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'plateau', 'none'], 
                       help='Learning rate scheduler')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--progressive_training', action='store_true', help='Use progressive training strategy')
    parser.add_argument('--focal_loss', action='store_true', help='Use focal loss')
    parser.add_argument('--mixup', action='store_true', help='Use mixup augmentation')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='models/msvtnet', help='Directory to save trained models')
    parser.add_argument('--results_dir', type=str, default='results/msvtnet', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Special case: Handle 2b data with 2a naming convention
    if "2b" in args.data_dir:
        print("Detected 2b data with 2a file naming convention")
        num_classes = 2  # Force 2 classes for 2b data
        class_names = ['Left Hand', 'Right Hand']
        actual_dataset_type = "2b"  # This is the key fix!
    else:
        # Default dataset type handling
        num_classes = 4 if args.dataset_type == "2a" else 2
        class_names = get_class_names(args.dataset_type)
        actual_dataset_type = args.dataset_type
    
    print(f"Using dataset type: BCIC IV {args.dataset_type} with {num_classes} classes")
    print(f"Classes: {class_names}")
    
    # Determine the mode for file naming (session_dependent or session_independent)
    mode = f"session_dependent_{args.dataset_type}" if args.session_dependent else f"session_independent_{args.dataset_type}"
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save arguments with proper mode in filename
    try:
        with open(os.path.join(args.results_dir, f'subject_{args.subject}_args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(f"Successfully saved arguments to {os.path.join(args.results_dir, f'subject_{args.subject}_args.json')}")
    except Exception as e:
        print(f"Error saving arguments: {e}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    print(f"Loading data for subject {args.subject}...")
    
    if args.session_dependent:
        print(f"Session-dependent mode: Training on session {args.train_session}, testing on session {args.test_session}")
        train_loader, test_loader = get_dataloader(
            args.data_dir, args.subject, args.batch_size, 
            dataset_type=args.dataset_type,
            session_dependent=True,
            train_session=args.train_session, 
            test_session=args.test_session,
            num_workers=args.num_workers,
            augment=args.augment
        )
    else:
        print("Session-independent mode: Using cross-validation across sessions")
        train_loader, test_loader = get_dataloader(
            args.data_dir, args.subject, args.batch_size,
            dataset_type=args.dataset_type,
            session_dependent=False,
            num_workers=args.num_workers,
            augment=args.augment
        )
    
    # Check class balance
    print("Checking data distribution...")
    check_data_distribution(train_loader, num_classes)
    check_data_distribution(test_loader, num_classes)
    
    # Get input dimensions from the data
    sample_input, _ = next(iter(train_loader))
    num_channels = sample_input.shape[1]  # [batch_size, channels, time]
    input_time_length = sample_input.shape[2]
    
    # Create model
    print("Creating MSVTNet model...")
    model = MSVTNet(
        num_channels=num_channels,
        num_classes=num_classes,  # Set based on dataset type
        input_time_length=input_time_length,
        dropout_rate=args.dropout
    )
    model = model.to(device)
    
    # Print model summary
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None
    
    # Training
    print(f"Starting training for subject {args.subject}...")
    start_time = datetime.datetime.now()
    
    if args.progressive_training:
        # Use progressive training strategy
        best_acc, train_losses, val_losses, train_accs, val_accs = train_with_curriculum(
            model, train_loader, test_loader, optimizer, device, scheduler,
            epochs=args.epochs, scaler=scaler, early_stopping=args.early_stopping,
            save_dir=args.save_dir, results_dir=args.results_dir, 
            subject=args.subject, dataset_type=actual_dataset_type, mode=mode  # Use actual_dataset_type here!
        )
    else:
        # Use standard training loop
        best_val_acc = 0.0
        epochs_no_improve = 0
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        
        for epoch in range(args.epochs):
            # Train
            train_loss, train_acc, train_preds, train_labels = train_one_epoch(
                model, train_loader, optimizer, device, scaler,
                use_mixup=args.mixup, use_focal_loss=args.focal_loss
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, cm, val_preds, val_labels = validate(model, test_loader, device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update scheduler
            if scheduler is not None:
                if args.scheduler == 'plateau':
                    scheduler.step(val_acc)
                else:
                    scheduler.step()
            
            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                
                # Save best model
                torch.save(model.state_dict(), os.path.join(args.save_dir, f'subject_{args.subject}_best_model.pth'))
                
                # Save confusion matrix for best model
                plot_confusion_matrix(cm, class_names, os.path.join(args.results_dir, f'subject_{args.subject}_confusion_matrix.png'))
                
                # Save classification report
                clf_report = classification_report(val_labels, val_preds, target_names=class_names)
                with open(os.path.join(args.results_dir, f'subject_{args.subject}_classification_report.txt'), 'w') as f:
                    f.write(clf_report)
                    
                # Save intermediate history at each improvement
                temp_history = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_accs': train_accs,
                    'val_accs': val_accs,
                    'best_acc': best_val_acc,
                    'training_time': "In progress"
                }
                
                with open(os.path.join(args.results_dir, f'subject_{args.subject}_history.json'), 'w') as f:
                    for k, v in temp_history.items():
                        if isinstance(v, np.ndarray):
                            temp_history[k] = v.tolist()
                    json.dump(temp_history, f, indent=2)
            else:
                epochs_no_improve += 1
            
            # Save intermediate results every 10 epochs
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                plot_training_curve(train_losses, val_losses, train_accs, val_accs, os.path.join(args.results_dir, f'subject_{args.subject}_learning_curve.png'))
                
            # Early stopping
            if epochs_no_improve >= args.early_stopping:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        best_acc = best_val_acc
    
    # Training complete
    end_time = datetime.datetime.now()
    training_time = end_time - start_time
    
    print(f"Training completed for subject {args.subject}")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Total training time: {training_time}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, f'subject_{args.subject}_final_model.pth'))
    
    # Save training history - THIS IS THE CRITICAL PART THAT WAS MISSING BEFORE
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_acc': best_acc,
        'training_time': str(training_time)
    }
    
    try:
        with open(os.path.join(args.results_dir, f'subject_{args.subject}_history.json'), 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            for k, v in history.items():
                if isinstance(v, np.ndarray):
                    history[k] = v.tolist()
                    
            json.dump(history, f, indent=2)
        print(f"Successfully saved training history to {os.path.join(args.results_dir, f'subject_{args.subject}_history.json')}")
    except Exception as e:
        print(f"ERROR: Failed to save history file: {e}")

if __name__ == "__main__":
    main()