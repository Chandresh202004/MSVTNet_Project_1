import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

from dataloader import get_dataloader
from msvtnet import MSVTNet

def parse_args():
    parser = argparse.ArgumentParser(description='MSVTNet Training for BCI Competition IV 2a with GPU')
    parser.add_argument('--data_path', type=str, default='D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed',
                        help='Path to preprocessed data')
    parser.add_argument('--subject', type=int, default=1, help='Subject number (1-9)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size of the model')
    parser.add_argument('--session_dependent', action='store_true', help='Use session-dependent approach')
    parser.add_argument('--train_session', type=int, default=1, help='Session for training in session-dependent mode')
    parser.add_argument('--test_session', type=int, default=2, help='Session for testing in session-dependent mode')
    parser.add_argument('--save_dir', type=str, default='D:/MSVTNet_Project/models', help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='D:/MSVTNet_Project/results', help='Directory to save results')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--auto_batch_size', action='store_true', help='Automatically determine optimal batch size')
    return parser.parse_args()

def train(model, train_loader, criterion, optimizer, device, use_mixed_precision=False):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # For mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
    
    # Start timing
    start_time = time.time()
    
    for inputs, labels in tqdm(train_loader, desc='Training'):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # More efficient than optimizer.zero_grad()
        
        if use_mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    main_out, aux1_out, aux2_out, aux3_out = outputs
                    loss = criterion(main_out, labels) + 0.3 * (
                        criterion(aux1_out, labels) + 
                        criterion(aux2_out, labels) + 
                        criterion(aux3_out, labels)
                    )
                    preds = torch.argmax(main_out, dim=1)
                else:
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                main_out, aux1_out, aux2_out, aux3_out = outputs
                loss = criterion(main_out, labels) + 0.3 * (
                    criterion(aux1_out, labels) + 
                    criterion(aux2_out, labels) + 
                    criterion(aux3_out, labels)
                )
                preds = torch.argmax(main_out, dim=1)
            else:
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
            
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, training_time

def evaluate(model, test_loader, criterion, device, use_mixed_precision=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Start timing
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    
                    if isinstance(outputs, tuple):
                        main_out = outputs[0]
                        loss = criterion(main_out, labels)
                        preds = torch.argmax(main_out, dim=1)
                    else:
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, dim=1)
            else:
                outputs = model(inputs)
                
                if isinstance(outputs, tuple):
                    main_out = outputs[0]
                    loss = criterion(main_out, labels)
                    preds = torch.argmax(main_out, dim=1)
                else:
                    loss = criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
            
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # End timing
    end_time = time.time()
    eval_time = end_time - start_time
    
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=['Left Hand', 'Right Hand', 'Feet', 'Tongue'])
    
    return epoch_loss, epoch_acc, cm, report, eval_time

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_training_curves(train_losses, test_losses, train_accs, test_accs, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Check for CUDA availability
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU #{args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        
        # Print GPU info
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(args.gpu_id).total_memory / 1024**3:.2f} GB")
        print(f"Memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        print("WARNING: CUDA is not available. Running on CPU.")
    
    # Set subject-specific directories
    subject_save_dir = os.path.join(args.save_dir, f"subject_{args.subject}")
    subject_results_dir = os.path.join(args.results_dir, f"subject_{args.subject}")
    os.makedirs(subject_save_dir, exist_ok=True)
    os.makedirs(subject_results_dir, exist_ok=True)
    
    # Set mode-specific directories
    mode = "session_dependent" if args.session_dependent else "session_independent"
    mode_save_dir = os.path.join(subject_save_dir, mode)
    mode_results_dir = os.path.join(subject_results_dir, mode)
    os.makedirs(mode_save_dir, exist_ok=True)
    os.makedirs(mode_results_dir, exist_ok=True)
    
    # Auto-determine optimal batch size if requested
    if args.auto_batch_size and device.type == 'cuda':
        try:
            with open('optimal_batch_size.txt', 'r') as f:
                args.batch_size = int(f.read().strip())
                print(f"Using optimal batch size from file: {args.batch_size}")
        except FileNotFoundError:
            from optimize_batch_size import find_optimal_batch_size
            args.batch_size = find_optimal_batch_size(device, max_batch_size=512, start_batch_size=16)
    
    # Get dataloaders
    train_loader, test_loader = get_dataloader(
        args.data_path,
        args.subject,
        batch_size=args.batch_size,
        session_dependent=args.session_dependent,
        train_session=args.train_session,
        test_session=args.test_session,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Testing dataset size: {len(test_loader.dataset)}")
    
    # Initialize model
    model = MSVTNet(in_channels=1, num_classes=4, hidden_size=args.hidden_size)
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Training loop
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    train_times = []
    eval_times = []
    best_test_acc = 0.0
    
    print(f"\nStarting training with {'mixed precision' if args.mixed_precision else 'full precision'}")
    total_start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_time = train(model, train_loader, criterion, optimizer, device, args.mixed_precision)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_times.append(train_time)
        
        # Evaluate
        test_loss, test_acc, confusion_mat, report, eval_time = evaluate(model, test_loader, criterion, device, args.mixed_precision)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        eval_times.append(eval_time)
        
        # Update learning rate
        scheduler.step(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Time: {train_time:.2f}s")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Time: {eval_time:.2f}s")
        
        # Print GPU memory usage
        if device.type == 'cuda':
            print(f"GPU memory usage:")
            print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(mode_save_dir, f"best_model.pth"))
            
            # Save confusion matrix for best model
            plot_confusion_matrix(
                confusion_mat, 
                ['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
                os.path.join(mode_results_dir, f"confusion_matrix.png")
            )
            
            # Save classification report
            with open(os.path.join(mode_results_dir, f"classification_report.txt"), 'w') as f:
                f.write(report)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
            }, os.path.join(mode_save_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(mode_save_dir, f"final_model.pth"))
    
    # Plot and save training curves
    plot_training_curves(
        train_losses, test_losses, train_accs, test_accs,
        os.path.join(mode_results_dir, f"training_curves.png")
    )
    
    # Save training history
    np.savez(
        os.path.join(mode_results_dir, f"training_history.npz"),
        train_losses=np.array(train_losses),
        test_losses=np.array(test_losses),
        train_accs=np.array(train_accs),
        test_accs=np.array(test_accs),
        train_times=np.array(train_times),
        eval_times=np.array(eval_times)
    )
    
    # Calculate and save training statistics
    avg_train_time = np.mean(train_times)
    avg_eval_time = np.mean(eval_times)
    
    with open(os.path.join(mode_results_dir, f"training_stats.txt"), 'w') as f:
        f.write(f"Total training time: {total_time/60:.2f} minutes\n")
        f.write(f"Average training time per epoch: {avg_train_time:.2f} seconds\n")
        f.write(f"Average evaluation time per epoch: {avg_eval_time:.2f} seconds\n")
        f.write(f"Best test accuracy: {best_test_acc:.4f}\n")
        f.write(f"GPU: {torch.cuda.get_device_name(args.gpu_id) if device.type == 'cuda' else 'CPU'}\n")
        f.write(f"Mixed precision: {args.mixed_precision}\n")
        f.write(f"Batch size: {args.batch_size}\n")
    
    print(f"\nTraining completed. Best test accuracy: {best_test_acc:.4f}")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Models saved to {mode_save_dir}")
    print(f"Results saved to {mode_results_dir}")

if __name__ == "__main__":
    main()