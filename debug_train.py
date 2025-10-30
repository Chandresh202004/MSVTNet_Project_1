import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path  # Use Path for better path handling

# Load a single subject's data
def load_data(subject, data_path):
    # Use Path for better path handling across operating systems
    data_path = Path(data_path)
    train_file = data_path / f"S{subject:02d}_session_1_preprocessed.npz"
    test_file = data_path / f"S{subject:02d}_session_2_preprocessed.npz"
    
    print(f"Looking for training file: {train_file}")
    print(f"Looking for testing file: {test_file}")
    
    # Check if files exist
    if not train_file.exists():
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Testing file not found: {test_file}")
    
    # Load training data
    train_data = np.load(train_file)
    train_X = train_data['data']
    train_y = train_data['labels']
    
    # Load testing data
    test_data = np.load(test_file)
    test_X = test_data['data']
    test_y = test_data['labels']
    
    print(f"Training data: {train_X.shape}, Labels: {train_y.shape}, Unique labels: {np.unique(train_y)}")
    print(f"Testing data: {test_X.shape}, Labels: {test_y.shape}, Unique labels: {np.unique(test_y)}")
    
    # Convert to PyTorch tensors
    train_X = torch.FloatTensor(train_X)
    train_y = torch.LongTensor(train_y)
    test_X = torch.FloatTensor(test_X)
    test_y = torch.LongTensor(test_y)
    
    # Add channel dimension if needed
    if len(train_X.shape) == 3:  # If shape is [batch, channels, time]
        train_X = train_X.unsqueeze(3)  # Add feature dimension [batch, channels, time, 1]
        test_X = test_X.unsqueeze(3)
    
    return train_X, train_y, test_X, test_y

# Simple CNN model for debugging
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes=4):
        super(SimpleCNN, self).__init__()
        
        # Print input shape for debugging
        print(f"Model input shape: {input_shape}")
        
        # Calculate dimensions based on input shape
        batch, channels, time_points, features = input_shape
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=(1, 10), stride=1, padding=(0, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        
        # Calculate new dimensions after first convolution and pooling
        time_after_conv1 = time_points // 2
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 10), stride=1, padding=(0, 5)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )
        
        # Calculate new dimensions after second convolution and pooling
        time_after_conv2 = time_after_conv1 // 2
        
        # We'll dynamically set the FC layer after the first forward pass
        self.flattened_size = None
        self.flatten = nn.Flatten()
        self.fc = None
        self.num_classes = num_classes
    
    def forward(self, x):
        # Print input shape for debugging
        print(f"Input shape: {x.shape}")
        
        x = self.conv1(x)
        print(f"After conv1: {x.shape}")
        
        x = self.conv2(x)
        print(f"After conv2: {x.shape}")
        
        x = self.flatten(x)
        print(f"After flatten: {x.shape}")
        
        # Initialize the FC layer if it's the first forward pass
        if self.fc is None:
            self.flattened_size = x.size(1)
            print(f"Dynamically setting flattened size to: {self.flattened_size}")
            self.fc = nn.Linear(self.flattened_size, self.num_classes)
            # Move FC layer to the same device as the input
            self.fc = self.fc.to(x.device)
        
        x = self.fc(x)
        return x

def train_simple_model():
    data_path = 'D:/MSVTNet_Project/datasets/BCIC_IV_2a/preprocessed'
    subject = 1  # Start with subject 1
    
    # List all files in the directory
    data_dir = Path(data_path)
    print(f"Checking directory: {data_dir}")
    if data_dir.exists():
        print("Files in directory:")
        for file in data_dir.glob("*.npz"):
            print(f"  {file.name}")
    else:
        print(f"Directory doesn't exist: {data_dir}")
        
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load data
        train_X, train_y, test_X, test_y = load_data(subject, data_path)
        
        # Create dataloaders
        train_dataset = TensorDataset(train_X, train_y)
        test_dataset = TensorDataset(test_X, test_y)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Get a sample batch to determine input shape
        inputs, _ = next(iter(train_loader))
        print(f"Batch shape from dataloader: {inputs.shape}")
        
        # Create model
        model = SimpleCNN(inputs.shape, num_classes=4).to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Create optimizer here after the FC layer is initialized
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Print batch results for first few batches only
                if batch_idx < 3:  # Only print first 3 batches to avoid too much output
                    print(f"Batch {batch_idx} - Loss: {loss.item():.4f}, Accuracy: {100 * (predicted == labels).sum().item() / labels.size(0):.2f}%")
                    print(f"Labels: {labels[:5].cpu().numpy()}")
                    print(f"Predictions: {predicted[:5].cpu().numpy()}")
            
            # Test the model
            model.eval()
            total_test_loss = 0
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    total_test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()
            
            train_loss = total_train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            test_loss = total_test_loss / len(test_loader)
            test_acc = 100 * test_correct / test_total
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_simple_model()