
"""
Conditional U-Net Model Implementation for Polygon Colorization
Complete PyTorch implementation from scratch
Task: Conditional Image Segmentation with Text Input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# U-NET BUILDING BLOCKS
# =============================================================================

class DoubleConv(nn.Module):
    """Double convolution block used in UNet"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool and double conv"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with transpose conv and double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# =============================================================================
# COLOR EMBEDDING FOR CONDITIONAL INPUT
# =============================================================================

class ColorEmbedding(nn.Module):
    """Embedding layer for color names"""
    def __init__(self, num_colors, embed_dim):
        super(ColorEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_colors, embed_dim)
        self.embed_dim = embed_dim
        
    def forward(self, color_idx):
        """
        Args:
            color_idx: Tensor of shape (batch_size,) containing color indices
        Returns:
            embedded: Tensor of shape (batch_size, embed_dim)
        """
        return self.embedding(color_idx)

# =============================================================================
# CONDITIONAL U-NET ARCHITECTURE
# =============================================================================

class ConditionalUNet(nn.Module):
    """UNet with color conditioning"""
    def __init__(self, n_channels=1, n_classes=3, bilinear=True, num_colors=8, color_embed_dim=32):
        super(ConditionalUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.color_embed_dim = color_embed_dim
        
        # Color embedding
        self.color_embedding = ColorEmbedding(num_colors, color_embed_dim)
        
        # Encoder path - Input channels = image channels + color embedding channels
        self.inc = DoubleConv(n_channels + color_embed_dim, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        
        # Decoder path
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
        
    def forward(self, x, color_idx):
        """
        Args:
            x: Input image tensor of shape (batch_size, n_channels, H, W)
            color_idx: Color index tensor of shape (batch_size,)
        Returns:
            logits: Output tensor of shape (batch_size, n_classes, H, W)
        """
        # Get color embedding
        color_embed = self.color_embedding(color_idx)  # (batch_size, color_embed_dim)
        
        # Expand color embedding to match spatial dimensions
        batch_size, embed_dim = color_embed.shape
        h, w = x.shape[2], x.shape[3]
        
        # Reshape and expand: (batch_size, embed_dim, 1, 1) -> (batch_size, embed_dim, h, w)
        color_embed = color_embed.unsqueeze(2).unsqueeze(3).expand(batch_size, embed_dim, h, w)
        
        # Concatenate input image with color embedding
        x = torch.cat([x, color_embed], dim=1)
        
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits

# =============================================================================
# CUSTOM DATASET CLASS
# =============================================================================

class PolygonDataset(Dataset):
    """Custom Dataset for polygon colorization"""
    def __init__(self, data_file, input_dir, output_dir, transform=None, target_transform=None):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Create color to index mapping
        unique_colors = list(set([item['colour'] for item in self.data]))
        unique_colors.sort()  # Ensure consistent ordering
        self.color_to_idx = {color: idx for idx, color in enumerate(unique_colors)}
        self.idx_to_color = {idx: color for color, idx in self.color_to_idx.items()}
        
        print(f"Dataset initialized with {len(self.data)} samples")
        print(f"Unique colors: {unique_colors}")
        print(f"Color mapping: {self.color_to_idx}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load input polygon image (convert to grayscale)
        input_path = os.path.join(self.input_dir, item['input_polygon'])
        try:
            input_image = Image.open(input_path).convert('L')  # Convert to grayscale
        except Exception as e:
            print(f"Error loading input image {input_path}: {e}")
            # Create a dummy image if file not found
            input_image = Image.new('L', (128, 128), 0)
        
        # Load target colored image
        output_path = os.path.join(self.output_dir, item['output_image'])
        try:
            target_image = Image.open(output_path).convert('RGB')
        except Exception as e:
            print(f"Error loading target image {output_path}: {e}")
            # Create a dummy image if file not found
            target_image = Image.new('RGB', (128, 128), (0, 0, 0))
        
        # Get color index
        color_idx = self.color_to_idx[item['colour']]
        
        # Apply transforms
        if self.transform:
            input_image = self.transform(input_image)
        else:
            # Default transform if none provided
            default_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
            input_image = default_transform(input_image)
            
        if self.target_transform:
            target_image = self.target_transform(target_image)
        else:
            # Default transform if none provided
            default_target_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
            target_image = default_target_transform(target_image)
            
        return input_image, target_image, torch.tensor(color_idx, dtype=torch.long)

# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Dice + Cross-Entropy Loss"""
    def __init__(self, weight_dice=0.5, weight_ce=0.5):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.MSELoss()  # Using MSE for RGB regression
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs for stable training
        inputs_sigmoid = torch.sigmoid(inputs)
        
        dice = self.dice_loss(inputs, targets)
        mse = self.ce_loss(inputs_sigmoid, targets)
        
        return self.weight_dice * dice + self.weight_ce * mse

# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_dice_score(pred, target, smooth=1.0):
    """Calculate Dice coefficient"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def calculate_iou(pred, target, smooth=1.0):
    """Calculate Intersection over Union"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def calculate_pixel_accuracy(pred, target):
    """Calculate pixel-wise accuracy"""
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    
    correct = (pred == target).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, wandb_log=True):
    """Complete training loop with logging"""
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print('-' * 50)
        
        for batch_idx, (inputs, targets, color_idx) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            color_idx = color_idx.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, color_idx)
            loss = criterion(outputs, targets)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            running_dice += calculate_dice_score(outputs, targets)
            
            if batch_idx % 5 == 0:  # Print every 5 batches
                print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        with torch.no_grad():
            for inputs, targets, color_idx in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                color_idx = color_idx.to(device)
                
                outputs = model(inputs, color_idx)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_dice += calculate_dice_score(outputs, targets)
        
        # Calculate average metrics
        train_loss_avg = running_loss / len(train_loader)
        train_dice_avg = running_dice / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        val_dice_avg = val_dice / len(val_loader)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_dice_scores.append(train_dice_avg)
        val_dice_scores.append(val_dice_avg)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss_avg)
        
        # Print epoch results
        print(f'Train Loss: {train_loss_avg:.4f}, Train Dice: {train_dice_avg:.4f}')
        print(f'Val Loss: {val_loss_avg:.4f}, Val Dice: {val_dice_avg:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Log to wandb
        if wandb_log:
            try:
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss_avg,
                    'val_loss': val_loss_avg,
                    'train_dice': train_dice_avg,
                    'val_dice': val_dice_avg,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            except:
                pass
        
        # Save best model
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_avg,
                'val_dice': val_dice_avg,
            }, 'best_conditional_unet.pth')
            print(f'New best model saved! Val Loss: {val_loss_avg:.4f}')
    
    print('Training completed!')
    return train_losses, val_losses, train_dice_scores, val_dice_scores

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """Main function to run the training"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10, fill=0),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Initialize datasets (you'll need to update these paths)
    print("Loading datasets...")
    try:
        train_dataset = PolygonDataset(
            'dataset/training/data.json', 
            'dataset/training/inputs', 
            'dataset/training/outputs',
            transform=train_transform, 
            target_transform=target_transform
        )
        
        val_dataset = PolygonDataset(
            'dataset/validation/data.json',
            'dataset/validation/inputs', 
            'dataset/validation/outputs',
            transform=val_transform, 
            target_transform=target_transform
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please make sure the dataset paths are correct!")
        return
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    model = ConditionalUNet(
        n_channels=1, 
        n_classes=3, 
        num_colors=len(train_dataset.color_to_idx),
        color_embed_dim=64,
        bilinear=True
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Loss function and optimizer
    criterion = CombinedLoss(weight_dice=0.6, weight_ce=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Initialize wandb (optional)
    try:
        import wandb
        wandb.init(
            project="polygon-colorization-unet",
            name="conditional-unet-v1",
            config={
                "architecture": "Conditional U-Net",
                "dataset": "Polygon Colorization",
                "epochs": 5,
                "batch_size": 8,
                "learning_rate": 1e-4,
                "optimizer": "AdamW",
                "loss": "Combined Dice + MSE"
            }
        )
        wandb_log = True
    except:
        print("Wandb not available, continuing without logging")
        wandb_log = False
    
    # Train the model
    train_losses, val_losses, train_dice_scores, val_dice_scores = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        num_epochs=10, device=device, wandb_log=wandb_log
    )
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_dice_scores, label='Train Dice')
    plt.plot(val_dice_scores, label='Val Dice')
    plt.title('Dice Score Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # Show sample prediction
    model.eval()
    with torch.no_grad():
        sample_input, sample_target, sample_color = next(iter(val_loader))
        sample_input = sample_input[0:1].to(device)
        sample_color = sample_color[0:1].to(device)
        sample_pred = model(sample_input, sample_color)
        sample_pred = torch.sigmoid(sample_pred)
        
        plt.subplot(131)
        plt.imshow(sample_input[0, 0].cpu().numpy(), cmap='gray')
        plt.title('Input')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(sample_target[0].permute(1, 2, 0).cpu().numpy())
        plt.title('Target')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(sample_pred[0].permute(1, 2, 0).cpu().numpy())
        plt.title('Prediction')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Best validation loss: {min(val_losses):.4f}")
    print(f"Best validation dice: {max(val_dice_scores):.4f}")
    print("Model saved as 'best_conditional_unet.pth'")

if __name__ == "__main__":
    main()