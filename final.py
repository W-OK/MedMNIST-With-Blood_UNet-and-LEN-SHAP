import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Dataset definition
class BloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace(".png", "_mask.png"))

        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class DualAttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(DualAttentionGate, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        ch_att = self.channel_attention(x)  # Channel attention
        sp_att = self.spatial_attention(x)  # Spatial attention
        att = ch_att * sp_att  # Combine channel and spatial attention
        return att * x  # Apply attention to input

# BloodUNet definition
class Blood_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Blood_UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Dual Attention Gate
        self.dual_attention = DualAttentionGate(512)

        # Decoder
        self.decoder1 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder3 = self.conv_block(128, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # Max pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        # Dual Attention Gate
        enc4_att = self.dual_attention(enc4)

        # Decoder with skip connections
        dec1 = self.decoder1(torch.cat([self.upsample(enc4_att), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.upsample(dec1), enc2], dim=1))
        dec3 = self.decoder3(torch.cat([self.upsample(dec2), enc1], dim=1))

        out = self.final_conv(dec3)
        return out

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        print("Input size:", input.size())
        print("Target size:", target.size())

        smooth = 1e-6
        target_resized = F.interpolate(target.unsqueeze(1), size=(input.size(2), input.size(3)),
                                       mode='nearest').squeeze(1)
        input_flat = input.view(-1)
        target_flat = target_resized.view(-1)
        intersection = torch.sum(input_flat * target_flat)
        dice_coeff = (2. * intersection + smooth) / (torch.sum(input_flat) + torch.sum(target_flat) + smooth)
        return 1 - dice_coeff


# LEN Model
class LEN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(LEN, self).__init__()
        # Define layers for LESION ENHANCEMENT MAP model
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Forward pass of LESION ENHANCEMENT MAP model
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x

# Shape Model
class ShapeModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ShapeModel, self).__init__()
        # Define layers for SHAPE model
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Forward pass of SHAPE model
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def visualize_outputs(len_outputs, shape_outputs, figsize=(8, 8), fontsize=8):
    batch_size = len(len_outputs)
    cols = 20  # 每行显示的列数
    rows = (batch_size + cols - 1) // cols  # 计算行数

    fig, axes = plt.subplots(rows * 2, cols, figsize=(figsize[0] * cols, figsize[1] * rows))

    for i in range(rows * 2):
        for j in range(cols):
            index = i * cols + j
            if index < batch_size:
                if i % 2 == 0:
                    len_output_image = len_outputs[index].squeeze().cpu().numpy()
                    if len_output_image.shape[0] == 3:  # 如果通道维度在第一个维度上
                        len_output_image = len_output_image.transpose(1, 2, 0)
                    axes[i, j].imshow(len_output_image, cmap='gray')
                    axes[i, j].set_title(f'LEN Model Output (Sample {index + 1})', fontsize=fontsize)
                else:
                    shape_output_image = shape_outputs[index].squeeze().cpu().numpy()
                    if shape_output_image.shape[0] == 3:  # 如果通道维度在第一个维度上
                        shape_output_image = shape_output_image.transpose(1, 2, 0)
                    axes[i, j].imshow(shape_output_image, cmap='gray')
                    axes[i, j].set_title(f'Shape Model Output (Sample {index + 1})', fontsize=fontsize)
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()



checkpoint_path = 'best_model_checkpoint.pth'

if __name__ == "__main__":
    # DataLoader creation and split into training and validation
    dataset = BloodDataset(image_dir="./Train/Image", mask_dir="./Train/Layer_Masks", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Model, loss, and optimizer
    model = Blood_UNet(3, 1).to(device)
    criterion = DiceLoss()  # 使用Dice损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # LEN and Shape Model
    len_model = LEN().to(device)
    shape_model = ShapeModel().to(device)

    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    best_accuracy = 0

    train_losses = []
    val_losses = []
    len_losses = []
    shape_losses = []
    accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        total_len_loss = 0
        total_shape_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                total_val_loss += loss.item()

                # Compute LEN and Shape model losses
                len_output = len_model(outputs)
                shape_output = shape_model(outputs)
                len_loss = criterion(len_output, labels.to(device))  # Assuming labels are suitable for LEN model
                shape_loss = criterion(shape_output, labels.to(device))  # Assuming labels are suitable for Shape model
                total_len_loss += len_loss.item()
                total_shape_loss += shape_loss.item()

                # Compute accuracy
                predicted = torch.round(torch.sigmoid(outputs))  # Binarize the outputs
                total += labels.numel()
                correct += (predicted == labels).sum().item()

        average_val_loss = total_val_loss / len(val_loader)
        average_len_loss = total_len_loss / len(val_loader)
        average_shape_loss = total_shape_loss / len(val_loader)
        accuracy = correct / total

        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        len_losses.append(average_len_loss)
        shape_losses.append(average_shape_loss)
        accuracies.append(accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}, LEN Loss: {average_len_loss:.4f}, Shape Loss: {average_shape_loss:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), checkpoint_path)

    print("Training and validation finished.")

    # Load the best model's weights for evaluation
    model.load_state_dict(torch.load(checkpoint_path))

    # Evaluation on validation set
    correct = 0
    total = 0
    len_correct = 0
    shape_correct = 0
    model.eval()
    len_outputs = []
    shape_outputs = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            len_output = len_model(outputs)
            shape_output = shape_model(outputs)
            len_outputs.append(len_output)
            shape_outputs.append(shape_output)

            # Compute accuracy for UNet model
            predicted = torch.round(torch.sigmoid(outputs))  # Binarize the outputs
            total += labels.numel()
            correct += (predicted == labels).sum().item()

            # Compute accuracy for LEN model
            len_predicted = torch.round(torch.sigmoid(len_output))
            len_correct += (len_predicted == labels).sum().item()

            # Compute accuracy for Shape model
            shape_predicted = torch.round(torch.sigmoid(shape_output))
            shape_correct += (shape_predicted == labels).sum().item()

    # Calculate accuracy for UNet model
    accuracy = correct / total

    # Calculate accuracy for LEN model
    len_accuracy = len_correct / total

    # Calculate accuracy for Shape model
    shape_accuracy = shape_correct / total

    print(f"Accuracy of the UNet model on the validation set: {accuracy:.4f}")
    print(f"Accuracy of the LEN model on the validation set: {len_accuracy:.4f}")
    print(f"Accuracy of the Shape model on the validation set: {shape_accuracy:.4f}")

    # Visualize LEN and Shape model outputs after training
    len_outputs = torch.cat(len_outputs, dim=0)
    shape_outputs = torch.cat(shape_outputs, dim=0)
    visualize_outputs(len_outputs, shape_outputs)
    plt.show()

    # Plot Losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

    # Plot LEN and Shape Losses
    plt.plot(len_losses, label='LEN Loss')
    plt.plot(shape_losses, label='Shape Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LEN and Shape Model Losses')
    plt.legend()
    plt.show()

    # Plot Accuracy
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
