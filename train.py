import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.UNet import UNet  # 确保 UNet 模型正确导入
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard
import albumentations as A
from albumentations.pytorch import ToTensorV2
from losses.lovasz_losses import lovasz_softmax  # 确保已正确导入 lovasz_losses

# 数据集定义
class SteelDefectDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.png'))  

        image = np.array(Image.open(img_path).convert("L"))  # 转换为灰度图
        label = np.array(Image.open(label_path).convert("L"))  # 标签也是灰度图

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        label = label.long()  # 确保 label 是 tensor 并转换为 long 类型
        return image, label

# 数据增广
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=0.445450, std=0.12117627335),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=0.445450, std=0.12117627335),
    ToTensorV2(),
])

# 混淆矩阵计算
def compute_confusion_matrix(preds, labels, num_classes=4):
    with torch.no_grad():
        preds = preds.view(-1)
        labels = labels.view(-1)
        mask = (labels >= 0) & (labels < num_classes)
        preds = preds[mask]
        labels = labels[mask]
        confusion = torch.bincount(num_classes * labels + preds, minlength=num_classes**2).reshape(num_classes, num_classes)
    return confusion

# 从混淆矩阵计算 IoU
def compute_iou_from_confusion(confusion_matrix, num_classes=4, ignore_background=True):
    intersection = torch.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(1)
    predicted_set = confusion_matrix.sum(0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / (union + 1e-10)

    if ignore_background:
        IoU_no_background = IoU[1:]
        mean_IoU = IoU_no_background.mean().item()
    else:
        mean_IoU = IoU.mean().item()

    return IoU, mean_IoU

# 训练函数，加入提前停止条件
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, save_path="UNet_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    
    best_iou = 0.0  # 记录验证集上最佳的IoU
    writer = SummaryWriter('runs/UNet_experiment')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # 组合使用交叉熵损失和 lovasz_softmax 损失
            lovasz_loss = lovasz_softmax(torch.softmax(outputs, dim=1), labels, ignore=255)
            ce_loss = criterion(outputs, labels)
            loss = ce_loss + lovasz_loss
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)

        # 验证模型
        avg_iou = validate_model(model, val_loader, criterion, writer, epoch)
        writer.add_scalar('IoU/val_mean_no_background', avg_iou, epoch)

        # 如果在验证集上的IoU更高，则保存模型权重
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(), save_path)
            print(f"New best model saved with IoU: {best_iou:.4f}")

    writer.close()

# 验证函数
def validate_model(model, val_loader, criterion, writer=None, epoch=None, num_classes=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    running_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes).to(device)
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="验证中"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            
            preds = torch.argmax(outputs, dim=1)
            confusion_matrix += compute_confusion_matrix(preds, labels, num_classes)
    
    avg_loss = running_loss / len(val_loader.dataset)
    IoU, mean_IoU = compute_iou_from_confusion(confusion_matrix, num_classes, ignore_background=True)
    
    print(f"验证损失: {avg_loss:.4f}")
    print("每类 IoU:")
    for cls in range(num_classes):
        print(f"类别 {cls} IoU: {IoU[cls]:.4f}")
    print(f"平均 IoU（排除背景）: {mean_IoU:.4f}")

    if writer and epoch is not None:
        writer.add_scalar('Loss/val', avg_loss, epoch)
        writer.add_scalar('IoU/val_mean_no_background', mean_IoU, epoch)
        for cls in range(num_classes):
            writer.add_scalar(f'IoU/val_class_{cls}', IoU[cls].item(), epoch)
    
    return mean_IoU

# 路径配置
train_image_dir = 'dataset/images/training'
train_label_dir = 'dataset/annotations/training'
test_image_dir = 'dataset/images/test'
test_label_dir = 'dataset/annotations/test'

train_dataset = SteelDefectDataset(train_image_dir, train_label_dir, transform=train_transform)
val_dataset = SteelDefectDataset(test_image_dir, test_label_dir, transform=val_transform)


train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

# 模型、损失和优化器
model = UNet(in_channels=1, num_classes=4)  # 输入通道改为1（灰度图），输出类别改为4（背景+三类缺陷）
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=4e-4)

# 开始训练
train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=1000, save_path="UNet.pth")
