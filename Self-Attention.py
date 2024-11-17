import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 100
output_dir = "./training_plots"  # 折线图保存目录

# 创建保存目录
os.makedirs(output_dir, exist_ok=True)

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 自注意力模块定义
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    # def forward(self, x):
    #     batch_size, channels, height, width = x.size()
    #     query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, N, C//8)
    #     key = self.key(x).view(batch_size, -1, height * width)  # (B, C//8, N)
    #     attention = torch.softmax(torch.bmm(query, key), dim=-1)  # (B, N, N)
    #     value = self.value(x).view(batch_size, -1, height * width)  # (B, C, N)
    #     out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, N)
    #     out = out.view(batch_size, channels, height, width)  # Reshape back to input shape
    #     out = self.gamma * out + x  # Weighted residual connection
    #     return out

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # (B, N, C//8)
        key = self.key(x).view(batch_size, -1, height * width)  # (B, C//8, N)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)  # (B, N, N)
        value = self.value(x).view(batch_size, -1, height * width)  # (B, C, N)
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, N)
        out = out.view(batch_size, channels, height, width)  # Reshape back to input shape
        # print("SelfAttention output shape:", out.shape)
        out = self.gamma * out + x  # Weighted residual connection
        return out


# 定义带自注意力机制的CNN模型
class CNNWithAttention(nn.Module):
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.attention1 = SelfAttention(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.attention2 = SelfAttention(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # Adjusted input size
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.attention1(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.attention2(x)
        x = x.view(-1, 128 * 4 * 4)  # Adjusted flattening size
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNWithAttention().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练和测试函数
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy

def test(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(loader), accuracy

# 记录训练过程
train_losses, train_accuracies = [], []
test_losses, test_accuracies = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc = test(model, test_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% "
          f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # 绘制并保存折线图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), train_losses, label="Train Loss")
    plt.plot(range(1, epoch + 2), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, epoch + 2), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_plot.png")
    plt.savefig(plot_path)
    plt.close()  # 关闭当前画布