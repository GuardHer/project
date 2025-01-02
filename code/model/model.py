import torch
import torch.nn as nn

class ConvNet(nn.Module):
    """
    模型结构介绍
    卷积层：六个卷积层，每个卷积层后面跟着批量归一化和ReLU激活函数。
    池化层：两个最大池化层，用于减少空间维度。
    全连接层：三个全连接层，带有Dropout正则化。
    输出层：最终的输出层用于二分类
    """
    def __init__(self, in_channels=1):
        super(ConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层，核大小为2x2，步长为2
        self.relu = nn.ReLU()  # ReLU激活函数

        # 第一层卷积层：输入通道=in_channels，输出通道=16，卷积核大小=3x3，步长=1，填充=1
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(16)  # 批量归一化，16个通道

        # 第二层卷积层：输入通道=16，输出通道=32，卷积核大小=3x3，步长=1，填充=1
        self.conv1_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(32)  # 批量归一化，32个通道

        # 第三层卷积层：输入通道=32，输出通道=32，卷积核大小=5x5，步长=1，填充=1
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(32)  # 批量归一化，32个通道

        # 第四层卷积层：输入通道=32，输出通道=64，卷积核大小=5x5，步长=1，填充=1
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(64)  # 批量归一化，64个通道

        # 第五层卷积层：输入通道=64，输出通道=48，卷积核大小=5x5，步长=1，填充=1
        self.conv2_3 = nn.Conv2d(64, 48, kernel_size=5, stride=1, padding=1)
        self.batchNorm5 = nn.BatchNorm2d(48)  # 批量归一化，48个通道

        # 第六层卷积层：输入通道=48，输出通道=192，卷积核大小=5x5，步长=1，填充=1
        self.conv2_4 = nn.Conv2d(48, 192, kernel_size=5, stride=1, padding=1)
        self.batchNorm6 = nn.BatchNorm2d(192)  # 批量归一化，192个通道

        # 全连接层1：输入特征=192 * 52 * 52，输出特征=512
        self.fc1 = nn.Linear(192 * 52 * 52, 512)
        self.dropout1 = nn.Dropout(0.3)  # Dropout，概率为0.3

        # 全连接层2：输入特征=512，输出特征=512
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.5)  # Dropout，概率为0.5

        # 全连接层3：输入特征=512，输出特征=512
        self.fc3 = nn.Linear(512, 512)

        # 输出层：输入特征=512，输出特征=2（用于二分类）
        self.output = nn.Linear(512, 2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.batchNorm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2_1(x)
        x = self.batchNorm3(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.batchNorm4(x)
        x = self.relu(x)
        x = self.conv2_3(x)
        x = self.batchNorm5(x)
        x = self.relu(x)
        x = self.conv2_4(x)
        x = self.batchNorm6(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 192 * 52 * 52)  # 展平张量

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.output(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    model = ConvNet(in_channels=1).to(device)
    print(model)
    x = torch.randn(32, 1, 224, 224).to(device)
    print(model(x).shape)
    print("Model loaded successfully")