import torch
from torchsummary import summary
from model.model import ConvNet, device
from torch import nn, optim
from utils.dataset import ChestXRayDataLoader
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

class Trainer:
    """
    损失函数：使用 nn.CrossEntropyLoss() 进行分类任务。
    优化器：使用 optim.Adam，学习率为 lr。
    学习率调度：使用 optim.lr_scheduler.StepLR，每 step_size 个epoch调整一次学习率，调整因子为 gamma。
    训练循环：模型训练 num_epochs 个epoch，并根据准确率保存最佳模型。
    评估：在测试集上评估模型，计算准确率、精确率、召回率和F1分数。
    绘制结果：绘制训练损失和测试准确率随epoch变化的曲线
    """
    def __init__(self, data_dir, num_epochs=10, lr=0.001, step_size=7, gamma=0.1, channels=1):
        self.data_loader = ChestXRayDataLoader(data_dir, batch_size=32, channels=channels)
        self.num_epochs = num_epochs
        self.net = ConvNet(in_channels=channels).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.accuracy_index = []
        self.losses_index = []
        self.best_accuracy = 0.0
        summary(self.net, (self.data_loader.channels, self.data_loader.height, self.data_loader.width))

    def train(self):
        for epoch in range(self.num_epochs):
            start_time = datetime.now()
            self.net.train()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(self.data_loader.trainLoader, 0), total=len(self.data_loader.trainLoader), desc=f'Epoch {epoch+1}/{self.num_epochs}')
            for i, data in progress_bar:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            epoch_loss = epoch_loss / len(self.data_loader.trainLoader)
            time_elapsed = datetime.now() - start_time
            self.losses_index.append(epoch_loss)
            acc = self.evaluate()
            self.accuracy_index.append(acc)
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f} Test acc: {acc:.4f} time={time_elapsed}')

            # 保存最佳模型
            if acc > self.best_accuracy:
                self.best_accuracy = acc
                torch.save(self.net.state_dict(), 'best_model.pth')

            # 调整学习率
            self.scheduler.step()

        # 保存最终模型
        torch.save(self.net.state_dict(), 'final_model.pth')

    def evaluate(self):
        self.net.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for data in self.data_loader.testLoader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        acc = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        print(f'Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        return acc

    def plot_results(self):
        epochs = range(1, len(self.losses_index) + 1)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.losses_index, 'b', label='Training loss')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.accuracy_index, 'r', label='Test Accuracy')
        plt.title('Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
        plt.savefig('results.png')

if __name__ == "__main__":
    data_dir = "../dataset/chest_xray"
    trainer = Trainer(data_dir, num_epochs=20, lr=0.001, step_size=7, gamma=0.1, channels=1)
    trainer.train()
    trainer.plot_results()