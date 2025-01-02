import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        self.results['accuracy'].append(accuracy)
        self.results['precision'].append(precision)
        self.results['recall'].append(recall)
        self.results['f1'].append(f1)

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

        return accuracy, precision, recall, f1


if __name__ == "__main__":
    from model.model import ConvNet, device
    from utils.dataset import ChestXRayDataLoader

    data_dir = "../dataset/chest_xray"
    data_loader = ChestXRayDataLoader(data_dir)
    model = ConvNet().to(device)
    model.load_state_dict(torch.load('best_model.pth'))

    evaluator = Evaluator(model, data_loader.testLoader, device)
    evaluator.evaluate()
