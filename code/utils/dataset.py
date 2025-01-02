import os
from torchvision import transforms as T, datasets
from torch.utils.data import DataLoader

class ChestXRayDataLoader:
    def __init__(self, data_dir, batch_size=32, channels=1):
        self.height = 224
        self.width = 224
        self.channels = channels
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.TRAIN = 'train'
        self.TEST = 'test'
        self.VAL = 'val'
        self.trainLoader = None
        self.validLoader = None
        self.testLoader = None
        self.class_names = None
        self.class_to_idx = None
        self._prepare_data_loaders()

    def data_transforms(self, phase=None):
        if phase == self.TRAIN:
            data_T = T.Compose(
                [T.Resize(size=(self.height, self.width), interpolation=2),
                 T.Grayscale(num_output_channels=self.channels),
                 T.ColorJitter(brightness=0.05, contrast=0.8, saturation=0.3),
                 T.RandomHorizontalFlip(p=0.5),
                 T.RandomVerticalFlip(p=0.5),
                 T.ToTensor(),
                 T.Normalize([0.485] * self.channels, [0.229] * self.channels)])
        elif phase == self.TEST or phase == self.VAL:
            data_T = T.Compose(
                [T.Resize(size=(self.height, self.width), interpolation=2),
                 T.Grayscale(num_output_channels=self.channels),
                 T.ToTensor(),
                 T.Normalize([0.485] * self.channels, [0.229] * self.channels)])
        return data_T

    def _prepare_data_loaders(self):
        train_set = datasets.ImageFolder(os.path.join(self.data_dir, self.TRAIN), transform=self.data_transforms(self.TRAIN))
        test_set = datasets.ImageFolder(os.path.join(self.data_dir, self.TEST), transform=self.data_transforms(self.TEST))
        valid_set = datasets.ImageFolder(os.path.join(self.data_dir, self.VAL), transform=self.data_transforms(self.VAL))

        self.class_names = train_set.classes
        self.class_to_idx = train_set.class_to_idx

        self.trainLoader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        self.validLoader = DataLoader(valid_set, batch_size=self.batch_size, shuffle=True)
        self.testLoader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)

    def show_sample_batch(self):
        images, labels = next(iter(self.trainLoader))
        print(images.shape)
        print(labels.shape)

# Usage
if __name__ == "__main__":
    data_dir = "../../dataset/chest_xray"
    data_loader = ChestXRayDataLoader(data_dir, channels=3)  # Example with 3 channels
    data_loader.show_sample_batch()
    print(data_loader.class_names)
    print(data_loader.class_to_idx)