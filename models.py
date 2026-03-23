import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Resize, Compose


class SimpleNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
                nn.Linear(in_features=3*32*32, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=1024),
                nn.ReLU(),
                nn.Linear(in_features=1024, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=num_classes),
                nn.ReLU()
        )
        # self.fc1 = nn.Linear(in_features=3*32*32, out_features=256)
        
      

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
       
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = self.make_block(in_channels=3, out_channels=32)
        self.conv2 = self.make_block(in_channels=32, out_channels=64)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64*8*8, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256, out_features=num_classes)
            # nn.Softmax(dim=1)
        )
    def make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    num_classes = 10
    model = SimpleCNN(num_classes=num_classes)
    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    print(output.shape)
    # print(output)