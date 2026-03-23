from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        del self.model.fc
        self.fc1 = nn.Linear(in_features=2048, out_features=num_classes)
    
    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    

if __name__ == "__main__":
    num_classes = 10
    model = ResNet50(num_classes=num_classes)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)