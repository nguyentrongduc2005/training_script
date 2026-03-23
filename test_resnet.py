from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
import torch
from torchsummary import summary


if __name__ == "__main__":
# Load the pre-trained ResNet50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(in_features=2048, out_features=10)


    for name, param in model.named_parameters():
        if ("fc" not in name) and ("layer4." not in name):
            param.requires_grad = False
        # print(name, param.requires_grad)

    summary(model, input_size=(3, 224, 224))

