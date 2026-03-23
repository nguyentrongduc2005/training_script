from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Resize, Compose
import matplotlib.pyplot as plt
from dataset import Mydataset

# class DataLoader:

if __name__ == "__main__":
    tranform = Compose([
        # Resize((100, 100)),
        ToTensor()
    ])
    dataset = CIFAR10(root='./data', train=True, download=True, transform=tranform)
    img, label = dataset[0]
    # print(f"Image shape: {img.size}, Label: {label}")
    # plt.imshow(img)
    # plt.show()

    training_loader = DataLoader(
        dataset=dataset,
        batch_size=64, 
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    for images, labels in training_loader:
        print(f"Batch of images shape: {images.shape}, Batch of labels shape: {labels.shape}")
        
