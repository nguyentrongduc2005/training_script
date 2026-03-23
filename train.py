import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import ToTensor, Resize, Compose
from models import SimpleNN
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    num_epochs = 100
    tranform = Compose([
        ToTensor()
    ])

    # step1: load dataset
    train_dataset = CIFAR10(root='./data', train=True, transform=tranform)
    training_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64, 
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataset = CIFAR10(root='./data', train=False, transform=tranform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # step2: build model
    model = SimpleNN(num_classes=10)
    
    # step3: define loss function 
    criterion = nn.CrossEntropyLoss()

    # step4: define optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(num_epochs):
        model.train()
        for iter, (images, labels) in enumerate(training_loader):
            # Move data to GPU if available
            if torch.cuda.is_available():
                images = images.to('cuda')
                labels = labels.to('cuda')
                model.to('cuda')

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{iter+1}/{len(training_loader)}], Loss: {loss.item():.4f}")

            # Backward pass and optimization
            optimizer.zero_grad() #  updata gradients=0
            loss.backward()       # tính toán gradients
            optimizer.step()      # cập nhật weights

        # validation
        model.eval()
        all_predictions = []
        all_labels = []
        for iter, (images, labels) in enumerate(test_loader):
            all_labels.extend(labels)
            if torch.cuda.is_available():
                images = images.to('cuda')
                labels = labels.to('cuda')
                model.to('cuda')
            with torch.no_grad():
                outputs = model(images)
                indices = torch.argmax(outputs.cpu(), dim=1)
                all_predictions.extend(indices)
                loss = criterion(outputs, labels)
                # print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{iter+1}/{len(test_loader)}], Loss: {loss.item():.4f}")
        all_predictions = [pred.item() for pred in all_predictions]
        all_labels = [label.item() for label in all_labels]
        print(f"epoch: {epoch + 1}")
        print(classification_report(all_labels, all_predictions, target_names=test_dataset.classes))
        






