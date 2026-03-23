
import torch
import os
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import ToTensor, Resize, Compose, RandomAffine, ColorJitter
from models import SimpleCNN
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import itertools


def log_confusion_matrix(writer, cm, class_names, epoch):
    """
    Chuẩn hóa Confusion Matrix về % và đẩy lên Tensorboard.
    """
    # 1. Thực hiện chuẩn hóa (Normalization)
    # cm.sum(axis=1) tính tổng theo hàng (tổng số mẫu thật của mỗi class)
    # np.newaxis giúp chia theo đúng chiều của matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Thay thế NaN bằng 0 nếu có hàng nào toàn số 0

    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Epoch {epoch})")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 2. Ghi chỉ số % vào các ô
    threshold = cm_norm.max() / 2.
    for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
        color = "white" if cm_norm[i, j] > threshold else "black"
        # Hiển thị dưới dạng phần trăm với 2 chữ số thập phân: ví dụ 95.50%
        plt.text(j, i, f"{cm_norm[i, j]:.2f}", 
                 horizontalalignment="center", 
                 color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 3. Đẩy lên Tensorboard
    writer.add_figure('Confusion Matrix (Normalized)', figure, global_step=epoch)
    plt.close(figure)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--root', '-r', type=str, default='./data', help='root directory for the dataset')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch-size', '-b', type=int, default=64, help='batch size for training')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate for optimizer')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='momentum for optimizer')
    parser.add_argument('--image-size', '-i', type=int, default=32, help='size of the input images')
    parser.add_argument('--logging', '-l', type=str, default='tensorboard', help='tensorboard')
    parser.add_argument('--save-model', '-s', type=str, default='./models', help='save model')
    parser.add_argument('--checkpoints', '-c', type=str, default=None, help='save model')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    num_epochs = parse_args().epochs
    batch_size = parse_args().batch_size
    learning_rate = parse_args().learning_rate
    momentum = parse_args().momentum
    image_size = parse_args().image_size

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(parse_args().save_model):
        os.makedirs(parse_args().save_model)

    
    train_tranform = Compose([
        RandomAffine(degrees=(5,5) , translate=(0.15, 0.15), scale=(0.85, 1.15), shear=5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        Resize((image_size, image_size)),
        ToTensor()
    ])
    test_tranform = Compose([
        Resize((image_size, image_size)),
        ToTensor()
    ])



    # step1: load dataset
    train_dataset = CIFAR10(root='./data', train=True, transform=train_tranform)
    
    image, _ = train_dataset[5]
    print(image.shape)
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    exit(0)

    training_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    test_dataset = CIFAR10(root='./data', train=False, transform=test_tranform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    

    # step2: build model
    model = SimpleCNN(num_classes=10).to(device)
    writer = SummaryWriter(log_dir=parse_args().logging)
    # step3: define loss function 
    criterion = nn.CrossEntropyLoss()

    # step4: define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    start_epoch = 0
    best_accuracy = 0.0
    if parse_args().checkpoints:
        checkpoint = torch.load(parse_args().checkpoints)

        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_accuracy = checkpoint['best_accuracy']
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_train_loss = 0.0
        train_bar = tqdm(training_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", 
                     unit="batch", colour="cyan", leave=False)
        for iter, (images, labels) in enumerate(train_bar):
            
            #to device GPU
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(training_loader) + iter)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{iter+1}/{len(training_loader)}], Loss: {loss.item():.4f}")

            # Backward pass and optimization
            optimizer.zero_grad() #  updata gradients=0
            loss.backward()       # tính toán gradients
            optimizer.step()      # cập nhật weights

        # validation
        model.eval()
        running_val_loss = 0.0
        all_predictions = []
        all_labels = []
        val_bar = tqdm(test_loader, desc="Validating", unit="batch", leave=False, colour="green")
        for iter, (images, labels) in enumerate(val_bar):
            all_labels.extend(labels)
            #to device GPU
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                indices = torch.argmax(outputs.cpu(), dim=1)
                all_predictions.extend(indices)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                # print(f"Epoch [{epoch+1}/{num_epochs}], Iteration [{iter+1}/{len(test_loader)}], Loss: {loss.item():.4f}")
        all_predictions = [pred.item() for pred in all_predictions]
        all_labels = [label.item() for label in all_labels]
        # --- SUMMARY: PRINT REPORT ---
        avg_train_loss = running_train_loss / len(training_loader)
        avg_val_loss = running_val_loss / len(test_loader)
        acc = accuracy_score(all_labels, all_predictions)
        # avg_loss = running_loss / len(training_loader)
       
        writer.add_scalar('Val/Accuracy', acc, epoch)
        log_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), test_dataset.classes, epoch)
        print(f"✅ Epoch [{epoch+1}/{num_epochs}] Hoàn tất | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {acc*100:.2f}%")
        
        # Chỉ in classification report chi tiết ở các epoch quan trọng (ví dụ mỗi 5 epoch hoặc epoch cuối)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            print("\nDetailed Report:")
            print(classification_report(all_labels, all_predictions, target_names=test_dataset.classes))
            print("-" * 60)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy
        }
        torch.save(checkpoint, f"{parse_args().save_model}/model_last_cnn.pth")
        if best_accuracy < acc:
            best_accuracy = acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_accuracy': best_accuracy

            }
            torch.save(checkpoint, f"{parse_args().save_model}/best_model_cnn.pth")
        







