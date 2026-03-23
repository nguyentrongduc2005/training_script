from argparse import ArgumentParser

import torch

from models import SimpleCNN
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Resize, Compose
from torchsummary import summary





def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--image-size', '-i', type=int, default=32, help='size of the input images')
    parser.add_argument('--image-path', '-p', type=str, default=None, help='size of the input images')
    parser.add_argument('--checkpoints', '-c', type=str, default="./models/best_model_cnn.pth", help='save model')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    image_size = parse_args().image_size
    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleCNN(num_classes=10).to(device)

    summary(model, input_size=(3, image_size, image_size))
    # exit(0)
    if parse_args().checkpoints:
        checkpoint_ = torch.load(parse_args().checkpoints)
        model.load_state_dict(checkpoint_['model_state_dict'])
    else:
        print("No checkpoint found. Please provide a valid checkpoint path.")
        exit(1)

    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor()
    ])
    image_path = None
    if parse_args().image_path:
        image_path = parse_args().image_path
    else:
        print("No image path provided. Please provide a valid image path.")
        exit(1)
    
    image = Image.open(image_path).convert('RGB')
    
    original_image = image.copy()
    image = transform(image)
    image = image.unsqueeze(0).to(device)
    
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        model.eval()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
       
        probabilities = softmax(output)
        print(probabilities[0][predicted.item()].item())
        # print(f"Predicted probabilities: {probabilities}")

        plt.imshow(original_image)
        plt.title(f"Predicted class: {predicted.item()} probability: {probabilities[0][predicted.item()].item():.4f}")
        plt.show()
