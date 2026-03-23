import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class Mydataset(Dataset):
    # def __init__(self, csv_file, img_dir, transform=None):
    #     """
    #     csv_file: File chứa danh sách tên ảnh và nhãn (label)
    #     img_dir: Thư mục chứa ảnh
    #     transform: Các bước tiền xử lý ảnh (Resize, Augmentation)
    #     """

    #     self.data_info = pd.read_csv(csv_file)
    #     self.img_dir = img_dir
    #     self.transform = transform

    # def __len__(self):
    #     return 0

    # def __getitem__(self, idx):
    #     return None
    
    def __init__(self, root, train=True):
        """
        csv_file: File chứa danh sách tên ảnh và nhãn (label)
        img_dir: Thư mục chứa ảnh
        transform: Các bước tiền xử lý ảnh (Resize, Augmentation)
        """
        mode = 'train' if train else 'test'
    
        self.root = os.path.join(root, mode)
        self.categories = os.listdir(self.root)

        self.images_path = []
        self.labels = []
        
        for i, category in enumerate(self.categories):
            category_path = os.path.join(self.root, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    self.images_path.append(os.path.join(category_path, img_name))
                    self.labels.append(i)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        # image = Image.open(img_path)
        image = cv2.imread(img_path)
        
        label = self.labels[idx]

        return image, label


if __name__ == "__main__":
    dataset = Mydataset(root='./data/buttlefly', train=True)
    print(f"Number of samples: {len(dataset)}")
    img, label = dataset[8000]

    plt.imshow(img) 
    print(f"Label: {label}")
    plt.show()