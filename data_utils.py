import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class ImageFolderDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.cls_dict = self.build_cls_to_idx()
        self.file_paths, self.labels, self.count_dict = self.load_dataset()
        self.transform = transform

    def build_cls_to_idx(self) -> dict:
        self.classes = sorted(os.listdir(self.data_dir)) # torchvision.datasets.ImageFolder와 동일하게 구현하려면 sorted 필요
        cls_dict = dict()
        for idx, cls_ in enumerate(self.classes):
            cls_dict[cls_] = idx
        return cls_dict
    
    def load_dataset(self):
        file_paths, labels = list(), list()
        count_dict = dict()
        for cls_ in self.cls_dict:
            temp_files = list(map(lambda x: os.path.join(self.data_dir, cls_, x), os.listdir(os.path.join(self.data_dir, cls_))))
            num = len(temp_files)
            count_dict[cls_] = num
            file_paths += temp_files
            labels += [self.cls_dict[cls_] for _ in range(num)]
        return file_paths, labels, count_dict

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, label = self.file_paths[idx], self.labels[idx]
        img = Image.open(file_path).convert('RGB')
        img = self.transform(image=np.array(img))['image']
        return img, label, file_path