import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class JPGImageDataset(Dataset):
    def __init__(self, image_dir, image_size=(224, 224), num_images=None, transform=None):
        self.image_dir = image_dir
        self.image_size = image_size
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
        
        if num_images:
            self.image_files = self.image_files[:num_images]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5)
            ])(image)
        
        return image

# Data CLASS#
class GET_DATALOADER:
    def get_jpg_dataloader(self, image_dir, image_size=(224, 224), \
        batch_size=32, shuffle=True, num_images=20000):
        dataset = JPGImageDataset(image_dir, image_size, num_images=num_images)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=3,pin_memory=True),dataset
