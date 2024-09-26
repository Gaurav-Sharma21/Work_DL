import os 
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image 


class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        #Looping through subfolders
        for label in os.listdir(data_dir):
            folder_path = os.path.join(data_dir, label)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.endswith(".jpg"):
                        self.images.append(os.path.join(folder_path, img_name))
                        self.labels.append(0 if label == "no_tumor" else 1)
        print(f"Found {len(self.images)} images in {data_dir}")
    
    def get_label(self, img_name):
        return 0 if "no_tumor" in img_name else 1
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)


        return image, label