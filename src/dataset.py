import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

class PetDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=256, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = (mask != 2).astype(np.float32)
        
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        
        if self.augment and np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
            
        image = (image / 255.0 - MEAN) / STD
        
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()
            
        return image, mask


def get_splits(root_dir, val_size=0.15, seed=42, img_size=256):
    images_dir = os.path.join(root_dir, 'images')
    masks_dir = os.path.join(root_dir, 'annotations', 'trimaps')
    trainval_file = os.path.join(root_dir, 'annotations', 'trainval.txt')
    
    with open(trainval_file, 'r') as f:
        raw_names = [line.split()[0] for line in f if line.strip()]
        
    valid_names = []
    removed_count = 0

    for name in raw_names:
        mask_path = os.path.join(masks_dir, f"{name}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is not None and (mask != 2).any():
            valid_names.append(name)
        else:
            removed_count += 1

    print(f"empty masks removed: {removed_count}")
        
    image_paths = [os.path.join(images_dir, f"{n}.jpg") for n in valid_names]
    mask_paths = [os.path.join(masks_dir, f"{n}.png") for n in valid_names]
    
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, 
        mask_paths, 
        test_size=val_size, 
        random_state=seed, 
        shuffle=True
    )
    
    train_dataset = PetDataset(train_imgs, train_masks, img_size=img_size, augment=True)
    val_dataset = PetDataset(val_imgs, val_masks, img_size=img_size, augment=False)
    
    return train_dataset, val_dataset