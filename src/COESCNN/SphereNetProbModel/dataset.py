"""
@file dataset.py
@date 2023-22-12
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Implementation of dataset for training camera orientation estimation.
"""

from imports import *

"""
@class SphereNetSegModelDataset

@brief Dataset for camera orientation estimation training and testing. 
       Dataset consists of:
            1) panorama in equirectangular format
            2) perspective taken on same place which is transformed to equirectangular format
            3) ground truth as black and white image where white is truth
"""
class SphereNetSegModelDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        truth_files = [f for f in os.listdir(data_dir) if f.endswith("_truth.jpg")]
        
        self.data = []

        for truth_path in truth_files:
            image_id = os.path.splitext(truth_path)[0].replace('_truth', '')
            panorama_image_path = os.path.join(data_dir, f"{image_id}_panorama.jpg")
            picture_image_path = os.path.join(data_dir, f"{image_id}_photo.jpg")
            truth_image_path = os.path.join(data_dir, truth_path)
            
            panorama_exists = os.path.exists(panorama_image_path) 
            picture_exists = os.path.exists(picture_image_path)
            if not panorama_exists or not picture_exists:
                continue
                    
            self.data.append((panorama_image_path, picture_image_path, truth_image_path))

        if self.train:
            self.data, _ = train_test_split(self.data, test_size=0.2, random_state=42)
        else:
            _, self.data = train_test_split(self.data, test_size=0.2, random_state=42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        panorama_image_path, picture_image_path, truth_image_path = self.data[idx]
        panorama_image = Image.open(panorama_image_path)
        picture_image = Image.open(picture_image_path)
        truth_image = Image.open(truth_image_path)
        
        panorama_image = panorama_image.convert('L')
        picture_image = picture_image.convert('L')
        
        if self.transform:
            panorama_image = self.transform(panorama_image)
            picture_image = self.transform(picture_image)
            
            transform_truth = transforms.Compose([
                transforms.Resize((475, 950)),
                transforms.ToTensor(),
            ])
            truth_image = transform_truth(truth_image)            
                                    
        return panorama_image, picture_image, truth_image

### End of dataset.py ###
