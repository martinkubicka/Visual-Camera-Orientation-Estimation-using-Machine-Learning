"""
@file dataset.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Implementation of dataset for training camera orientation estimation.
"""

from imports import *

"""
@class SphereNetModelDataset

@brief Dataset for camera orientation estimation training and testing. 
       Dataset consists of:
            1) panorama in equirectangular format
            2) perspective taken on same place which is transformed to equirectangular format
            3) ground truth .csv file with yaw, roll, pitch 
"""
class SphereNetModelDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        self.data = []

        for csv_file in csv_files:
            image_id = os.path.splitext(csv_file)[0]
            panorama_image_path = os.path.join(data_dir, f"{image_id}_panorama.jpg")
            picture_image_path = os.path.join(data_dir, f"{image_id}_photo.jpg")
            
            panorama_exists = os.path.exists(panorama_image_path) 
            picture_exists = os.path.exists(picture_image_path)
            if not panorama_exists or not picture_exists:
                continue
            
            csv_data = []
            with open(os.path.join(data_dir, csv_file), newline='') as csvfile:
                try:
                    csvreader = csv.reader(csvfile)
                    for row in csvreader:
                        data = row[:3]
                        csv_data = [np.radians(float(data[0])), np.radians(float(data[1])), np.radians(float(data[2]))]
                        csv_data = [round(num, 4) for num in csv_data]
                except:
                    continue
                    

            self.data.append((panorama_image_path, picture_image_path, csv_data))

        if self.train:
            self.data, _ = train_test_split(self.data, test_size=0.2, random_state=42)
        else:
            _, self.data = train_test_split(self.data, test_size=0.2, random_state=42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        panorama_image_path, picture_image_path, csv_data = self.data[idx]
        panorama_image = Image.open(panorama_image_path)
        picture_image = Image.open(picture_image_path)
        
        panorama_image = panorama_image.convert('RGB')
        picture_image = picture_image.convert('RGB')
        
        if self.transform:
            panorama_image = self.transform(panorama_image)
            picture_image = self.transform(picture_image)
                                    
        return panorama_image, picture_image, torch.tensor(csv_data, dtype=torch.float32)

### End of dataset.py ###
