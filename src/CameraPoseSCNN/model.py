from sphere_cnn import SphereConv2D, SphereMaxPool2D
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import csv
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
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

            csv_data = []
            with open(os.path.join(data_dir, csv_file), newline='') as csvfile:
                csvreader = csv.reader(csvfile)
                for row in csvreader:
                    data = row[:3]
                    csv_data = [float(data[0]), float(data[1]), float(data[2])]

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
        
        panorama_image = panorama_image.convert('L')
        picture_image = picture_image.convert('L')
        
        new_size = (panorama_image.width // 10, panorama_image.height // 10)
        panorama_image = panorama_image.resize(new_size, Image.ANTIALIAS)
        picture_image = picture_image.resize(new_size, Image.ANTIALIAS)

        if self.transform:
            panorama_image = self.transform(panorama_image)
            picture_image = self.transform(picture_image)

        return panorama_image, picture_image, torch.tensor(csv_data, dtype=torch.float32)

class MergedModel(nn.Module):
    def __init__(self):
        super(MergedModel, self).__init__()
        
        # first scnn
        self.sphere_conv1 = SphereConv2D(1, 32, stride=1)
        self.sphere_pool1 = SphereMaxPool2D(stride=2)
        self.sphere_conv2 = SphereConv2D(32, 64, stride=1)
        self.sphere_pool2 = SphereMaxPool2D(stride=2)
        
        # second scnn
        self.sphere_conv3 = SphereConv2D(1, 32, stride=1)
        self.sphere_pool3 = SphereMaxPool2D(stride=2)
        self.sphere_conv4 = SphereConv2D(32, 64, stride=1)
        self.sphere_pool4 = SphereMaxPool2D(stride=2)
        
        self.fc = nn.Linear(2 * 64 * 51 * 103, 3)

    def forward(self, x1, x2, transform_data):
        # first scnn
        x1 = F.relu(self.sphere_pool1(self.sphere_conv1(x1)))
        x1 = F.relu(self.sphere_pool2(self.sphere_conv2(x1)))
        x1 = x1.view(-1, 64 * 51 * 103)
        
        # second scnn
        x2 = F.relu(self.sphere_pool3(self.sphere_conv3(x2)))
        x2 = F.relu(self.sphere_pool4(self.sphere_conv4(x2)))
        x2 = x2.view(-1, 64 * 51 * 103)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.fc(x)
        
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data1, data2, transform_data) in enumerate(train_loader):
        data1, data2, transform_data = data1.to(device), data2.to(device), transform_data.to(device)
        optimizer.zero_grad()
        output = model(data1, data2, transform_data)
        _, transform_data = transform_data.max(dim=1) # TODO ints
        
        loss = F.cross_entropy(output, transform_data)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data1), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data1, data2, transform_data in test_loader:
            data1, data2, transform_data = data1.to(device), data2.to(device), transform_data.to(device)
            output = model(data1, data2, transform_data) # TODO ints
            
            print(transform_data)
            print(output)
            
            _, transform_data = transform_data.max(dim=1)
            test_loss += F.cross_entropy(output, transform_data).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(transform_data.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    torch.manual_seed(1)
    kwargs = {}

    np.random.seed(1)
    device = torch.device("cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    data_dir = "/kaggle/input/dataimagesfrompanoramas/dataImagesFromPanoramas/" # TODO CHANGE ME

    # train set
    train_dataset = CustomDataset(data_dir, transform=transform, train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, **kwargs)

    # test set
    test_dataset = CustomDataset(data_dir, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, **kwargs)

    model = MergedModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, 100 + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        torch.save(model.state_dict(), 'model.pkl')

if __name__ == '__main__':
    main()

