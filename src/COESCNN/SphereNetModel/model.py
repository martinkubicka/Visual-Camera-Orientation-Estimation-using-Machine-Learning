"""
@file model.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Implementation of SCNN model using SphereNet for training camera orientation estimation.
"""

from imports import *

"""
@class SphereNetModel

@brief Model for camera orientation estimating using SphereNet convolutions and MaxPools.
"""
class SphereNetModel(nn.Module):
    def __init__(self):
        super(SphereNetModel, self).__init__()
        
        # Features in/out
        f1 = 8
        f2 = 16
        f3 = 32
        
        # Conv blocks
        self.conv1_panorama = SphereConv2D(3, f1, stride=1)
        self.pool1_panorama = SphereMaxPool2D(stride=2) 
        
        self.conv2_panorama = SphereConv2D(f1, f2, stride=1)
        self.pool2_panorama = SphereMaxPool2D(stride=2)
        
        self.conv3_panorama = SphereConv2D(f2, f3, stride=1)
        self.pool3_panorama = SphereMaxPool2D(stride=2)
                            
        # Fully connected layers        
        self.out_layer1 = nn.Linear(f3 * 192 * 96 * 2, 2048) 
        self.out_layer2 = nn.Linear(2048, 2)

    def forward(self, panorama, cutout):               
        # Panorama        
        x1 = self.conv1_panorama(panorama)         
        x1 = F.relu(x1)
        x1 = self.pool1_panorama(x1)
                
        x1 = self.conv2_panorama(x1)
        x1 = F.relu(x1)
        x1 = self.pool2_panorama(x1)
        
        x1 = self.conv3_panorama(x1)
        x1 = F.relu(x1)
        x1 = self.pool3_panorama(x1)
                
        x1 = x1.view(x1.size(0), -1)
        
        # Image
        x2 = self.conv1_panorama(cutout)
        x2 = F.relu(x2)
        x2 = self.pool1_panorama(x2)
        
        x2 = self.conv2_panorama(x2)
        x2 = F.relu(x2)
        x2 = self.pool2_panorama(x2)
        
        x2 = self.conv3_panorama(x2)
        x2 = F.relu(x2)
        x2 = self.pool3_panorama(x2)
        
        x2 = x2.view(x2.size(0), -1)
        
        # Concatenation
        x = torch.cat((x1, x2), dim=1)
         
        # Fully connected layers
        x = self.out_layer1(x)
        x = F.relu(x)
        x = self.out_layer2(x)
                
        return x
        
### End of model.py ###
