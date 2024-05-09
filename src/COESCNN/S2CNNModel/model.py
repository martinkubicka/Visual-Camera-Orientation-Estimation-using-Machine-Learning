"""
@file model.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Implementation of SCNN model for training camera orientation estimation.
"""

from imports import *
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), './s2cnn/'))
sys.path.append(os.path.join(os.path.dirname(__file__), './s2cnn/soft/'))
sys.path.append(os.path.join(os.path.dirname(__file__), './s2cnn/utils/'))
from s2_conv import S2Convolution
from so3_conv import SO3Convolution
from so3_integrate import so3_integrate
from s2_grid import s2_near_identity_grid
from so3_grid import so3_near_identity_grid

"""
@class S2CNNModel

@brief Model for camera orientation estimating using s2cnn convolutions.
"""
class S2CNNModel(nn.Module):
    def __init__(self):
        super(S2CNNModel, self).__init__()
        
        # Features
        f1 = 16
        f2 = 32
        f3 = 64
        f4 = 128

        # Bandwidths
        b_in = 500
        b_l1 = 70
        b_l2 = 30
        b_l3 = 10
        b_l4 = 5

        # PANORAMA
        
        # Grids
        grid_s2 = s2_near_identity_grid(max_beta=np.pi/16)
        grid_so3_1 = so3_near_identity_grid(max_beta=np.pi/16)
        grid_so3_2 = so3_near_identity_grid(max_beta=np.pi/8)
        grid_so3_3 = so3_near_identity_grid(max_beta=np.pi/4)

        self.conv1_panorama = S2Convolution(
            nfeature_in=3,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2)    
        self.bn1_panorama = nn.BatchNorm3d(f1)
        self.dropout1_panorama = nn.Dropout(0.1) 

        self.conv2 = SO3Convolution(
            nfeature_in=f1,
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3_1)
        self.bn2 = nn.BatchNorm3d(f2)
        self.dropout2 = nn.Dropout(0.1) 
        
        self.conv3 = SO3Convolution(
            nfeature_in=f2, 
            nfeature_out=f3,
            b_in=b_l2,
            b_out=b_l3,
            grid=grid_so3_2)
        self.bn3 = nn.BatchNorm3d(f3)
        self.dropout3 = nn.Dropout(0.1) 
        
        self.conv4 = SO3Convolution(
            nfeature_in=f3, 
            nfeature_out=f4,
            b_in=b_l3,
            b_out=b_l4,
            grid=grid_so3_3)
        self.bn4 = nn.BatchNorm3d(f4)
        self.dropout4 = nn.Dropout(0.1) 
                
        # IMAGE

        # Grids
        grid_s2_cut = s2_near_identity_grid(max_beta=np.pi/16)
        grid_so3_1_cut = so3_near_identity_grid(max_beta=np.pi/16)
        grid_so3_2_cut = so3_near_identity_grid(max_beta=np.pi/8)
        grid_so3_3_cut = so3_near_identity_grid(max_beta=np.pi/4)
        
        self.conv1_cutout = S2Convolution(
            nfeature_in=3,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2_cut)
        
        self.bn1_cutout = nn.BatchNorm3d(f1)
        self.dropout1_cutout = nn.Dropout(0.1) 
        
        self.conv2_cut = SO3Convolution(
            nfeature_in=f1,  # * 2
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3_1_cut)
        
        self.bn2_cut = nn.BatchNorm3d(f2)
        self.dropout2_cut = nn.Dropout(0.1) 
        
        self.conv3_cut = SO3Convolution(
            nfeature_in=f2, 
            nfeature_out=f3,
            b_in=b_l2,
            b_out=b_l3,
            grid=grid_so3_2_cut)
        self.bn3_cut = nn.BatchNorm3d(f3)
        self.dropout3_cut = nn.Dropout(0.1) 
        
        self.conv4_cut = SO3Convolution(
            nfeature_in=f3, 
            nfeature_out=f4,
            b_in=b_l3,
            b_out=b_l4,
            grid=grid_so3_3_cut)
        self.bn4_cut = nn.BatchNorm3d(f4)
        self.dropout4_cut = nn.Dropout(0.1) 
        
        # Fully connected layers
        self.out_layer1 = nn.Linear(f4 * 2, f3)
        self.dropout = nn.Dropout(0.2) 
        self.out_layer2 = nn.Linear(f3, 3)  

    def forward(self, panorama, cutout):
        
        # Panorama
        x1 = self.conv1_panorama(panorama)           
        x1 = self.bn1_panorama(x1)
        x1 = F.relu(x1)
        x1 = self.dropout1_panorama(x1)
                
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)
        x1 = self.dropout2(x1)
        
        x1 = self.conv3(x1)
        x1 = self.bn3(x1)
        x1 = F.relu(x1)
        x1 = self.dropout3(x1)
        
        x1 = self.conv4(x1)
        x1 = self.bn4(x1)
        x1 = F.relu(x1)
        x1 = self.dropout4(x1)
        
        x1 = so3_integrate(x1)
        
        # Image
        x2 = self.conv1_cutout(cutout)
        x2 = self.bn1_cutout(x2)
        x2 = F.relu(x2)
        x2 = self.dropout1_cutout(x2)
        
        x2 = self.conv2_cut(x2)
        x2 = self.bn2_cut(x2)
        x2 = F.relu(x2)
        x2 = self.dropout2_cut(x2)
        
        x2 = self.conv3_cut(x2)
        x2 = self.bn3_cut(x2)
        x2 = F.relu(x2)
        x2 = self.dropout3_cut(x2)
        
        x2 = self.conv4_cut(x2)
        x2 = self.bn4_cut(x2)
        x2 = F.relu(x2)
        x2 = self.dropout4_cut(x2)
        
        x2 = so3_integrate(x2)
        
        # Concatenation
        x1 = torch.cat((x1, x2), dim=1)
        
        # Fully connected layers
        x = self.out_layer1(x1)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.out_layer2(x)
        
        return x
        
### End of model.py ###
