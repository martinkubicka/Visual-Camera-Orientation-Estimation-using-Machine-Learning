"""
@file angles.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Script with algorithms which gets pitch and yaw from probability map prediction.
"""

import torch
import numpy as np
import scipy.ndimage as ndimage

"""
@brief Function for getting pitch from probability map. 
       Algorithm gets middle between leftmost pixel and rightmost pixel.

@param pred probability map
@param height height probabilty map

@return -1 if no pixel found othwerwise middle between leftmost pixel and rightmost pixel.
"""
def get_pitch(pred, height):
    batch_preds = torch.sigmoid(pred)
    batch_preds = (batch_preds > 0.999999).float()    
    batch_preds = batch_preds[0, 0]
    batch_preds = torch.flip(batch_preds, [0])
    
    y_indices, _ = (batch_preds == 1).nonzero(as_tuple=True)

    if y_indices.nelement() > 0:
        highest_y_value = torch.max(y_indices)
        lowest_y_value = torch.min(y_indices)

        return ((highest_y_value + lowest_y_value) / 2) / (height / 180)
    else:
        return -1

"""
@brief Function for getting yaw from probability map. 
       Algorithm gets middle between the highest pixel and the lowest pixel.

@param pred probability map
@param width width probabilty map

@return -1 if no pixel found othwerwise middle between the highest pixel and the lowest pixel.
"""
def get_yaw(pred, width):
    batch_preds = torch.sigmoid(pred)
    batch_preds = (batch_preds > 0.999999).float()   
    batch_preds = batch_preds[0, 0]
    
    batch_preds_np = batch_preds.numpy()

    labeled_array, num_features = ndimage.label(batch_preds_np)

    if num_features == 0:
        return -1

    largest_component = 0
    max_size = 0
    for component in range(1, num_features + 1):
        component_size = (labeled_array == component).sum()
        if component_size > max_size:
            largest_component = component
            max_size = component_size

    x_indices = np.where(labeled_array == largest_component)[1]

    highest_x_value = np.max(x_indices)
    lowest_x_value = np.min(x_indices)
    
    return ((highest_x_value + lowest_x_value) / 2) / (width / 360)

    
### End of angles.py ###
