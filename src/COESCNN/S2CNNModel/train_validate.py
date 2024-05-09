"""
@file train_validate.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Script contains functions for training and validating model.
"""

from imports import *

train_loss = [] # list of all train losses during training
validation_loss = [] # list of all validaton losses during training

"""
@brief Function for calculating distance on the unit circle between truth and output from model for yaw and roll angles.

@param output model output
@param truth ground truth

@return distance on the unit circle between truth and output
"""
def two_pi_range(output, truth):
    dif = torch.abs(output % (2 * math.pi) - truth)
    return torch.min(2 * math.pi - dif, dif)

"""
@brief Function for calculating distance on the unit circle between truth and output from model for pitch angle.

@param output model output
@param truth ground truth

@return distance on the unit circle between truth and output
"""
def one_pi_range(output, truth):
    dif = torch.abs(output % math.pi - truth)
    return torch.min(math.pi - dif, dif)

"""
@brief Main function for calculating distance on unit circle between truth and output.

@param output model output
@param truth ground truth

@return normalized output
"""
def normalize(out, truth):
    pitch = one_pi_range(out[:, 0], truth[:, 0])
    yaw = two_pi_range(out[:, 1], truth[:, 1])
    roll = two_pi_range(out[:, 2], truth[:, 2])
    out = torch.stack((pitch, yaw, roll), dim=1)
    return out

"""
@brief Training loop.

@param model model
@param dataloader dataset
@param loss_fn loss function
@param optimizer optimizer
@param device device - cuda or cpu
"""
def train(model, dataloader, loss_fn, optimizer, device):
    model.train()
    
    total_loss = 0
    total_batches = len(dataloader)
    correct_predictions_count = 0
    total_predictions_count = 0
    
    for batch in dataloader:        
        panorama_images, picture_images, labels = batch        
        panorama_images = panorama_images.to(device)
        picture_images = picture_images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        out = model(panorama_images, picture_images)
                                                           
        loss_p = loss_fn(normalize(out, labels), torch.zeros(picture_images.size(0), 3).to(device))
         
        diff = torch.abs(out[:, :2] - labels[:, :2])
        correct_predictions = diff < 0.15
        correct_predictions_count += correct_predictions.sum().item()
        total_predictions_count += np.prod(correct_predictions.shape).item()
        total_loss += loss_p.item()
                
        loss_p.backward()
        optimizer.step()

    avg_loss_p = total_loss / total_batches
    avg_accuracy = correct_predictions_count / total_predictions_count
    train_loss.append(avg_loss_p)
    
    print(f'Train Loss: {avg_loss_p:.4f}')
    print(f'Train Accuracy: {avg_accuracy:.4f}')
    
"""
@brief Validation loop.

@param model model
@param dataloader dataset
@param loss_fn loss function
@param optimizer optimizer
@param device device - cuda or cpu
"""
def validate(model, dataloader, loss_fn, device):
    model.eval()
    
    total_loss = 0
    total_batches = len(dataloader)
    correct_predictions_count = 0
    total_predictions_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            panorama_images, picture_images, labels = batch
            panorama_images = panorama_images.to(device)
            picture_images = picture_images.to(device)
            labels = labels.to(device)

            out = model(panorama_images, picture_images)
            
            loss_p = loss_fn(normalize(out, labels), torch.zeros(picture_images.size(0), 3).to(device))

            diff = torch.abs(out[:, :2] - labels[:, :2])
            correct_predictions = diff < 0.15
            correct_predictions_count += correct_predictions.sum().item()
            total_predictions_count += np.prod(correct_predictions.shape).item()
            total_loss += loss_p.item()

    avg_loss_p = total_loss / total_batches
    avg_accuracy = correct_predictions_count / total_predictions_count
    validation_loss.append(avg_loss_p)

    print(f'Validation Loss: {avg_loss_p:.4f}')
    print(f'Validation Accuracy: {avg_accuracy:.4f}')
    
### End of train_validate.py ###
    