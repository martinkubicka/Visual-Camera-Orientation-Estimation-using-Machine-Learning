"""
@file train_validate.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Script contains functions for training and validating model.
"""

from imports import *

train_loss = []
validation_loss = []

"""
@brief Function for calculating dice score

@param pred predictions
@param targs ground truths

@return dice score
"""
def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

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
    dices = []
    
    for batch in dataloader:        
        panorama_images, picture_images, labels = batch        
        panorama_images = panorama_images.to(device)
        picture_images = picture_images.to(device)
        labels = labels.to(device)
        
        out = model(panorama_images, picture_images)
                
        dice = dice_score(out, labels)
        dices.append(dice)
                        
        correct_predictions_count += torch.sum(out == labels)
        total_predictions_count += labels.numel()
                                                           
        loss_p = loss_fn(out, labels)
                
        loss_p.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss_p.item()
    
    avg_loss_p = total_loss / total_batches
    train_loss.append(avg_loss_p)
    
    d_list = [t.cpu().numpy() for t in dices]
    
    print(f'Train Loss: {avg_loss_p:.4f}')
    print(f'Dice score: {np.mean(d_list) :.4f}')
    
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
    dices = []
    
    with torch.no_grad():
        for batch in dataloader:
            panorama_images, picture_images, labels = batch
            panorama_images = panorama_images.to(device)
            picture_images = picture_images.to(device)
            labels = labels.to(device)

            out = model(panorama_images, picture_images)
            
            dice = dice_score(out, labels)
            dices.append(dice)
            
            correct_predictions_count += torch.sum(out == labels)
            total_predictions_count += labels.numel()
            
            loss_p = loss_fn(out, labels)
            total_loss += loss_p.item()

    avg_loss_p = total_loss / total_batches
    validation_loss.append(avg_loss_p)
    
    d_list = [t.cpu().numpy() for t in dices]

    print(f'Validation Loss: {avg_loss_p:.4f}')
    print(f'Dice score: {np.mean(d_list) :.4f}')
    
### End of train_validate.py ###
