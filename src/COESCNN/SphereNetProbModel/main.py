"""
@file main.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Main function for SphereNet model.
"""

from imports import *
from dataset import SphereNetSegModelDataset
from model import SphereNetSegModel
from train_validate import train_loss, validation_loss, train, validate
from visualize import create_loss_graph

epochs = 1000

"""
@class DiceLoss

@brief DiceLoss implementation. Inspired by: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
"""
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def main():    
    torch.cuda.empty_cache()
    
    transform = transforms.Compose([
        transforms.Resize((475, 950)),
        transforms.ToTensor(),
    ])    
    
    data_dir = "./dataset" # dataset directory
    
    batch_size = 4
    
    train_dataset = SphereNetSegModelDataset(data_dir, transform=transform, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    test_dataset = SphereNetSegModelDataset(data_dir, transform=transform, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
    model = SphereNetSegModel()
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=epochs)
    
    loss_fn = DiceLoss().to(device)
    
    ## Uncomment if fine-tuning or reloading model
    # checkpoint = torch.load('model.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # loss_fn.load_state_dict(checkpoint['criterion_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        train(model, train_dataloader, loss_fn, optimizer, device)
        validate(model, test_dataloader, loss_fn, device)
        torch.save({
            'criterion_state_dict': loss_fn.state_dict(),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, 'model.pth')
        
        print('-' * 10)
        
        create_loss_graph(epoch + 1, train_loss, validation_loss)
        torch.cuda.empty_cache()
        scheduler.step()
        
if __name__ == '__main__':
    main()

### End of main.py ###
