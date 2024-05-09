"""
@file main.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Main function for S2CNN model.
"""

import collections
import collections.abc
if not hasattr(collections, 'Iterable'):
    collections.Iterable = collections.abc.Iterable
from imports import *
from dataset import S2CNNDataset
from model import S2CNNModel
from train_validate import train_loss, validation_loss, train, validate
from visualize import create_loss_graph

epochs = 1000

lambda_poly = lambda epoch: (1 - epoch / epochs) ** 1.0

def main():    
    torch.cuda.empty_cache()
    
    transform = transforms.Compose([
        transforms.Resize((1000, 1000)),
        transforms.ToTensor(),
    ])    
    
    data_dir = "./dataset" # dataset directory
    
    batch_size = 16
    
    train_dataset = S2CNNDataset(data_dir, transform=transform, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
    test_dataset = S2CNNDataset(data_dir, transform=transform, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
    model = S2CNNModel()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_poly)
    
    loss_fn = nn.MSELoss().to(device)
    
    ## Uncomment if fine-tuning or reloading model
    # checkpoint = torch.load('model.pth', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # loss_fn_p.load_state_dict(checkpoint['criterion_state_dict'])
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
