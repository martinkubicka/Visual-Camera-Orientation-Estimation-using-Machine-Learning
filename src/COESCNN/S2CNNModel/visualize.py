"""
@file visualize.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Contains function for creating graphs with losses after each epoch.
"""

from imports import *

""""
@brief Function for creating graph with training and testing losses.

@param epochs Actual epoch.
@param train_loss Train losses list
@param validation_loss Validation losses list
"""
def create_loss_graph(epochs, train_loss, validation_loss):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_loss, label='Training Loss', marker='o')
    plt.plot(range(1, epochs + 1), validation_loss, label='Validation Loss', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend() 

    plt.savefig('./graphs/' + str(epochs) + '.jpg')
    plt.close()
    
### End of visualize.py ###
