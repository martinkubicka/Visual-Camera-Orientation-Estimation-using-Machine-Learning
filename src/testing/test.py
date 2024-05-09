"""
@file test.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Main script for testing accuracy for pitch and yaw angles.
"""

import torch
import sys
import os
sys.path.append(os.path.abspath('../COESCNN/SphereNetProbModel/'))
from model import SphereNetSegModel
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from angles import *
import csv

data_dir = './dataset' # dataset directory
model_name = 'model.pth' # model path
width = 950 # input/output width
height = 475 # input/output height

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
@brief Function for initializing and loading trained model.

@param device device
@param model_path path to model

@return model
"""
def init_model(device, model_path):
    model = SphereNetSegModel().to(device)
        
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'], strict=False)
    model.eval()
    return model

"""
@brief Function for preprocessing inputs

@param panorama panorama path
@param image image path

@return processed tuple (panorama, image)
"""
def preprocess_image(panorama, image):
    transform = transforms.Compose([
        transforms.Resize((height, width)), 
        transforms.ToTensor(),
    ])   
    
    panorama = Image.open(panorama)
    perspective = Image.open(image)
    
    panorama = panorama.convert('L')
    perspective = perspective.convert('L')
        
    panorama = transform(panorama).unsqueeze(0).to(device)
    perspective = transform(perspective).unsqueeze(0).to(device)
    
    return panorama, perspective

"""
@brief Main fuction for getting predicted pitch for pair of inputs

@param panorama panorama path
@param image image path
@param model model

@return predicted pitch
"""
def get_pitch_pred(panorama, image, model):        
    panorama, perspective = preprocess_image(panorama, image)
    
    with torch.no_grad():
        out = model(panorama, perspective)        
        pitch = get_pitch(out, height)     

    return pitch
    
"""
@brief Main fuction for getting predicted yaw for pair of inputs

@param panorama panorama path
@param image image path
@param model model

@return predicted yaw
"""
def get_yaw_pred(panorama, image, model):            
    panorama, perspective = preprocess_image(panorama, image)
    
    with torch.no_grad():
        out = model(panorama, perspective)
        yaw = get_yaw(out, width)     

    return yaw

"""
@brief Function for loading dataset

@param data_dir dataset directory path

@return list of lists [panorama_path, picture_path, pitch, yaw]
"""
def load_data(data_dir):
    d = []
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

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
                    csv_data = [float(data[0]), float(data[1]), float(data[2])]
                    csv_data = [round(num, 4) for num in csv_data]
            except:
                continue
                
        d.append([panorama_image_path, picture_image_path, csv_data[0], csv_data[1]])
        
    return d
    
def main(opt):            
    correct10 = 0
    correct15 = 0
    correct20 = 0
    counter = 1
    error = 0

    d = load_data(data_dir)
    
    all_len = len(d)
            
    model = init_model(device, model_name)
    
    for i in d:     
        if opt == 1:
            pred = get_pitch_pred(i[0], i[1], model)
            truth = i[2]
        else: 
            pred = get_yaw_pred(i[0], i[1], model)
            truth = i[3]
            
        if (pred == -1):
            continue
        if (abs(pred.item() - truth) <= 10.0):
            correct10 += 1
        if (abs(pred.item() - truth) <= 15.0):
            correct15 += 1
        if (abs(pred.item() - truth) <= 20.0):    
            correct20 += 1
            
        error += abs(pred.item() - truth)
    
        print(counter, correct10, correct15, correct20)
        print((correct10 / all_len) * 100)
        print((correct15 / all_len) * 100)
        print((correct20 / all_len) * 100)
        print(error / counter)
        print('------')
    
        counter += 1

if __name__ == '__main__':
    while True:
        opt = int(input("1) Test pitch angle 2) Test Yaw Angle: "))
        if opt == 1 or opt == 2:
            break
    main(opt)

### End of test.py ###
