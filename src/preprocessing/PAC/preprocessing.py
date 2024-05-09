"""
@file preprocessing.py
@date 2023-22-12
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Preprocessing of images scraped with datasetGenerator.py.
"""

from PIL import Image
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.blackEdgeRemover import cropPanorama
import cv2
import lib.multi_Perspec2Equirec as m_P2E
import csv
import random
import numpy as np
import subprocess
import traceback

actualDirectory = os.path.dirname(os.path.abspath(__file__))

panoramaHeight = 2048
panoramaWidth = 4096
panoramaResolution = (panoramaWidth, panoramaHeight)

NUMBER_OF_GENERATED_IMAGES_FROM_PANORAMA = 1

input_path = "./in" # input path
output_path = "./dataset" # ouput path

"""
@brief Function for getting panoramas file names from generated dataset.

@return list of panorama file names
"""
def getData(in_dir):
    dataFolder = os.path.join(actualDirectory, in_dir)
    result = []

    for filename in os.listdir(dataFolder):
        if filename.endswith("_panorama.jpg"):
            result.append(filename)
    return result

"""
@brief Function for generating random parameters needed for camera orientation estimation.

@return FOV, pitch, yaw, roll, height, width
"""
def randomGenerate():
    FOV = random.uniform(50, 151)
    yaw = random.uniform(0, 361)
    pitch = random.uniform(0, 181)
    roll = random.uniform(0, 361)
    
    # images in the middle of pitch will be generated with higher probability
    weights = {'low': 1, 'mid': 8, 'high': 1}
    cum_weights = [weights['low'], weights['low'] + weights['mid'], sum(weights.values())]
    segment = random.choices(['low', 'mid', 'high'], cum_weights=cum_weights, k=1)[0]
    
    if segment == 'low':
        pitch = random.uniform(0, 41)
    elif segment == 'mid':
        pitch =random.uniform(41, 141)
    else:
        pitch = random.uniform(141, 181)
    
    height = 960
    width = 1920

    return FOV, pitch, yaw, roll, height, width

"""
@brief Function for resizing image.

@param inputPath image input path
@param outputPath image output path
"""
def preprocessPanorama(inputPath, outputPath):
    panorama = Image.open(inputPath)
    resizedPanorama = panorama.resize(panoramaResolution)
    resizedPanorama.save(outputPath)

"""
@brief Function for getting perpesctive from panorama.

@param panoramaPath path to panorama
@param outputPath output path where perspective will be saved
@param FOV field of view
@param pitch pitch
@param yaw yaw
@param roll roll
@param height height
@param width width
"""
def getPerspective(panoramaPath, outputPath, FOV, yaw, pitch, roll, height, width):    
    command = [
        'convert360', '--convert', 'e2p', 
        '--i', panoramaPath, 
        '--o', outputPath, 
        '--w', str(width), 
        '--h', str(height), 
        '--u_deg', str(yaw - 180.0), 
        '--v_deg', str(pitch - 90.0), 
        '--in_rot_deg', "0", # change this to str(-roll), if roll is wanted
        '--h_fov', str(FOV)
    ]
    
    subprocess.run(command, capture_output=True, text=True)

"""
@brief function for transforming perspective to equirectangular format.

@param outputPath path to perspective (formatted image will be saved to same path)
@param FOV field of view
@param pitch pitch
@param yaw yaw
@param roll roll
"""
def perspectiveToEquirectangular(outputPath, FOV, yaw, pitch):
    equ = m_P2E.Perspective([outputPath],
                            [[FOV, yaw-180, pitch]])

    img = equ.GetEquirec(panoramaHeight, panoramaWidth)
    cv2.imwrite(outputPath, img)

"""
@brief function for saving ground truth to .csv file.

@param pitch pitch
@param yaw yaw
@param roll roll
@param path path where .csv file will be saved
"""
def saveToCsv(pitch, yaw, roll, FOV, path):
    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([pitch, yaw, roll, FOV])

"""
@brief Function for bluring image.

@param imagePath path to image
"""
def blur(imagePath):
    image = cv2.imread(imagePath)
    blurredImage = cv2.GaussianBlur(image, (25, 25), 0)
    cv2.imwrite(imagePath, blurredImage)

"""
@brief Function adding contrast to image.

@param imagePath path to image
"""
def contrast(imagePath):
    image = cv2.imread(imagePath)
    contrasted = cv2.convertScaleAbs(image, alpha=1.5, beta=10)
    cv2.imwrite(imagePath, contrasted)

"""
@brief Function for adding noise to image.

@param imagePath path to image
"""
def noise(imagePath):
    img = cv2.imread(imagePath)
    noise = np.zeros(img.shape, np.uint8)
    cv2.randn(noise, 0, 180)
    noisy_img = cv2.add(img, noise)
    cv2.imwrite(imagePath, noisy_img)

"""
@brief Function for generating random augmentation.

@param imagePath path to image
"""
def augmentation(imagePath):
    augmentationFunctions = [blur, contrast, noise]

    augmentationFunction = random.choice(augmentationFunctions)
    augmentationFunction(imagePath)

"""
@brief main function for preprocessing

@param data panorama file names
"""

def preprocess(data):
    for i in data:
        try:            
            inputPanoramaPath = os.path.join(actualDirectory, input_path + '/' + i)

            # check if not just black image or crop black borders if needed
            if not cropPanorama(inputPanoramaPath):
                continue
            
            for index in range(0, NUMBER_OF_GENERATED_IMAGES_FROM_PANORAMA):
                outputPanoramaPath = os.path.join(actualDirectory, output_path + '/' + i.replace("_panorama", str(index) + "_panorama"))
                perspectiveOutputPath = outputPanoramaPath.replace("_panorama", "_photo")
                
                try:
                    preprocessPanorama(inputPanoramaPath, outputPanoramaPath)
                except:
                    print(traceback.format_exc())

                FOV, pitch, yaw, roll, height, width = randomGenerate()

                getPerspective(outputPanoramaPath, perspectiveOutputPath, FOV, yaw, pitch, roll, height, width)
                
                augmentation(perspectiveOutputPath)
                
                perspectiveToEquirectangular(perspectiveOutputPath, FOV, 180.0, 0.0)

                saveToCsv(pitch, yaw, roll, FOV, os.path.join(actualDirectory, output_path + '/' + i.replace("_panorama.jpg", str(index) + ".csv")))
        except KeyboardInterrupt:
            exit()
        except:
            print(traceback.format_exc())
            continue

if __name__ == '__main__':
    print("Preprocessing started")
    data = getData(input_path)
    preprocess(data)

### End of preprocessing.py ###
