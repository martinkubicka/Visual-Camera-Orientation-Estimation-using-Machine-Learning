from PIL import Image
import os
from blackEdgeRemover import cropPanorama
import cv2
import lib.Equirec2Perspec as E2P
import lib.multi_Perspec2Equirec as m_P2E
import csv
import math
import random
import numpy as np

panoramaHeight = 2048
panoramaWidth = 4096
panoramaResolution = (panoramaWidth, panoramaHeight)

def getData():
    dataFolder = "../data/panoramas"
    result = []

    for filename in os.listdir(dataFolder):
        if filename.endswith("_panorama.jpg"):
            result.append(filename)
    return result

def randomGenerate():
    FOV = random.uniform(50, 151)
    yaw = random.uniform(0, 361)
    pitch = random.uniform(-90, 91)
    roll = random.uniform(0, 361)

    height = 960
    width = 1920

    return FOV, pitch, yaw, roll, height, width

def preprocessPanorama(inputPath, outputPath):
    panorama = Image.open(inputPath)
    resizedPanorama = panorama.resize(panoramaResolution)
    resizedPanorama.save(outputPath)

def getPerspective(panoramaPath, outputPath, FOV, pitch, yaw, roll, height, width):
    equ = E2P.Equirectangular(panoramaPath)
    img = equ.GetPerspective(FOV, yaw, pitch, roll, height, width)
    cv2.imwrite(outputPath, img)

def perspectiveToEquirectangular(outputPath, FOV, pitch, yaw, roll):
    equ = m_P2E.Perspective([outputPath],
                            [[FOV, yaw-180, pitch, roll]])

    img = equ.GetEquirec(panoramaHeight, panoramaWidth)
    cv2.imwrite(outputPath, img)

def saveToCsv(pitch, yaw, roll, path):
    with open(path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([math.radians(pitch), math.radians(yaw), math.radians(roll)])

def blur(imagePath):
    image = cv2.imread(imagePath)
    blurredImage = cv2.GaussianBlur(image, (25, 25), 0)
    cv2.imwrite(imagePath, blurredImage)

def contrast(imagePath):
    image = cv2.imread(imagePath)
    contrasted = cv2.convertScaleAbs(image, alpha=1.5, beta=10)
    cv2.imwrite(imagePath, contrasted)

def noise(imagePath):
    img = cv2.imread(imagePath)
    noise = np.zeros(img.shape, np.uint8)
    cv2.randn(noise, 0, 180)
    noisy_img = cv2.add(img, noise)
    cv2.imwrite(imagePath, noisy_img)

def augmentation(imagePath):
    augmentationFunctions = [blur, contrast, noise]

    augmentationFunction = random.choice(augmentationFunctions)
    augmentationFunction(imagePath)

def preprocess(data):
    for i in data:
        try:            
            inputPanoramaPath = "../data/panoramas/" + i
            outputPanoramaPath = "./dataImagesFromPanoramas/" + i
            perspectiveOutputPath = outputPanoramaPath.replace("_panorama.jpg", "_photo.jpg")
            
            # check if not just black image or crop black borders if needed
            if not cropPanorama(inputPanoramaPath):
                continue

            preprocessPanorama(inputPanoramaPath, outputPanoramaPath)

            FOV, pitch, yaw, roll, height, width = randomGenerate()
            
            getPerspective(outputPanoramaPath, perspectiveOutputPath, FOV, pitch, yaw, roll, height, width)

            augmentation(perspectiveOutputPath)

            perspectiveToEquirectangular(perspectiveOutputPath, FOV, pitch, yaw, roll)

            saveToCsv(pitch, yaw, roll, "./dataImagesFromPanoramas/" + i.replace("_panorama.jpg", ".csv"))

            return
        except:
            continue

if __name__ == '__main__':
    data = getData()
    preprocess(data)
