"""
@file preprocessing.py
@date 2024-04-14
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Script for converting geoPose in cylindrical format to equirectangular format
"""

import os
import sys
import csv
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import lib.Equirec2Perspec as E2P
import lib.multi_Perspec2Equirec as m_P2E
import cv2
from PIL import Image
import math

panos = './in' # path for panoramas 
persps = './in' # path for images (can be different then path with panoramas)
out = './dataset' # output path

actualPath = os.path.dirname(os.path.abspath(__file__))
panos = os.path.join(actualPath, panos)
persps = os.path.join(actualPath, persps)
out = os.path.join(actualPath, out)

panoramaHeight = 2048
panoramaWidth = 4096
panoramaResolution = (panoramaWidth, panoramaHeight)

"""
@brief Function which convert input in cylindrical format of geoPose dataset
       to equirectangular format based on: https://paulbourke.net/panorama/pano2sphere/.

@param input input path
@param output output path
"""
def cylToEquir(input, output):
    input_image = Image.open(input)
    
    output_width = 8192
    output_height = 4096

    input_aspect_ratio = input_image.width / input_image.height
    new_height = int(output_width / input_aspect_ratio)

    resized_image = input_image.resize((output_width, new_height), Image.Resampling.LANCZOS)
    output_image = Image.new("RGB", (output_width, output_height), "black")
    
    y_position = (output_height - new_height) // 2 # put in the middle

    output_image.paste(resized_image, (0, y_position))
    resized_image = output_image.resize(panoramaResolution, Image.Resampling.LANCZOS)
    resized_image.save(output)

"""
@brief Function for converting image (second input) to equirectangular format in the middle

@param input input path
@param output output path
@param fov FOV value
"""
def getPerspEquir(input, output, fov):
    equ = m_P2E.Perspective([input],
                            [[fov, 180.0-180, 0.0]])

    img = equ.GetEquirec(panoramaHeight, panoramaWidth)
    cv2.imwrite(output, img)

"""
@brief Function for getting orientation from input text file in format defined in geoPose dataset.

@param input input path

@return pitch, yaw, roll angles
"""
def getOrientation(input):
    with open(input, 'r') as file:
        lines = file.readlines()
    
    orientation = lines[1]
    yaw, pitch, roll = orientation.split(" ")
    pitch, yaw, roll = float(pitch), float(yaw), float(roll)
    
    yaw = (yaw % (-2 * math.pi)) * -1
    pitch = pitch + math.pi / 2
    roll = (roll % (-2 * math.pi)) * -1
        
    return pitch * (180 / math.pi), yaw * (180 / math.pi), roll * (180 / math.pi)

"""
@brief Function for getting FOV from input text file in format defined in geoPose dataset.

@param input input path

@return FOV
"""
def getFOV(input):
    with open(input, 'r') as file:
        lines = file.readlines()

    return float(lines[5].strip()) * (180 / math.pi)

"""
@brief Main function for getting angles and FOV from text file

@param input input path
@param output output path
"""
def parseInfo(input, output):
    pitch, yaw, roll = getOrientation(input)
    fov = getFOV(input)
    with open(output, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([pitch, yaw, roll, fov])

# main loop
for folder in os.listdir(panos):
    pano_path = os.path.join(panos, folder)
    persp_path = os.path.join(persps, folder)
    if os.path.isdir(pano_path) and os.path.isdir(persp_path):
        pano = os.path.join(pano_path, 'cyl/pano.png')
        info = os.path.join(persp_path, 'info.txt')
        persp_jpg = os.path.join(persp_path, 'photo.jpg')
        persp_jpeg = os.path.join(persp_path, 'photo.jpeg')
        persp = None
        
        if os.path.exists(pano) and os.path.exists(info) and os.path.exists(persp_jpg):
            persp = persp_jpg
        elif os.path.exists(pano) and os.path.exists(info) and os.path.exists(persp_jpeg):
            persp = persp_jpeg
        else:
            continue
        
        cylToEquir(pano, os.path.join(out, folder + "_panorama.jpg"))
        parseInfo(info, os.path.join(out, folder + ".csv"))
        getPerspEquir(persp, os.path.join(out, folder + "_photo.jpg"), getFOV(info))

### End of preprocessing.py ###
