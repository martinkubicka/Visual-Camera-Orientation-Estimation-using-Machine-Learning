"""
@file preprocessing.py
@date 2023-22-12
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Converting PAC dataset to segmentation (pitch and yaw estimation) dataset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import lib.multi_Perspec2Equirec as m_P2E
from PAC.preprocessing import getPerspective, perspectiveToEquirectangular
import shutil
from PIL import Image, ImageFilter
import numpy as np
import csv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

panoramaHeight = 2048
panoramaWidth = 4096
panoramaResolution = (panoramaWidth, panoramaHeight)

height = 960
width = 1920

input_path = "./in" # input path
output_path = "./dataset" # output path

actualDirectory = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(actualDirectory, input_path)
output_path = os.path.join(actualDirectory, output_path)

"""
@brief Function for creating segmentation ground truth

@param input_path converting equirectangular truth to black and white
@param out_path output path
"""
def getSegTruth(input_path, out_path):
    original_image = Image.open(input_path)

    gray_image = original_image.convert("L")

    gray_array = np.array(gray_image)
    threshold = 1
    black_white_array = np.where(gray_array <= threshold, 0, 255).astype(np.uint8)
    black_white_image = Image.fromarray(black_white_array, 'L')
    
    radius = 120
    bw_image_fade = black_white_image.filter(ImageFilter.GaussianBlur(radius=radius))
    bw_image_fade.save(out_path)

"""
@brief Function for getting path etc. from input folder

@param in_path input path

@return list of tuples (panorama_path, picture_path, ground_truth_csv)
"""
def get_data(in_path):
    csv_files = [f for f in os.listdir(in_path) if f.endswith(".csv")]
    files = []
    
    for csv_file in csv_files:
        image_id = os.path.splitext(csv_file)[0]
        panorama_image_path = os.path.join(in_path, f"{image_id}_panorama.jpg")
        picture_image_path = os.path.join(in_path, f"{image_id}_photo.jpg")
        
        if not os.path.exists(panorama_image_path) or not os.path.exists(picture_image_path):
            continue

        csv_data = []
        with open(os.path.join(in_path, csv_file), newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                data = row[:4]
                csv_data = [float(data[0]), float(data[1]), float(data[2])]
                csv_data = [round(num, 4) for num in csv_data]   
                csv_data.append(data[3])
        
        files.append((panorama_image_path, picture_image_path, csv_data))
    
    return files

"""
@brief Function for creating cutout positioned on truth angles

@param panorama_path panorama path
@param out_path output path
@param FOV FOV
@param orientation orientation list [pitch, yaw, roll]
@param perspective_height perspective height
@param perspective_width perspective width
"""
def create_cutout(panorama_path, out_path, FOV, orientation, perspective_height, perspective_width):    
    getPerspective(panorama_path, out_path, FOV, orientation[1], orientation[0], orientation[2], perspective_height, perspective_width)
    
    shutil.copy(panorama_path, output_path + '/' + panorama_path.split('/')[-1])
    shutil.copy(panorama_path.replace('_panorama', '_photo'), output_path + '/' + panorama_path.replace('_panorama', '_photo').split('/')[-1])
    perspectiveToEquirectangular(out_path, float(FOV), float(orientation[1]), float(orientation[0]) - 90)

def main(in_path, out_path):
    data = get_data(in_path)
    
    for i in data:
        panorama_path = i[0]
        perspective_path = i[1]
        ground_truth = i[2]
        
        out = out_path + '/' + perspective_path.split('/')[-1].replace('_photo', '_truth')
        
        create_cutout(panorama_path, out, ground_truth[-1], ground_truth[:3],  height, width)
        getSegTruth(out, out)
                
if __name__ == '__main__':
    main(input_path, output_path)

### End of preprocessing.py ###
