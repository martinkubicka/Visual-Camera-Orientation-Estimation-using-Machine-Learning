"""
@file preprocessing.py
@date 2023-22-12
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Preprocessing of images scraped with datasetGenerator.py, inspired by PAC preprocessing and Segmentation/PitchYaw preprocessing, but cutouts are created on right pitch not in the middle.
"""

from PIL import Image
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from PAC.preprocessing import preprocessPanorama, randomGenerate, getPerspective, augmentation, perspectiveToEquirectangular, saveToCsv, getData
from Probability.PitchYaw.preprocessing import main
from utils.blackEdgeRemover import cropPanorama
import traceback

panoramaHeight = 2048
panoramaWidth = 4096
panoramaResolution = (panoramaWidth, panoramaHeight)

NUMBER_OF_GENERATED_IMAGES_FROM_PANORAMA = 1

input_path = "./in" # input path
input_output_pac = './in_out_pac' # input/output path for creating pac dataset with right pitch positioned cutouts
output_path = "./dataset" # ouput path

actualDirectory = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(actualDirectory, input_path)
input_output_pac = os.path.join(actualDirectory, input_output_pac)
output_path = os.path.join(actualDirectory, output_path)

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
                outputPanoramaPath = os.path.join(actualDirectory, input_output_pac + '/' + i.replace("_panorama", str(index) + "_panorama"))
                perspectiveOutputPath = outputPanoramaPath.replace("_panorama", "_photo")
                
                try:
                    preprocessPanorama(inputPanoramaPath, outputPanoramaPath)
                except:
                    print(traceback.format_exc())

                FOV, pitch, yaw, roll, height, width = randomGenerate()

                getPerspective(outputPanoramaPath, perspectiveOutputPath, FOV, yaw, pitch, roll, height, width)
                
                augmentation(perspectiveOutputPath)
                
                perspectiveToEquirectangular(perspectiveOutputPath, FOV, 180.0, pitch - 90.0)

                saveToCsv(pitch, yaw, roll, FOV, os.path.join(actualDirectory, input_output_pac + '/' + i.replace("_panorama.jpg", str(index) + ".csv")))
        except KeyboardInterrupt:
            exit()
        except:
            print(traceback.format_exc())
            continue

if __name__ == '__main__':
    data = getData(input_path)
    preprocess(data)
    main(input_output_pac, output_path)
    
### End of preprocessing.py ###
