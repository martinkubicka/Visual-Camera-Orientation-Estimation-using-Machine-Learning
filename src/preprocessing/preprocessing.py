from PIL import Image
import os
import csv
import math

def getData():
    dataFolder = "../data/panoramas"
    result = []

    for filename in os.listdir(dataFolder):
        if filename.endswith("_photo.jpg"):
            photoPath = os.path.join(dataFolder, filename)
            panoramaPath = photoPath.split("_photo.jpg")[0] + "_panorama.jpg"
            txtPath = photoPath.split("_photo.jpg")[0] + ".txt"
            
            if not os.path.isfile(panoramaPath) or not os.path.isfile(txtPath):
                continue
            
            with open(txtPath, 'r') as file:
                lines = file.readlines()
            
            result.append([panoramaPath, photoPath, [line.rstrip() for line in lines[-3:]]])
    return result

def preprocess(data):
    imageResolution = (1920, 1080)
    panoramaResolution = (4096, 2048)
    
    for i in data:
        # panorama
        panorama = Image.open(i[0])
        resizedPanorama = panorama.resize(panoramaResolution)
        resizedPanorama.save("./data/" + i[0].split("/")[-1])
        
        # image
        image = Image.open(i[1])
        resizedImage = image.resize(imageResolution)
        resizedImage.save("./data/" + i[1].split("/")[-1])
        
        # pitch, yaw, roll
        with open("./data/" + i[1].split("/")[-1].replace("_photo.jpg", "") + ".csv", mode='w', newline='') as file:
            writer = csv.writer(file)

            # yaw not negative number
            if (float(i[2][1]) < 0):
                i[2][1] = str(2 * math.pi + float(i[2][1]))

            # roll not negative number
            if (float(i[2][2]) < 0):
                i[2][2] = str(2 * math.pi + float(i[2][2]))

            writer.writerow(i[2])

if __name__ == '__main__':
    data = getData()
    preprocess(data)
