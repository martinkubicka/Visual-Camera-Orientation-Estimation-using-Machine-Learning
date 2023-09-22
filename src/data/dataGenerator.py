"""
@file dataGenerator.py
@author Martin Kubicka <xkubic45@stud.fit.vutbr.cz>
@date 26.08.2023
@brief Main file for generating dataset.
"""

import os
from dotenv import load_dotenv
from download import get_panorama
from randomPanoramaIDGenerator import getRandomPanoID
import traceback
from mapillaryImageGenerator import downloadMapillaryImage

"""
@brief Function which returns API_GOOGLE_KEY stored in .env.

@return API_GOOGLE_KEY from .env file.
"""
def getGoogleStreetviewAPIKey():
    load_dotenv()
    return os.environ.get('API_GOOGLE_KEY')

"""
@brief Function which returns API_MAPILLARY_CLIENT_TOKEN stored in .env.

@return API_MAPILLARY_CLIENT_TOKEN from .env file.
"""
def getMapillaryAPIKey():
    load_dotenv()
    return os.environ.get('API_MAPILLARY_CLIENT_TOKEN')


"""
@brief Function which gets user input.

@return Number of panoramas which needs to be generated.
"""
def getNumberOfPanoramas():
    while True:
        try:
            return int(input("Enter number of panoramas generated: "))
        except:
            pass

"""
@brief Function for saving received data for future use in text file.

@param panoID google streetview api panorama id
@param imageID mappilary image id
@param latitude latitude
@param longitude longitude
@param pitch pitch
@param yaw yaw
@param roll roll
"""
def saveData(panoID, imageID, latitude, longitude, pitch, yaw, roll):
    data = f"{panoID}\n{imageID}\n{latitude}\n{longitude}\n{pitch}\n{yaw}\n{roll}\n"

    with open(f"panoramas/{panoID}.txt", "w") as file:
        file.write(data)

if __name__ == '__main__':
    API_GOOGLE_KEY = getGoogleStreetviewAPIKey()
    API_MAPILLARY_KEY = getMapillaryAPIKey()

    # auxiliary counter variable so we can know how many panoramas were saved
    count = 0
    numberOfPanoramas = getNumberOfPanoramas()
    while count < numberOfPanoramas:
        try:
            panoId = getRandomPanoID(API_GOOGLE_KEY, API_MAPILLARY_KEY)

            panorama = get_panorama(pano_id=panoId[0])
            mapillary = downloadMapillaryImage(API_MAPILLARY_KEY, panoId[0], panoId[1])

            # if we can't save mappilary image then continue
            if not mapillary:
                continue

            panorama.save("panoramas/" + panoId[0] + "_panorama.jpg", "jpeg")
            saveData(panoId[0], panoId[1], panoId[2], panoId[3], panoId[4], panoId[5], panoId[6])

            print(str(count + 1) + ". panorama saved to panoramas/" + panoId[0] + "_panorama.jpg.")
            print("-------------------------")

            count += 1
        except:
            print(traceback.format_exc())
            print("An error occured. Trying again..")
            print("-------------------------")

### End of dataGenerator.py ###
