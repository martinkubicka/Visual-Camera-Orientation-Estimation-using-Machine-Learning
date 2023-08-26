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

"""
@brief Function which returns API_KEY stored in .env.

@return API_KEY from .env file.
"""
def getAPIKey():
    load_dotenv()
    return os.environ.get('API_KEY')

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

if __name__ == '__main__':
    API_KEY = getAPIKey()

    # auxiliary counter variable so we can know how many panoramas were saved
    count = 0
    numberOfPanoramas = getNumberOfPanoramas()
    while count < numberOfPanoramas:
        try:
            panoId = getRandomPanoID(API_KEY)
            panorama = get_panorama(pano_id=panoId)
            panorama.save("panoramas/" + panoId + ".jpg", "jpeg")

            print(str(count + 1) + ". panorama saved to panoramas/" + panoId + ".jpg.")
            print("-------------------------")

            count += 1
        except:
            print("An error occured. Trying again..")
            print("-------------------------")

### End of dataGenerator.py ###
