"""
@file randomPanoramaIDGenerator.py
@author Martin Kubicka <xkubic45@stud.fit.vutbr.cz>
@date 26.08.2023
@brief Definitions of functions for getting random panorama IDs from google streetview static api.
"""

import urllib.parse
import random
import requests
from mapillaryImageGenerator import getMapillaryImage

PANO_METADATA_URL = 'https://maps.googleapis.com/maps/api/streetview/metadata?'

"""
@brief Function for generating random latitude.

@return Random latitude number.
"""
def getRandomLatitude():
    num = random.uniform(-90, 90)
    return f'{num:.6f}'

"""
@brief Function for generating random longitude.

@return Random longitude number.
"""
def getRandomLongitude():
    num = random.uniform(-180, 180)
    return f'{num:.6f}'

"""
@brief Function for checking if there is panorama on generated URL.

@param url unvalidated url

@return tuple(True if url is valid otherwise False, panorama ID if valid otherwise None)
"""
def checkIfPanoExists(url):
    response = requests.get(url).json()

    if response.get("status") == "OK":
        return True, response.get("pano_id")
    else:
        return False, None

"""
@brief Function for generating not validated panorama URL.

@param latitude Latitude
@param longitude Longitude
@param API_KEY Api key for accessing google streetview static APIs.

@return Not validated panorama URL.
"""
def getPanoMetadataUrl(latitude, longitude, API_KEY):
    params = {
        'location': f'{latitude},{longitude}',
        'key': API_KEY,
        'radius': 3,
    }

    requestUrl = PANO_METADATA_URL + urllib.parse.urlencode(params)

    return requestUrl

"""
@brief Main function for getting random panorama ID and mapillary image. Panorama is based on mapillary image latitude and longitude.

@param API_KEY Api key for accessing google streetview static APIs.

@return google streetview panorama ID, mapillary image ID, latitude, longitude, pitch, yaw, roll of a image
"""
def getRandomPanoID(API_GOOGLE_KEY, API_MAPILLARY_KEY):
    latitude, longitude, url, pano, mappilaryImageExists = None, None, None, (False, None), False

    while latitude == None or not pano[0] or not mappilaryImageExists:
        latitude = getRandomLatitude()
        longitude = getRandomLongitude()

        # at first we are getting mapillary image
        mappilaryImage = getMapillaryImage(API_MAPILLARY_KEY, longitude, latitude)
        if mappilaryImage[0]:
            mappilaryImageExists = True
        else:
            continue

        # then we are getting panorama image
        url = getPanoMetadataUrl(mappilaryImage[2], mappilaryImage[1], API_GOOGLE_KEY)
        pano = checkIfPanoExists(url)

    return pano[1], mappilaryImage[6], mappilaryImage[2], mappilaryImage[1], mappilaryImage[3], mappilaryImage[4], mappilaryImage[5]

### End of randomPanoramaIDGenerator.py ###
