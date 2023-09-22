"""
@file mappilaryImageGenerator.py
@author Martin Kubicka <xkubic45@stud.fit.vutbr.cz>
@date 22.09.2023
@brief Definitions of functions for getting mapillary image using mapillary API.
"""

import requests
import json

"""
@brief Function for checking if image exists on given coordinates.

@param API_KEY mapillary api key
@param long longitude
@param lat latitude

@return bool if image exists on given coordinates, image id or None if not found
"""
def checkIfImageExists(API_KEY, long, lat):
    response = requests.get("https://graph.mapillary.com/images?access_token=" + API_KEY + "&fields=id&bbox=" + str(long) + "," + str(lat) + "," + str(float(long) + 0.5) + "," + str(float(lat) + 0.5))

    if response.status_code == 400:
        print("An error occured. Trying again..")
        return False, None
    else:
        responseJson = json.loads(response.content.decode('utf-8'))

        if "error" in responseJson or len(responseJson["data"]) == 0:
            return False, None
        else:
            return True, responseJson["data"][0]["id"]

"""
@brief Function for getting image data based on mapillary image ID

@param API_KEY mapillary API key
@param imageID mapillary image ID

@return bool - True if successful otherwise False, latitude, longitude, pitch, yaw, roll of a image
"""
def getImageData(API_KEY, imageID):
    response = requests.get("https://graph.mapillary.com/" + imageID + "?access_token=" + API_KEY + "&fields=computed_geometry,computed_rotation")
    responseJson = json.loads(response.content.decode('utf-8'))
    if response.status_code == 400 or "error" in responseJson:
        return False, None, None, None, None, None
    else:
        return True, responseJson["computed_geometry"]["coordinates"][0], responseJson["computed_geometry"]["coordinates"][1], responseJson["computed_rotation"][0], responseJson["computed_rotation"][1], responseJson["computed_rotation"][2], responseJson["id"]

"""
Credit to the original author of the code I incorporated into my project.
Source: https://gist.github.com/cbeddow/79d68aa6ed0f028d8dbfdad2a4142cf5
The following portion of the code is used under the terms of the original license.
Author: cbeddow

@brief Function for downloading image based on mapillary image ID

@param API_KEY mapillary API key
@param fileName name of downloaded image file
@param imageId mapillary image ID

@return bool False if download not successful otherwise True
"""
def downloadMapillaryImage(API_KEY, fileName, imageId):
    url = "https://graph.mapillary.com/" + imageId + "?fields=thumb_2048_url&access_token=" + API_KEY
    try:
        response = requests.get(url)

        if response.status_code == 400:
            return False

        data = response.json()
        image = data['thumb_2048_url']

        with open("panoramas/" + fileName + "_photo.jpg", 'wb') as file:
            image_data = requests.get(image, stream=True).content
            file.write(image_data)

        return True
    except:
        return False

"""
@brief Main function for getting mapillary image

@param API_KEY mapillary API key
@param long longitude
@param lat latitude

@return bool - True if successful otherwise False, latitude, longitude, pitch, yaw, roll of a image or None if not successful
"""
def getMapillaryImage(API_KEY, long, lat):
    imageID = checkIfImageExists(API_KEY, long, lat)

    if (imageID[0]):
        print("Mappilary image successfully found.")
        return getImageData(API_KEY, imageID[1])
    else:
        print("Mappilary image not found.\nGetting new one..")
        return False, None, None, None, None, None, None

### End of mapillaryImageGenerator.py ###
