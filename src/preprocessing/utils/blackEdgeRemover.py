"""
@file blackEdgeRemover.py
@date 2023-22-12
@author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
@brief Script for black edge detection and black edge removal from images.
"""

import cv2
import numpy as np

"""
@brief Function for checking if image is just black.

@param grayscale image in grayscale

@return True if image is black, otherwise False
"""
def checkIfBlackOnly(grayscale):
    if np.all(grayscale == 0):
        return True
    else:
        return False

"""
@brief Function for removing black edges from images. If images doesn't contain black edges nothing will be removed.

@param panoramaPath path to image

@return False if whole image is black, otherwise True.
"""
def cropPanorama(panoramaPath):
    panorama = cv2.imread(panoramaPath)

    # grayscale
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    # check if black only
    if (checkIfBlackOnly(gray)):
        return False

    # find contours
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = None
    largestArea = 0

    # find largest contour area
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largestArea:
            largestArea = area
            largestContour = contour

    if largestContour is not None:
        x, y, w, h = cv2.boundingRect(largestContour)
        cropped = panorama[y:y + h, x:x + w]
        cv2.imwrite(panoramaPath, cropped)

    return True

### End of blackEdgeRemover.py ###
