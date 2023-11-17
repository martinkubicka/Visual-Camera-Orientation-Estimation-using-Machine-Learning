import cv2
import numpy as np

def checkIfBlackOnly(grayscale):
    if np.all(grayscale == 0):
        return True
    else:
        return False

def cropPanorama(panoramaPath):
    panorama = cv2.imread(panoramaPath)

    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    if (checkIfBlackOnly(gray)):
        return False

    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largestContour = None
    largestArea = 0

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
