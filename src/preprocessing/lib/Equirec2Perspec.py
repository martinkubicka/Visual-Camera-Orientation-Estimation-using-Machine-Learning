"""
Credit goes to the original author and contributors of the 'Perspective-and-Equirectangular' repository: https://github.com/timy90022/Perspective-and-Equirectangular
"""

import cv2
import numpy as np

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
    
    def GetPerspective(self, FOV, THETA, PHI, height, width):
        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))

        # Generate x, y, z coordinates for a plane
        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])
        z_map = -np.tile(np.linspace(-h_len, h_len, height), [width, 1]).T

        # Normalize vectors (x, y, z) to unit vectors
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(D[:, :, np.newaxis], 3, axis=2)

        # Rotate about Y axis for THETA (left/right) and then about X axis for PHI (up/down)
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(0))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-0))
        
        # Integrate rotation by 45 degrees about the Z-axis to avoid black edges
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        [R3, _] = cv2.Rodrigues(x_axis * np.radians(45))  # Rotation by 45 degrees

        # Apply rotations
        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R3, xyz).T  # Apply the additional 45-degree rotation
        xyz = xyz.reshape([height, width, 3])

        # Convert XYZ back to lat/lon
        lat = np.arcsin(xyz[:, :, 2])
        lon = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])

        lon = lon / np.pi * 180
        lat = -lat / np.pi * 180

        # Map lat/lon to pixel coordinates
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        # Remap pixels to get the perspective image
        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp

            
