"""
Credit goes to the original author and contributors of the 'Perspective-and-Equirectangular' repository: https://github.com/timy90022/Perspective-and-Equirectangular
"""

import cv2
import numpy as np

class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, YAW, PITCH, ROLL, height, width):
        # Nastavenie rovníkových parametrov
        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        # Vypočítanie šírky a výšky zorného poľa (FOV)
        wFOV = FOV
        hFOV = float(height) / width * wFOV

        # Výpočet dĺžky strán zorného poľa
        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))

        # Vytvorenie mapovacích polí pre x, y, z súradnice
        x_map = np.ones([height, width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len, width), [height, 1])
        z_map = -np.tile(np.linspace(-h_len, h_len, height), [width, 1]).T

        # Vytvorenie pola vzdialenosti od stredového bodu
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.stack((x_map, y_map, z_map), axis=2) / np.repeat(D[:, :, np.newaxis], 3, axis=2)

        # Definícia osí y, x a z
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)

        # Zmeny pre yaw, pitch a roll
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(YAW))
        [R2, _] = cv2.Rodrigues(x_axis * np.radians(ROLL))
        [R3, _] = cv2.Rodrigues(y_axis * np.radians(PITCH))

        # Transformácia súradníc podľa rotácií
        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R3, xyz).T

        # Výpočet zemepisnej šírky a dĺžky
        lat = np.arcsin(xyz[:, 2])
        lon = np.arctan2(xyz[:, 1], xyz[:, 0])

        # Adjusting the longitude to have 0 on the left and 360 on the right
        lon = np.degrees(lon)
        lon = (lon + 360) % 360

        # Konverzia na stupne a zobrazenie na ploche
        lon = lon.reshape([height, width])
        lat = -lat.reshape([height, width]) / np.pi * 180

        lon = lon / 360 * equ_w  # Adjusted for 0 on the left and 360 on the right
        lat = lat / 90 * equ_cy + equ_cy

        result = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        return result
