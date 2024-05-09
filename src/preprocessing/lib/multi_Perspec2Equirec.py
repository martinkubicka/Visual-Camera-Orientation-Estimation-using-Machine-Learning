"""
Credit goes to the original author and contributors of the 'Perspective-and-Equirectangular' repository: https://github.com/timy90022/Perspective-and-Equirectangular
"""

import numpy as np
import lib.Perspec2Equirec as P2E


class Perspective:
    def __init__(self, img_array, F_T_P_array):
        assert len(img_array) == len(F_T_P_array)

        self.img_array = img_array
        self.F_T_P_array = F_T_P_array

    def GetEquirec(self, height, width):
        merge_image = np.zeros((height, width, 3))
        merge_mask = np.zeros((height, width, 3))

        for img_dir, [FOV, YAW, PITCH] in zip(self.img_array, self.F_T_P_array):            
            per = P2E.Perspective(img_dir, FOV, YAW, PITCH)  # Load perspective image
            img, mask = per.GetEquirec(height, width)  # Specify parameters(height, width)
            merge_image += img
            merge_mask += mask

        merge_mask = np.where(merge_mask == 0, 1, merge_mask)
        merge_image = (np.divide(merge_image, merge_mask))

        return merge_image
