import io

import PIL
import cv2
import numpy as np

import utils
import matplotlib.pyplot as plt


class SymmetryDrawer:
    def __init__(self):
        pass

    def draw_matchpoints(self, out, bilateral_detector, maximal=10):
        image = cv2.imread(bilateral_detector.source_path)
        reflected_image = np.fliplr(image)
        result = cv2.drawMatches(image, bilateral_detector.keypoints,
                                 reflected_image, bilateral_detector.reflected_keypoints,
                                 bilateral_detector.matchpoints[:maximal], None,
                                 flags=2)
        cv2.imwrite(out, result, None)

    def draw_keypoints(self, out, bilateral_detector):
        image = cv2.imread(bilateral_detector.source_path)
        color = (255, 0, 0)
        for kp in bilateral_detector.keypoints:
            cv2.circle(image, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size), color, 1)
        cv2.imwrite(out, image, None)

    def draw_hexbin(self, out, bilateral_detector):
        plt.hexbin(bilateral_detector.points_r, bilateral_detector.points_theta, bins=200,
                   cmap=plt.cm.Spectral_r)
        plt.colorbar()  # add color bar
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        buf.seek(0)
        im = PIL.Image.open(buf)
        im.save(out)

    def draw_symmetry(self, out, bilateral_detector):
        """
        Draw mirror line based on r theta polar co-ordinate
        """
        r, theta = bilateral_detector.symmetry_line
        image = cv2.imread(bilateral_detector.source_path)
        for y in range(len(image)):
            try:
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                image[y][x] = 255
                image[y][x + 1] = 255
            except IndexError:
                continue
        # draw plot
        cv2.imwrite(out, image)
