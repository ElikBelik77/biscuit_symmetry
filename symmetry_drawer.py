import io
import PIL
import cv2
import numpy as np
import matplotlib.pyplot as plt


class SymmetryDrawer:
    """
    Class for drawing symmetry lines and intermidiate stages of the algorithm.
    """
    def __init__(self):
        pass

    def draw_matchpoints(self, out, bilateral_detector, maximal=10):
        """
        This function draws the matching feature points and saves to a file.
        :param out: the out path.
        :param bilateral_detector: the bilateral detector to read from.
        :param maximal: the maximal amount of matches to draw.
        :return: None.
        """
        image = cv2.imread(bilateral_detector.source_path)
        reflected_image = np.fliplr(image)
        result = cv2.drawMatches(image, bilateral_detector.keypoints,
                                 reflected_image, bilateral_detector.reflected_keypoints,
                                 bilateral_detector.matchpoints[:maximal], None,
                                 flags=2)
        cv2.imwrite(out, result, None)

    def draw_keypoints(self, out, bilateral_detector):
        """
        This function draws the keypoints/features on the image and saves it to the file.
        :param out: the path to save the image to.
        :param bilateral_detector: the bilateral detector.
        :return: None
        """
        image = cv2.imread(bilateral_detector.source_path)
        color = (255, 0, 0)
        for kp in bilateral_detector.keypoints:
            cv2.circle(image, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size), color, 1)
        cv2.imwrite(out, image, None)

    def draw_hexbin(self, out, bilateral_detector):
        """
        This function draws the hexbin image and saves it.
        :param out: the path to save the hexbin image to.
        :param bilateral_detector: the bilateral detector to use.
        :return: None
        """
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
        This function draws the symmetry line on the image and saves it to a file
        :param out: the file to save to.
        :param bilateral_detector: the symmetry detector.
        :return: None.
        """
        r, theta = bilateral_detector.symmetry_line
        image = cv2.imread(bilateral_detector.source_path)
        for y in range(len(image)):
            try:
                x = int((r - y * np.sin(theta)) / np.cos(theta))
                image[y][x] = 255
                image[y][x + 1] = 255
                image[y][x - 1] = 255
            except IndexError:
                continue
        # draw plot
        cv2.imwrite(out, image)
