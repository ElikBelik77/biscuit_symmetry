import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils


class BilateralDetector:
    """
    Class implementing the algorithm in Loy's paper.
    """

    def __init__(self):
        self.source_path = None
        self.image = None
        self.reflected_image = None
        self.keypoints = None
        self.reflected_keypoints = None
        self.matchpoints = None
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.bf = cv2.BFMatcher()
        self.symmetry_line = (None, None)
        self.points_r = None
        self.points_theta = None

    def find(self, source_path):
        """
        This function finds the symmetry line.
        :param source_path: the source path of the iamge to analyze.
        :return: None.
        """
        self.source_path = source_path

        # Read image and flip bgr to rgb.
        self.image = utils.read_bgr_image(source_path)
        self.reflected_image = np.fliplr(self.image)

        # extract keypoint and descriptors.
        self.keypoints, descriptors = self.sift.detectAndCompute(self.image, None)
        self.reflected_keypoints, reflected_descriptors = self.sift.detectAndCompute(self.reflected_image, None)

        # find the matching points using knn.
        self.matchpoints = self.match_descriptors(descriptors, reflected_descriptors)
        # calculate the symmetry score for each match, and the potential symmetry line.
        matchpoints_weight = [self.calculate_symmetry_match(match) for match in self.matchpoints]
        potential_symmetry_axis = [self.get_potential_symmetry_axis(match) for match in self.matchpoints]
        self.points_r = [rij for rij, xc, yc, theta in potential_symmetry_axis]
        self.points_theta = [theta for rij, xc, yc, theta in potential_symmetry_axis]

        # put all the points in a hexbin with voting power of 1.
        image_hexbin = plt.hexbin(self.points_r, self.points_theta, bins=50, cmap=plt.cm.Spectral_r)
        sorted_vote = utils.sort_hexbin_by_votes(image_hexbin)
        # the symmetry line is induced by the most voted item in the hexbin.
        r, theta = utils.find_coordinate_maxhexbin(sorted_vote)
        self.symmetry_line = (r, theta)
        plt.close()
        return r, theta

    def get_potential_symmetry_axis(self, match):
        """
        This function calculates the symmetry line of a possible match.
        :param match: the match.
        :return: a tuple (rij, xc, yc, theta) of the magnitude of the line, starting point of the line, and the angle.
        """
        pi = self.keypoints[match.queryIdx]
        pj = self.reflected_keypoints[match.trainIdx]
        utils.normalize_angle(pi)
        utils.normalize_angle(pj)
        pj.pt = (self.image.shape[1] - pj.pt[0], pj.pt[1])
        theta = utils.angle_with_x_axis(pi.pt, pj.pt)
        xc, yc = utils.midpoint(pi, pj)
        rij = xc * np.cos(theta) + yc * np.sin(theta)
        return rij, xc, yc, theta

    def match_descriptors(self, descriptors, reflected_descriptors):
        """
        This function matches pairs of descriptors.
        :param descriptors: the descriptors to match
        :param reflected_descriptors: the descriptors to 'train' knn with.
        :return: list of matches between descriptors and reflected_descriptors based on the knn algorithm.
        """
        matchpoints = [item[0] for item in self.bf.knnMatch(descriptors, reflected_descriptors, k=2)]
        matchpoints = sorted(matchpoints, key=lambda x: x.distance)
        return matchpoints

    def calculate_symmetry_match(self, match):
        """
        This function calculates the score of the symmetry match.
        :param match: the match.
        :return: the symmetry score/weight.
        """
        pi = self.keypoints[match.queryIdx]
        pj = self.reflected_keypoints[match.trainIdx]
        utils.normalize_angle(pi)
        utils.normalize_angle(pj)
        pj.pt = (self.image.shape[1] - pj.pt[0], pj.pt[1])
        theta = utils.angle_with_x_axis(pi.pt, pj.pt)
        angular_symmetry = 1 - np.cos(pj.angle + pi.angle - 2 * theta)
        scale_symmetry = np.exp((-abs(pi.size - pj.size) / ((pi.size + pj.size)))) ** 2
        d = (pj.pt[0] - pi.pt[0]) ** 2 + (pj.pt[1] - pi.pt[1]) ** 2
        distance_weight = np.exp(-d ** 2 / 2)
        return angular_symmetry * scale_symmetry * distance_weight
