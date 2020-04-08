import cv2
import numpy as np


def find_coordinate_maxhexbin(sorted_vote):
    """
    This function find the coordinates of the most voted item in the hexbin.
    :param sorted_vote: the sorted hexbin.
    :return:
    """
    for k, v in sorted_vote.items():
        if k[1] == 0 or k[1] == np.pi:
            continue
        else:
            return k[0], k[1]


def read_bgr_image(image_path):
    """
    Reads an image in bgr and converts to rgb.
    :param image_path: the image.
    :return: the image in rgb
    """
    image = cv2.imread(image_path)
    im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return im_rgb


def normalize_angle(point):
    """
    This function normalizes the angle to radians in the interval [0,pi]
    :param point:
    :return:
    """
    point.angle = np.deg2rad(point.angle)
    point.angle = np.pi - point.angle
    if point.angle < 0.0:
        point.angle += 2 * np.pi


def midpoint(pi, pj):
    return pi.pt[0] / 2 + pj.pt[0] / 2, pi.pt[1] / 2 + pj.pt[1] / 2


def angle_with_x_axis(pi, pj):
    """
    Calculates the angle of the positive x-axis line from pi to pj.
    :param pi:
    :param pj:
    :return:
    """
    x, y = pi[0] - pj[0], pi[1] - pj[1]

    if x == 0:
        return np.pi / 2

    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle


def sort_hexbin_by_votes(image_hexbin):
    """
    This function sorted the hexbin by votes.
    :param image_hexbin: the hexbin to sort.
    :return: sorted hexbin dictionary.
    """
    counts = image_hexbin.get_array()
    ncnts = np.count_nonzero(np.power(10, counts))  # get non-zero hexbins
    verts = image_hexbin.get_offsets()  # coordinates of each hexbin
    output = {}

    for offc in range(verts.shape[0]):
        binx, biny = verts[offc][0], verts[offc][1]
        if counts[offc]:
            output[(binx, biny)] = counts[offc]
    return {k: v for k, v in sorted(output.items(), key=lambda item: item[1], reverse=True)}
