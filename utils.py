import cv2
import numpy as np


def find_coordinate_maxhexbin(sorted_vote, vertical):
    """Try to find the x and y coordinates of the hexbin with max count
    """
    for k, v in sorted_vote.items():
        # if mirror line is vertical, return the highest vote
        if vertical:
            return k[0], k[1]
        # otherwise, return the highest vote, whose y is not 0 or pi
        else:
            if k[1] == 0 or k[1] == np.pi:
                continue
            else:
                return k[0], k[1]


def read_bgr_image(image_path):
    """
    convert the image into the array/matrix with oroginal color
    """
    image = cv2.imread(image_path)  # convert the image into the array/matrix
    b, g, r = cv2.split(image)  # get b,g,r
    image = cv2.merge([r, g, b])  # switch it to rgb

    return image


def normalize_angle(point):
    point.angle = np.deg2rad(point.angle)
    point.angle = np.pi - point.angle
    if point.angle < 0.0:
        point.angle += 2 * np.pi


def midpoint(pi, pj):
    return pi.pt[0] / 2 + pj.pt[0] / 2, pi.pt[1] / 2 + pj.pt[1] / 2


def angle_with_x_axis(pi, pj):
    x, y = pi[0] - pj[0], pi[1] - pj[1]

    if x == 0:
        return np.pi / 2

    angle = np.arctan(y / x)
    if angle < 0:
        angle += np.pi
    return angle


def sort_hexbin_by_votes(image_hexbin):
    """Sort hexbins by decreasing count. (lower vote)
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
