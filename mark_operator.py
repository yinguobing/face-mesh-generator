"""A module provids common operations for point marks.

All marks, or points are numpy arrays of format like:
    mark = [x, y, z]
    marks = [[x, y, z],
             [x, y, z],
             ...,
             [x, y, z]]

Vectors are also numpy arrays:
    vector = [x, y, z]
    vectors = [[x, y, z],
               [x, y, z],
               ...,
               [x, y, z]]

"""
import numpy as np


class MarkOperator(object):
    """Operator instances are used to transform the marks."""

    def __init__(self):
        pass

    def get_distance(self, mark1, mark2):
        """Calculate the distance between two marks."""
        return np.linalg.norm(mark2 - mark1)

    def get_angle(self, vector1, vector2, in_radian=False):
        """Return the angel between two vectors."""
        d = np.dot(vector1, vector2)
        cos_angle = d / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        if cos_angle > 1.0:
            radian = 0
        elif cos_angle < -1.0:
            radian = np.pi
        else:
            radian = np.arccos(cos_angle)

        c = np.cross(vector1, vector2)
        if (c.ndim == 0 and c < 0) or (c.ndim == 1 and c[2] < 0):
            radian = 2*np.pi - radian

        return radian if in_radian is True else np.rad2deg(radian)

    def get_center(self, marks):
        """Return the center point of the mark group."""
        x, y, z = (np.amax(marks, 0) + np.amin(marks, 0)) / 2

        return np.array([x, y, z])

    def rotate(self, marks, radian, center=(0, 0)):
        """Rotate the marks around center by angle"""
        _points = marks - np.array(center, np.float)
        cos_angle = np.cos(radian)
        sin_angle = np.sin(radian)
        rotaion_matrix = np.array([[cos_angle, sin_angle],
                                   [-sin_angle, cos_angle]])

        return np.dot(_points, rotaion_matrix) + center
