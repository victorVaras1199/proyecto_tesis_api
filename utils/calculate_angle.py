from typing import List

import numpy as np


def calculate_angle(a, b, c) -> float:
	"""Calculates the angle between three points.

	Args:
		a (List[float]): Coordinates [x, y] of the first point.
		b (List[float]): Coordinates [x, y] of the second point (vertex).
		c (List[float]): Coordinates [x, y] of the third point.

	Returns:
		float: The angle in degrees between the three points.
	"""
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)

	radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

	angle = np.abs(radians * 180.0 / np.pi)

	if angle > 180.0:
		angle = 360.0 - angle

	return angle
