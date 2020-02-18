# -*- coding: utf-8 -*-
"""Commonly used parameters."""
import numpy as np


# Coding constants
FILLVAL = 9e20
MUX_NAMES = ["track_idx", "row_idx"]

# Physical and metric constants
HOUR = np.timedelta64(1, "h")
M2KM = 1e-3
KM2M = 1e3
EARTH_RADIUS = 6_371_009.0  # in metres; NB in iris: 6367470 m
SCALE_VO = 1e-3
