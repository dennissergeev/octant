# -*- coding: utf-8 -*-
"""Classes and functions for the analysis of PMCTRACK output."""
import numpy as np

HOUR = np.timedelta64(1, "h")
M2KM = 1e-3
KM2M = 1e3

COLUMNS = ["lon", "lat", "vo", "time", "area", "vortex_type"]
ARCH_KEY = "trackrun"
ARCH_KEY_CAT = ARCH_KEY + "_categories"

EARTH_RADIUS = 6_371_009.0  # in metres
# NB in iris: 6367470 m
