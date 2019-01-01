# -*- coding: utf-8 -*-
"""Classes and functions for the analysis of PMCTRACK output."""
import numpy as np

HOUR = np.timedelta64(1, "h")
M2KM = 1e-3

COLUMNS = ["lon", "lat", "vo", "time", "area", "vortex_type"]  # , 'cat']
ARCH_KEY = "trackrun"
