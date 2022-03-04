#!/usr/bin/env python

"""analysis.py: Analysis module to elevation angle data."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


import sys
sys.path.append("py/")
import datetime as dt
import numpy as np

import pydarn
from get_fit_data import FetchData
import calc_elv

def process_elevation_angle(rad, dates, tdiff=None):
    """
    Main process to run Elevaton angle
    """
    fdata = FetchData( rad, dates )
    b, _ = fdata.fetch_data()
    o = fdata.convert_to_pandas(b)
    o["elv0"] = calc_elv.caclulate_elevation_angle(np.array(o.elv), np.array(o.bmnum),
                                                   np.array(o.tfreq), pydarn.read_hdw_file(rad),
                                                   tdiff)    
    return

if __name__ == "__main__":
    rad = "cvw"
    dates = [dt.datetime(2016,1,1), dt.datetime(2016,1,1,6)]
    tdiff = -0.346
    process_elevation_angle(rad, dates, tdiff)
    pass