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
import os
import datetime as dt
import numpy as np

import pydarn
from get_fit_data import FetchData
import calc_elv
from plotlib import RangeTimeIntervalPlot as RTI
import matplotlib.dates as mdates

def process_elevation_angle(rad, dates, tdiffs):
    """
    Main process to run Elevaton angle
    """
    fdata = FetchData( rad, dates )
    b, _ = fdata.fetch_data()
    o = fdata.convert_to_pandas(b)
    o.tfreq = np.round(o.tfreq/1e3, 1)
    bm = 7
    o["mdates"] = o.time.apply(lambda x: mdates.date2num(x))
    #o = o[(o.tfreq>=10.300) & (o.tfreq<=10.8) & (o.bmnum==bm)]
    o = o[(o.bmnum==bm)]
    #print(np.unique(o.tfreq))
    print(o.phi0.tolist())
    rti = RTI(100, np.array(o.mdates), 
              "Rad: %s, Beam: %d, %s"%(rad.upper(), bm, dates[0].strftime("%d %b %Y")))
    hdw = pydarn.read_hdw_file(rad)
    for i, tdiff in enumerate(tdiffs):
        o["elv"+str(i)] = calc_elv.caclulate_elevation_angle(np.array(o.phi0), np.array(o.bmnum),
                                                             np.array(o.tfreq), hdw, tdiff)
        tdiff = hdw.tdiff if tdiff is None else tdiff
        ax, im = rti.addParamPlot(o, r"$T_{diff}$=%.3f $\mu s$"%tdiff, zparam="v", 
                                  p_max=100, p_min=-100, p_step=25)#zparam="elv"+str(i))
        print(o["elv"+str(i)].tolist())
        break
    rti.save("tmp/elevation.png")
    rti.close()
    return

if __name__ == "__main__":
    if not os.path.exists("tmp/"): os.mkdir("tmp/")
    rad = "cvw"
    dates = [dt.datetime(2014,4,23), dt.datetime(2014,4,23,1)]
    tdiffs = [None, -0.353, -0.351, -0.349, -0.347]
    process_elevation_angle(rad, dates, tdiffs)
    pass