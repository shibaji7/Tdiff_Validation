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
from plotlib import RangeTimeIntervalPlot as RTI, Histogram2D
import matplotlib.dates as mdates
import calc_vheight as CV

def process_elevation_angle(rad, dates, tdiffs, bm=None, 
                            tfreq_range=None, gflag=None,
                            hop=None, sranges=None, timerange=None):
    """
    Main process to run Elevaton angle
    """
    fdata = FetchData( rad, dates )
    b, _ = fdata.fetch_data()
    d = fdata.convert_to_pandas(b)
    d["srange"] = d.frang + (d.rsep*d.slist)
    d.tfreq = np.round(d.tfreq/1e3, 1)
    d["mdates"] = d.time.apply(lambda x: mdates.date2num(x))
    bm = 9 if (bm==None) else bm
    hop = 0.5 if (hop==None) else hop
    if tfreq_range: d = d[(d.tfreq>=tfreq_range[0]) & (d.tfreq<=tfreq_range[1])]
    if gflag: d = d[(d.gflg==1)]
    hdw = pydarn.read_hdw_file(rad)
    d.tfreq = d.tfreq*1e3
    figtitle = "Rad: %s, Beam: %d, %s"%(rad.upper(), bm, dates[0].strftime("%d %b %Y"))
    rti = RTI(100, np.array(d.mdates), figtitle, num_subplots=len(tdiffs))
    h2d = Histogram2D(nrows=2, fig_title=figtitle)
    for i, tdiff in enumerate(tdiffs):
        o = d.copy()
        o["elv"+str(i)] = calc_elv.caclulate_elevation_angle(np.array(o.phi0), np.array(o.bmnum),
                                                             np.array(o.tfreq), hdw, tdiff)
        o["vh"+str(i)] = CV.calculate_vHeight(np.array(o.srange), np.array(o["elv"+str(i)]), hop=hop)
        tdiff = hdw.tdiff if tdiff is None else tdiff
        rti.addParamPlot(o, bm, r"$T_{diff}$=%.3f $\mu s$"%tdiff, zparam="elv"+str(i))
        if sranges: o = o[(o.srange>=sranges[0]) & (o.srange<=sranges[1])]
        if timerange: o = o[(o.time>=timerange[0]) & (o.time<=timerange[1])]        
        h2d.addHistPlot(o, r"$T_{diff}$=%.3f $\mu s$"%tdiff, yl="vh"+str(i))
    rti.save("tmp/elevation.png")
    rti.close()
    h2d.save("tmp/2d-hist.png")
    h2d.close()
    return

if __name__ == "__main__":
    if not os.path.exists("tmp/"): os.mkdir("tmp/")
    rad = "cvw"
    dates = [dt.datetime(2014,4,23), dt.datetime(2014,4,23,16)]
    tdiffs = [None, -0.347]
    tfreq_range, gflag, bm, hop = [10.3, 10.8], 1, 9, 1.
    sranges, timerange = [1000, 3000], [dt.datetime(2014,4,23,13), dt.datetime(2014,4,23,14,15)]
    process_elevation_angle(rad, dates, tdiffs, bm, tfreq_range, gflag, hop, sranges, timerange)
    pass