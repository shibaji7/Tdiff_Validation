#!/usr/bin/env python

"""batch.py: Analysis module to elevation angle data."""

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
import pandas as pd

from multiprocessing import Pool

import pydarn
from get_fit_data import FetchData
import calc_elv
from plotlib import RangeTimeIntervalPlot as RTI, Histogram2D
import matplotlib.dates as mdates
import calc_vheight as CV
import utils

def fetch_remote_dataset(date, rad):
    conn = utils.get_session(key_filename=utils.get_pubfile())
    a_name = "dbscan"
    fname = "outputs/cluster_tags_def_params/{rad}.{a_name}.gmm.{dn}.csv"
    LFS = "LFS/LFS_clustering_superdarn_data/"
    floc = fname.format(rad=rad, a_name=a_name, dn=date.strftime("%Y%m%d"))
    if not os.path.exists(floc): utils.fetch_file(conn, floc, LFS)
    o = pd.read_csv(floc)
    os.remove(floc)
    return o

def vh_type_calc(srange, elv, gflg):
    vh_type = np.zeros_like(srange)
    # IS and 0.5 hop
    idx = (gflg==0) & (srange<=2250)
    vh_type[idx] = CV.calculate_vHeight(srange[idx], elv[idx], hop=0.5)
    # IS and 1.5 hop
    idx = (gflg==0) & (srange>2250)
    vh_type[idx] = CV.calculate_vHeight(srange[idx], elv[idx], hop=1.5)
    # GS and 1.0 hop
    idx = (gflg==1) & (srange<=3240)
    vh_type[idx] = CV.calculate_vHeight(srange[idx], elv[idx], hop=1.)
    # GS and 2 hop
    idx = (gflg==1) & (srange>3240)
    vh_type[idx] = CV.calculate_vHeight(srange[idx], elv[idx], hop=2.)
    return vh_type

def process_elevation_angle(date):
    fname = "outputs/cluster_tags_def_params/%s.csv"%date.strftime("%d-%b-%Y")
    if not os.path.exists(fname):
        tdiffs = [-0.347]
        rad = "cvw"
        fdata = FetchData( rad, [date, date+dt.timedelta(1)] )
        b, _ = fdata.fetch_data()
        d = fdata.convert_to_pandas(b)
        d["srange"] = d.frang + (d.rsep*d.slist)
        # Fetch remore algo output files
        o = fetch_remote_dataset(date, rad)
        if len(o) == len(d):
            d["ribiero_gflg"] = o.ribiero_gflg
            hdw = pydarn.read_hdw_file(rad)
            cols = ["srange", "gflg", "ribiero_gflg"]
            cols.extend(["elv"+str(i) for i in range(len(tdiffs))])
            cols.extend(["vh"+str(i) for i in range(len(tdiffs))])
            cols.extend(["trd_vh"+str(i)+"_sel" for i in range(len(tdiffs))])
            cols.extend(["new_vh"+str(i)+"_sel" for i in range(len(tdiffs))])
            for i, tdiff in enumerate(tdiffs):
                d["elv"+str(i)] = calc_elv.caclulate_elevation_angle(np.array(d.phi0), np.array(d.bmnum),
                                                                     np.array(d.tfreq), hdw, tdiff)
                d["vh"+str(i)] = CV.calculate_vHeight(np.array(d.srange), np.array(d["elv"+str(i)]), hop=0.5)

                # Traditional IS/GS analysis
                # Selected range-wise VH calc
                srange, elv, gflg = np.array(d.srange), np.array(d["elv"+str(i)]), np.array(d.gflg)
                vh_type = vh_type_calc(srange, elv, gflg)
                d["trd_vh"+str(i)+"_sel"] = vh_type

                # New algo IS/GS analysis
                # Selected range-wise VH calc
                gflg = np.array(o.ribiero_gflg)
                vh_type = vh_type_calc(srange, elv, gflg)
                d["new_vh"+str(i)+"_sel"] = vh_type
            d[cols].to_csv(fname, index=False, header=True, float_format="%g")
    return

if __name__ == "__main__":
    sdate, edate = [dt.datetime(2012,1,1), dt.datetime(2012,10,1)]
    dates = []
    while sdate <= edate:
        dates.append(sdate)
        sdate += dt.timedelta(1)
    with Pool(8) as p:
        p.map(process_elevation_angle, dates)