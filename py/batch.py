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
from scipy.ndimage import gaussian_filter
import pickle
from matplotlib.dates import num2date

import pydarn
from get_fit_data import FetchData
import calc_elv
from plotlib import RangeTimeIntervalPlot as RTI, Histogram2D
import matplotlib.dates as mdates
import calc_vheight as CV
import utils

import matplotlib
import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import random
from matplotlib import patches
import matplotlib.patches as mpatches

def sunriseset(rad, dates):
    import rad_fov
    import pydarn
    from astral.sun import sun
    from astral import Observer
    dumin, ix = 15, 0
    suntime = {"doy":[], "sunrise":[], "sunset":[], "index":[]}
    hdw = pydarn.read_hdw_file(rad)
    rfov = rad_fov.CalcFov(hdw=hdw, ngates=100)
    lat, lon = np.mean(rfov.latFull.ravel()) , np.mean(rfov.lonFull.ravel())
    for _i, dn in enumerate(dates):
        o = Observer(rfov.latFull[0,0],rfov.lonFull[0,0])
        s = sun(o, date=dn)
        #suntime["doy"].append(dn.dayofyear)
        suntime["index"].append(_i)
        suntime["sunrise"].append(s["sunrise"].hour+1 + (dumin/60.)*int(s["sunrise"].minute/dumin))
        suntime["sunset"].append(s["sunset"].hour+1 + (dumin/60.)*int(s["sunset"].minute/dumin))
    return suntime

def fetch_remote_dataset(date, rad):
    conn = utils.get_session(key_filename=utils.get_pubfile())
    a_name = "dbscan"
    fname = "outputs/cluster_tags/{rad}.{a_name}.gmm.{dn}.csv"
    LFS = "LFS/LFS_clustering_SD_data/"
    floc = fname.format(rad=rad, a_name=a_name, dn=date.strftime("%Y%m%d"))
    if not os.path.exists(floc): utils.fetch_file(conn, floc, LFS)
    if os.path.exists(floc): 
        o = pd.read_csv(floc)
        os.remove(floc)
    else: o = pd.DataFrame()
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
    fname = "outputs/cluster_tags/%s.csv"%date.strftime("%d-%b-%Y")
    if not os.path.exists(fname):
        tdiffs = [-0.347]
        rad = "cvw"
        # Fetch remore algo output files
        d = fetch_remote_dataset(date, rad)
        if ("ribiero_gflg" in d.columns.tolist()):
            hdw = pydarn.read_hdw_file(rad)
            cols = ["srange", "trad_gsflg", "ribiero_gflg", "v", "w_l", "tfreq", "bmnum", "time"]
            cols.extend(["elv"+str(i) for i in range(len(tdiffs))])
            cols.extend(["vh"+str(i) for i in range(len(tdiffs))])
            cols.extend(["trd_vh"+str(i)+"_sel" for i in range(len(tdiffs))])
            cols.extend(["new_vh"+str(i)+"_sel" for i in range(len(tdiffs))])
            for i, tdiff in enumerate(tdiffs):
                d["elv"+str(i)] = calc_elv.caclulate_elevation_angle(np.array(d.phi0), np.array(d.bmnum),
                                                                     np.array(d.tfreq*1e3), hdw, tdiff)
                d["vh"+str(i)] = CV.calculate_vHeight(np.array(d.srange), np.array(d["elv"+str(i)]), hop=0.5)

                # Traditional IS/GS analysis
                # Selected range-wise VH calc
                srange, elv, gflg = np.array(d.srange), np.array(d["elv"+str(i)]), np.array(d.trad_gsflg)
                vh_type = vh_type_calc(srange, elv, gflg)
                d["trd_vh"+str(i)+"_sel"] = vh_type

                # New algo IS/GS analysis
                # Selected range-wise VH calc
                gflg = np.array(d.ribiero_gflg)
                vh_type = vh_type_calc(srange, elv, gflg)
                d["new_vh"+str(i)+"_sel"] = vh_type
            d[cols].to_csv(fname, index=False, header=True, float_format="%g")
    return

def plot_comp_plots(vh=0.5, th=41):
    tdiff = -0.347
    Zmax = []
    def get_query(gtype,gkind,th):
        import glob
        from operator import and_
        files = glob.glob("outputs/ElRa/*%s-%d.th-%d.csv"%(gtype,gkind,th))
        for i, f in enumerate(files):
            try:
                if i ==0: 
                    o = pd.read_csv(f, index_col=0)
                    x, y = np.array(o.iloc[0]), np.array([float(m) for m in o.index.tolist()[2:]])
                    Z = o.values[2:,:]
                else: 
                    o = pd.read_csv(f, index_col=0)
                    z = np.ma.masked_invalid(o.values[2:,:])
                    Z = np.nansum(np.dstack((Z,z)),2)
            except: pass
        
        X, Y  = np.meshgrid( x, y )
        Zx = np.zeros_like(Z)
        Zj = Z
        Zmax.append(np.nansum(Z))
        for i in range(Zx.shape[0]):
            Zx[i,:] = Z[i,:]/np.nansum(Z[i,:])
        #Zx[Zx<0.001] = np.nan
        Y0 = np.zeros_like(Y)
        for i in range(Y.shape[1]):
            if type(vh) is float: Y0[:,i] = CV.calculate_vHeight(Y[:,i], X[:,i], 0.5)
            if type(vh) is str: Y0[:,i] = vh_type_calc(Y[:,i], X[:,i], gkind)
        return x, y, X, Y, Y0, Zx, Zj
    
    ii, labs = 1, [("(a)","Trad"), ("(b)","Trad"), ("(c)","ML"), ("(d)","ML")]
    fig0, axs0 = plt.figure(figsize=(5,4), dpi=300), []
    fig1, axs1 = plt.figure(figsize=(5,4), dpi=300), []
    fig2, axs2 = plt.figure(figsize=(5,3), dpi=300), []
    fig4, axs4 = plt.figure(figsize=(5,4), dpi=300), []
    for gtype in ["trad_gsflg", "ribiero_gflg"]:
        for gkind in [0, 1]:
            _ = get_query(gtype,gkind,th)
    for gtype in ["trad_gsflg", "ribiero_gflg"]:
        for gkind in [0, 1]:
            x, y, X, Y, Y0, Zx, Zj = get_query(gtype,gkind,th)
            Zj = gaussian_filter(Zj, 1)
            
            ax0 = fig0.add_subplot(220+ii)
            im0 = ax0.pcolormesh(X, Y, Zj/np.max(Zmax), edgecolors="None", cmap="jet", 
                                 norm=mcolors.LogNorm(1e-5,1e-2))
            ax0.set_ylim(0,4000)
            ax0.set_xlim(0,th)
            axs0.append(ax0)
            ax0.text(.99, .9, "IS" if gkind==0 else "GS", ha="right", va="center", 
                     transform=ax0.transAxes, fontdict={"color":"w"})
            ax0.text(.05, .9, labs[ii-1][0], ha="left", va="center", transform=ax0.transAxes, 
                     fontdict={"color":"w"})
            ax0.text(.99, 1.05, "Method: %s"%labs[ii-1][1], ha="right", va="center", 
                     transform=ax0.transAxes, fontdict={"color":"k"})
            if (ii == 1) or (ii == 2): ax0.set_xticklabels([])
            if (ii == 2) or (ii == 4): ax0.set_yticklabels([])
            ax0.set_xlim(0,th)
            
            ax4 = fig4.add_subplot(220+ii)
            im4 = ax4.pcolormesh(X, Y, Zj/np.nansum(Zj), edgecolors="None", cmap="jet", 
                                 norm=mcolors.LogNorm(1e-5,1e-2))
            ax4.set_ylim(0,4000)
            ax4.set_xlim(0,th)
            axs4.append(ax4)
            ax4.text(.99, .9, "IS" if gkind==0 else "GS", ha="right", va="center", 
                     transform=ax4.transAxes, fontdict={"color":"w"})
            ax4.text(.05, .9, labs[ii-1][0], ha="left", va="center", transform=ax4.transAxes, 
                     fontdict={"color":"w"})
            ax4.text(.99, 1.05, "Method: %s"%labs[ii-1][1], ha="right", va="center", 
                     transform=ax4.transAxes, fontdict={"color":"k"})
            if (ii == 1) or (ii == 2): ax4.set_xticklabels([])
            if (ii == 2) or (ii == 4): ax4.set_yticklabels([])
            
                
            Zxi = gaussian_filter(Zx, 1)
            Zxi[Zxi<0.012] = np.nan
            ax1 = fig1.add_subplot(220+ii)
            im1 = ax1.pcolormesh(X.T, Y.T, Zxi.T, lw=0.01, edgecolors="None", cmap="Reds",
                                 vmax=0.04, vmin=0.01)
            ax1.set_ylim(0,4000)
            ax1.set_xlim(0,50)
            axs1.append(ax1)
            ax1.text(.99, .9, "IS" if gkind==0 else "GS", ha="right", va="center", 
                     transform=ax1.transAxes, fontdict={"color":"k"})
            ax1.text(.05, .9, labs[ii-1][0], ha="left", va="center", transform=ax1.transAxes, 
                     fontdict={"color":"k"})
            ax1.text(.99, 1.05, "Method: %s"%labs[ii-1][1], ha="right", va="center", 
                     transform=ax1.transAxes, fontdict={"color":"k"})
            if (ii == 1) or (ii == 2): ax1.set_xticklabels([])
            if (ii == 2) or (ii == 4): ax1.set_yticklabels([])
                
            Zxj = gaussian_filter(Zx, 1.5)
            Zxj[Zxj<0.012] = np.nan
            ax2 = fig2.add_subplot(220+ii)
            im2 = ax2.pcolormesh(Y.T, Y0.T, Zxj.T, lw=0.01, edgecolors="None", cmap="Reds", 
                                 vmax=0.04, vmin=0.01)
            ax2.plot(y, [CV.chisham(yi) for yi in y], color="w", lw=1, ls="-")
            ax2.plot(y, [CV.chisham(yi) for yi in y], color="b", lw=0.9, ls="-")
            ax2.plot(y, [CV.thomas(yi, gkind) for yi in y], color="w", lw=1, ls="-")
            ax2.plot(y, [CV.thomas(yi, gkind) for yi in y], color="darkgreen", lw=0.9, ls="-")
            ax2.set_ylim(0,1000)
            ax2.set_xlim(0,4000)
            axs2.append(ax2)
            ax2.text(.99, .9, "IS" if gkind==0 else "GS", ha="right", va="center", 
                     transform=ax2.transAxes, fontdict={"color":"k"})
            ax2.text(.05, .9, labs[ii-1][0], ha="left", va="center", transform=ax2.transAxes, 
                     fontdict={"color":"k"})
            ax2.text(.99, 1.05, "Method: %s"%labs[ii-1][1], ha="right", va="center", 
                     transform=ax2.transAxes, fontdict={"color":"k"})
            if (ii == 1) or (ii == 2): ax2.set_xticklabels([])
            if (ii == 2) or (ii == 4): ax2.set_yticklabels([])
            
            ii += 1
    cax = ax0.inset_axes([1.04, 0.1, 0.05, 0.8], transform=ax0.transAxes)
    cb0 = fig0.colorbar(im0, ax=ax0, cax=cax)
    cb0.set_label("Occurrence Rate")
    axs0[0].set_ylabel("Slant Range, km")
    axs0[2].set_ylabel("Slant Range, km")
    axs0[2].set_xlabel("Elevation Angle, degrees")
    axs0[3].set_xlabel("Elevation Angle, degrees")
    axs0[0].text(.05, 1.05, r"$T_{diff}=%d$ ns"%(tdiff*1e3), ha="left", va="center", 
                 transform=axs0[0].transAxes)
    fig0.subplots_adjust(hspace=.2, wspace=.2)
    fig0.savefig("tmp/H2D.ElRa.png", bbox_inches="tight")
    
    cax = ax1.inset_axes([1.04, 0.1, 0.05, 0.8], transform=ax1.transAxes)
    cb1 = fig1.colorbar(im1, ax=ax1, cax=cax)
    cb1.set_label("Probability")
    axs1[0].set_ylabel("Slant Range, km")
    axs1[2].set_ylabel("Slant Range, km")
    axs1[2].set_xlabel("Elevation Angle, degrees")
    axs1[3].set_xlabel("Elevation Angle, degrees")
    axs1[0].text(.05, 1.05, r"$T_{diff}=%d$ ns"%(tdiff*1e3), ha="left", va="center", 
                 transform=axs1[0].transAxes)
    fig1.subplots_adjust(hspace=.2, wspace=.2)
    fig1.savefig("tmp/H2D.ElRaN.png", bbox_inches="tight")
    
    cax = ax2.inset_axes([1.04, 0.1, 0.05, 0.8], transform=ax2.transAxes)
    cb2 = fig2.colorbar(im2, ax=ax2, cax=cax)
    cb2.set_label("Probability")
    axs2[0].set_ylabel("Virtual Height, km")
    axs2[2].set_ylabel("Virtual Height, km")
    axs2[2].set_xlabel("Slant Range, km")
    axs2[3].set_xlabel("Slant Range, km")
    axs2[0].text(.05, 1.05, r"$T_{diff}=%d$ ns"%(tdiff*1e3), ha="left", va="center", 
                 transform=axs2[0].transAxes)
    fig2.subplots_adjust(hspace=.2, wspace=.2)
    fig2.savefig("tmp/H2D.VhRaN.png", bbox_inches="tight")
    
    cax = ax4.inset_axes([1.04, 0.1, 0.05, 0.8], transform=ax0.transAxes)
    cb4 = fig4.colorbar(im4, ax=ax4, cax=cax)
    cb4.set_label("Probability")
    axs4[0].set_ylabel("Slant Range, km")
    axs4[2].set_ylabel("Slant Range, km")
    axs4[2].set_xlabel("Elevation Angle, degrees")
    axs4[3].set_xlabel("Elevation Angle, degrees")
    axs4[0].text(.05, 1.05, r"$T_{diff}=%d$ ns"%(tdiff*1e3), ha="left", va="center", 
                 transform=axs4[0].transAxes)
    fig4.subplots_adjust(hspace=.2, wspace=.2)
    fig4.savefig("tmp/H2D.ElRaP.png", bbox_inches="tight")
    return

def process_monthly_ElRa_files(year=2014, month="Jan", gtypes=["trad_gsflg", "ribiero_gflg"], 
                               gkinds=[0, 1], th=41):
    def fetch_monthly_data():
        import glob
        files = glob.glob("outputs/cluster_tags/*%s*%d.csv"%(month, year))
        files.sort()
        o = pd.DataFrame()
        for i, f in enumerate(files):
            o = pd.concat([o, pd.read_csv(f)])
        return o
    
    d = None
    for gtype in gtypes:
        for gkind in gkinds:
            fname = "outputs/ElRa/%s-%d.%s-%d.th-%d.csv"%(month,year,gtype,gkind,th)
            if not os.path.exists(fname):
                if d is None: d = fetch_monthly_data()
                du = d[(d[gtype]==gkind) & (d.elv0<=th)]
                du.drop(du[du.ribiero_gflg==-1].index, inplace=True)
                du["binsrang"] = du.srange.apply(lambda x: 45*int(x/45))
                du["binelv0"] = du.elv0.apply(lambda x: 0.5*int(x/0.5))
                du = du.groupby( ["binsrang", "binelv0"] ).size().reset_index(name="Size")
                du = du[ ["binsrang", "binelv0", "Size"] ].pivot( "binsrang", "binelv0" )
                du.to_csv(fname)
    return

def process_2D_histogram_dataset(year=2014, month="Jan", gtypes=["trad_gsflg", "ribiero_gflg"], 
                               gkinds=[0, 1, -1]):
    import swifter
    def fetch_monthly_data():
        import glob
        files = glob.glob("outputs/cluster_tags/*%s*%d.csv"%(month, year))
        files.sort()
        o = pd.DataFrame()
        for i, f in enumerate(files):
            m = pd.read_csv(f)
            start = dt.datetime.strptime(f.split("/")[-1].replace(".csv", ""), "%d-%b-%Y")
            fdata = FetchData( "cvw", [start,start+dt.timedelta(1)] )
            b, _ = fdata.fetch_data()
            ox = fdata.convert_to_pandas(b)
            if len(ox) == len(m):
                m["month"] = ox.time.swifter.apply(lambda x: x.month)
                m["hour"] = ox.time.swifter.apply(lambda x: x.hour)
                o = pd.concat([o, m])
        return o
    
    fname = "outputs/H2D/%s-%d.pickle"%(month,year)
    o = {}
    d = None
    if not os.path.exists(fname):
        for gtype in gtypes:
            for gkind in gkinds:
                    if d is None: d = fetch_monthly_data()
                    du = d[(d[gtype]==gkind)]
                    #if gkind!=-1: du.drop(du[du.ribiero_gflg==-1].index, inplace=True)
                    H, xedges, yedges = np.histogram2d(du.hour, du.month, bins=[np.arange(25), np.arange(1,14)])
                    #print(H, xedges, yedges)
                    #print(H.shape, xedges, yedges)
                    #du = du.groupby( ["month", "hour"] ).size().reset_index(name="Size")
                    #du = du[ ["month", "hour", "Size"] ].pivot( "month", "hour" )
                    o[gtype+str(gkind)] = (H, xedges, yedges)
        with open(fname, "wb") as handle:
            pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def plot_2D_histogram():
    import glob
    files = glob.glob("outputs/H2D/*.pickle")
    dates = [dt.datetime(2014,1,1)+dt.timedelta(30*i) for i in range(60)]
    suntime = sunriseset("cvw", dates)
    files.sort()
    dataset = {}
    keys = ["trad_gsflg0", "trad_gsflg1", "ribiero_gflg0", "ribiero_gflg1"]
    years = [2014,2015,2016,2017,2018]
    for y in years:
        dataset[y] = {}
        for k in keys:
            dataset[y][k] = {"dat": None, "Y": None, "H": None}
    
    for f in files:
        y = int(f.split("-")[-1].split(".")[0])
        with open(f, "rb") as h: o = pickle.load(h)
        for k in keys:
            if dataset[y][k]["dat"] is None: 
                dataset[y][k]["dat"] = o[k][0]
                dataset[y][k]["H"], dataset[y][k]["Y"] = o[k][1], o[k][2]
            else: dataset[y][k]["dat"] += o[k][0]
    H, Y = [], []
    dat = {}
    for y in years:
        Y.extend( y + ((dataset[y]["trad_gsflg0"]["Y"][:-1]-1)/12) )
        for k in keys:
            if k not in dat.keys(): dat[k] = dataset[y][k]["dat"]
            else: dat[k] = np.hstack((dat[k],dataset[y][k]["dat"]))
    H.extend( dataset[y]["trad_gsflg0"]["H"][:-1] )
    H, Y = np.array(H), np.array(Y)
    fig0, axs0 = plt.figure(figsize=(5,4), dpi=300), []
    for ii, key in enumerate(keys):
        gkind = int(key[-1])
        ax0 = fig0.add_subplot(221+ii)
        ZZ = dat[key]
        im0 = ax0.pcolormesh(H, Y, ZZ.T/np.nansum(ZZ), edgecolors="None", 
                             cmap="jet", norm=mcolors.LogNorm(1e-5,1e-2))
        axs0.append(ax0)
        ax0.text(.99, 1.05, "IS" if gkind==0 else "GS", ha="right", va="center", 
                 transform=ax0.transAxes, fontdict={"color":"k"})
        ax0.set_yticks([2014, 2015, 2016, 2017, 2018, 2019])
        if (ii == 0) or (ii == 1): ax0.set_xticklabels([])
        if (ii == 1) or (ii == 3): ax0.set_yticklabels([])
        ax0.plot(suntime["sunrise"], Y, color="k", lw=0.8, ls="--")
        ax0.plot(suntime["sunset"], Y, color="k", lw=0.8, ls="--")
    cax = ax0.inset_axes([1.04, 0.1, 0.05, 0.8], transform=ax0.transAxes)
    cb0 = fig0.colorbar(im0, ax=ax0, cax=cax)
    cb0.set_label("Probability")
    axs0[0].set_ylabel("Years")
    axs0[2].set_ylabel("Years")
    axs0[2].set_xlabel("Hours, UT")
    axs0[3].set_xlabel("Hours, UT")
    axs0[0].text(.05, 1.05, r"$T_{diff}=%d$ ns"%(-347), ha="left", va="center", 
                 transform=axs0[0].transAxes)
    fig0.subplots_adjust(hspace=.2, wspace=.2)
    fig0.savefig("tmp/H2D.HrYr.png", bbox_inches="tight")
    return
    
def process_1D_histogram_dataset(year=2014, month="Jan", gtypes=["trad_gsflg", "ribiero_gflg"], 
                               gkinds=[0, 1, -1]):
    def fetch_monthly_data():
        import glob
        files = glob.glob("outputs/cluster_tags/*%s*%d.csv"%(month, year))
        files.sort()
        o = pd.DataFrame()
        for i, f in enumerate(files):
            o = pd.concat([o, pd.read_csv(f)])
        return o
    
    d = fetch_monthly_data()
    fname = "outputs/H1D/%s-%d.pickle"%(month,year)
    o = {}
    if not os.path.exists(fname):
        print(fname)
        ## TODO, save by month-year only
        for gtype in gtypes:
            for gkind in gkinds:
                o[gtype+str(gkind)] = {}
                v = np.abs(d[d[gtype]==gkind].v)
                if len(v) > 0:
                    v[v>=100.] = 101.
                    o[gtype+str(gkind)]["h"], o[gtype+str(gkind)]["be"] = np.histogram(v, bins=np.arange(100))
        with open(fname, "wb") as handle:
            pickle.dump(o, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def plot_1D_velocity_histogram():
    import glob
    from operator import and_
    files = glob.glob("outputs/H1D/*.pickle")
    O = {}
    for i, f in enumerate(files):
        print(f)
        try:
            print(i)
            with open(f, "rb") as fx: x = pickle.load(fx)
            if i ==0:
                O["trad_gsflg0"], be = x["trad_gsflg0"]["h"], x["trad_gsflg0"]["be"]
                O["trad_gsflg1"] = x["trad_gsflg1"]["h"]
                O["ribiero_gflg0"] = x["ribiero_gflg0"]["h"]
                O["ribiero_gflg1"] = x["ribiero_gflg1"]["h"]
                O["ribiero_gflg-1"] = x["ribiero_gflg-1"]["h"]
            else: 
                O["trad_gsflg0"] += x["trad_gsflg0"]["h"]
                O["trad_gsflg1"] += x["trad_gsflg1"]["h"]
                O["ribiero_gflg0"] += x["ribiero_gflg0"]["h"]
                O["ribiero_gflg1"] += x["ribiero_gflg1"]["h"]
                O["ribiero_gflg-1"] += x["ribiero_gflg-1"]["h"]
        except: pass
    fig, axes = plt.subplots(dpi=180, figsize=(6, 2.5), nrows=1, ncols=2, sharey=True)
    ax = axes[0]
    ax.plot(be[:-1], O["trad_gsflg1"], color="b", lw=0.8, ls="-", drawstyle="steps-pre", label="GS")
    ax.plot(be[:-1], O["trad_gsflg0"], color="r", lw=0.8, ls="-", drawstyle="steps-pre", label="IS")
    ax.set_ylabel("Histogram")
    ax.set_xlabel("Velocity (m/s)")
    ax.legend(loc=1, fontsize=8)
    ax.text(0.2, 0.9, "(a)", ha="left", va="center", transform=ax.transAxes, fontdict={"size":8})
    ax.set_xlim(0, 80)
    ax.set_yscale("log")
    ax.text(0.01,1.05, "Traditional", ha="left", va="center", transform=ax.transAxes)
    ax = axes[1]
    ax.plot(be[:-1], O["ribiero_gflg1"], color="b", lw=0.8, ls="-", drawstyle="steps-pre", label="GS")
    ax.plot(be[:-1], O["ribiero_gflg0"], color="r", lw=0.8, ls="-", drawstyle="steps-pre", label="IS")
    ax.plot(be[:-1], O["ribiero_gflg-1"], color="k", lw=0.8, ls="-", drawstyle="steps-pre", label="US")
    ax.set_xlabel("Velocity (m/s)")
    ax.legend(loc=1, fontsize=8)
    ax.set_xlim(0, 80)
    ax.set_yscale("log")
    ax.text(0.01,1.05, "DBSCAN-GMM + Ribeiro", ha="left", va="center", transform=ax.transAxes)
    ax.text(1.05,0.99, "2014-2018", ha="center", va="top", transform=ax.transAxes, rotation=90)
    ax.text(0.2, 0.9, "(b)", ha="left", va="center", transform=ax.transAxes, fontdict={"size":8})
    ax.set_ylim(1e5, 1e8)
    fig.subplots_adjust(wspace=0.3, hspace=0.5)
    fig.savefig("tmp/H1D.png", bbox_inches="tight")
    return

if __name__ == "__main__":
    isplot = True
    if not isplot:
        sdate, edate = [dt.datetime(2014,1,1), dt.datetime(2018,12,31)]
        dates = []
        while sdate <= edate:
            dates.append(sdate)
            sdate += dt.timedelta(1)
        with Pool(12) as p:
            p.map(process_elevation_angle, dates)
    else:
        years, months = [2014, 2015, 2016, 2017, 2018], ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for y in years:
            for m in months:
                #process_monthly_ElRa_files(year=y, month=m, th=41)
                #process_1D_histogram_dataset(year=y, month=m)
                #process_2D_histogram_dataset(year=y, month=m)
                pass
                #break
            #break
        #plot_comp_plots(vh="comp", th=41)
        plot_2D_histogram()