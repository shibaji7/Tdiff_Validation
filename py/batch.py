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
            cols = ["srange", "trad_gsflg", "ribiero_gflg", "v", "w_l", "tfreq", "bmnum"]
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
        Zj = Z / np.nansum(Z)
        iX, zX = [], []
        for i in range(Zx.shape[0]):
            Zx[i,:] = Z[i,:]/np.nansum(Z[i,:])
            iX.append(np.nanargmax(Zx[i,:]))
        Zx[Zx<0.01] = np.nan
        Y0 = np.zeros_like(Y)
        for i in range(Y.shape[1]):
            if type(vh) is float: Y0[:,i] = CV.calculate_vHeight(Y[:,i], X[:,i], 0.5)
            if type(vh) is str: Y0[:,i] = vh_type_calc(Y[:,i], X[:,i], gkind)
        for i in range(Y0.shape[0]):
            zX.append(Y0[i,0])
        return x, y, X, Y, Y0, Zx, Zj, zX
    
    ii, labs = 1, [("(a)","Trad"), ("(b)","Trad"), ("(c)","ML"), ("(d)","ML")]
    fig0, axs0 = plt.figure(figsize=(5,4), dpi=240), []
    fig1, axs1 = plt.figure(figsize=(5,4), dpi=240), []
    fig2, axs2 = plt.figure(figsize=(5,3), dpi=240), []
    for gtype in ["trad_gsflg", "ribiero_gflg"]:
        for gkind in [0, 1]:
            x, y, X, Y, Y0, Zx, Zj, zX = get_query(gtype,gkind,th)
            Zj = gaussian_filter(Zj, 1)
            
            ax0 = fig0.add_subplot(220+ii)
            im0 = ax0.pcolormesh(X, Y, Zj, lw=0.01, edgecolors="None", cmap="jet", 
                                 norm=mcolors.LogNorm(1e-5,1e-2))
            ax0.set_ylim(0,4000)
            ax0.set_xlim(0,50)
            axs0.append(ax0)
            ax0.text(.99, .9, "IS" if gkind==0 else "GS", ha="right", va="center", 
                     transform=ax0.transAxes, fontdict={"color":"k"})
            ax0.text(.05, .9, labs[ii-1][0], ha="left", va="center", transform=ax0.transAxes, 
                     fontdict={"color":"w"})
            ax0.text(.99, 1.05, "Method: %s"%labs[ii-1][1], ha="right", va="center", 
                     transform=ax0.transAxes, fontdict={"color":"k"})
            if (ii == 1) or (ii == 2): ax0.set_xticklabels([])
            if (ii == 2) or (ii == 4): ax0.set_yticklabels([])
                
            ax1 = fig1.add_subplot(220+ii)
            im1 = ax1.pcolormesh(X.T, Y.T, Zx.T, lw=0.01, edgecolors="None", cmap="Reds",
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
                
            ax2 = fig2.add_subplot(220+ii)
            im2 = ax2.pcolormesh(Y.T, Y0.T, Zx.T, lw=0.01, edgecolors="None", cmap="Reds", 
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
    cb0.set_label("Probability")
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
    return

def plot_ElRa_2Dhist(gtype, gkind, vh):
    import glob
    from operator import and_
    files = glob.glob("outputs/ElRa/*%s-%d.csv"%(gtype,gkind))
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
    tdiff = -0.347
    
    
    X, Y  = np.meshgrid( x, y )
    Zx = np.zeros_like(Z)
    Zj = Z / np.nansum(Z)
    for i in range(Zx.shape[0]):
        Zx[i,:] = Z[i,:]/np.nansum(Z[i,:])
    
    
    fig = plt.figure(figsize=(5,4), dpi=150)
    ax = fig.add_subplot(111)
    Zj = gaussian_filter(Zj, 1)
    im = ax.pcolormesh(X, Y, Zj, lw=0.01, edgecolors="None", cmap="jet", norm=mcolors.LogNorm(1e-5,1e-1))
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Probability")
    ax.set_ylim(0,4000)
    ax.set_xlim(0,50)
    fig.subplots_adjust(hspace=.3, wspace=0.5)
    ax.set_ylabel("Slant Range, km")
    ax.set_xlabel("Elevation Angle, degrees")
    ax.text(.9, .9, "IS" if gkind==0 else "GS", ha="left", va="center", transform=ax.transAxes)
    fig.savefig("tmp/H2D.ElRa.%s-%d.png"%(gtype,gkind), bbox_inches="tight")
    
    fig = plt.figure(figsize=(5,4), dpi=150)
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X.T, Y.T, Zx.T, lw=0.01, edgecolors="None", cmap="Reds", vmax=0.04, vmin=0.01)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Probability")
    ax.set_ylim(0,4000)
    ax.set_xlim(0,50)
    ax.set_ylabel("Slant Range, km")
    ax.set_xlabel("Elevation Angle, degrees")
    ax.text(.9, .9, "IS" if gkind==0 else "GS", ha="left", va="center", transform=ax.transAxes)
    fig.subplots_adjust(hspace=.3, wspace=0.5)
    fig.savefig("tmp/H2D.ElRa-N.%s-%d.png"%(gtype,gkind), bbox_inches="tight")
    
    Y0 = np.zeros_like(Y)
    for i in range(Y.shape[1]):
        if type(vh) is float: Y0[:,i] = CV.calculate_vHeight(Y[:,i], X[:,i], 0.5)
        if type(vh) is str: Y0[:,i] = vh_type_calc(Y[:,i], X[:,i], gkind)
    vh = "half" if type(vh) is float else "comp"
    fig = plt.figure(figsize=(5,2.5), dpi=150)
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(Y.T, Y0.T, Zx.T, lw=0.01, edgecolors="None", cmap="Reds", vmax=0.04, vmin=0.01)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Probability")
    ax.set_ylim(0,2000)
    ax.set_xlim(0,4000)
    ax.set_ylabel("Slant Range, km")
    ax.set_xlabel("Elevation Angle, degrees")
    ax.text(.9, .9, "IS" if gkind==0 else "GS", ha="left", va="center", transform=ax.transAxes)
    fig.subplots_adjust(hspace=.3, wspace=0.5)
    fig.savefig("tmp/H2D.ElRa-N.%s-%d.%s.png"%(gtype,gkind,vh), bbox_inches="tight")
    return

def plot_VhRa_2Dhist(gtype, gkind, vh):
    import glob
    from operator import and_
    print("outputs/VhRa/*%s-%d.%s.csv"%(gtype,gkind,vh))
    files = glob.glob("outputs/VhRa/*%s-%d.%s.csv"%(gtype,gkind,vh))
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
    for i in range(Zx.shape[1]):
        Zx[:,i] = Z[:,i]/np.nansum(Z[:,i])
        
    fig = plt.figure(figsize=(5,2.5), dpi=150)
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(Y, X, Zx, lw=0.01, edgecolors="None", cmap="Reds", vmax=0.06, vmin=0.01)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Probability")
    ax.set_ylim(0,2000)
    ax.set_xlim(0,4000)
    ax.set_xlabel("Slant Range, km")
    ax.set_ylabel("Virtual Height, km")
    SR = np.linspace(0,4000,4001)
    CH_model_val = [CV.chisham(x) for x in SR]
    ax.plot(SR, CH_model_val, ls="--", lw=0.8, color="k")
    ax.text(.9, .9, "IS" if gkind==0 else "GS", ha="left", va="center", transform=ax.transAxes)
    fig.subplots_adjust(hspace=.3, wspace=0.5)
    fig.savefig("tmp/H2D.VhRa.%s-%d.%s.png"%(gtype,gkind,vh), bbox_inches="tight")
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

def process_monthly_VhRa_files(year=2014, month="Jan", gtypes=["trad_gsflg", "ribiero_gflg"], 
                               gkinds=[0, 1], vhs=["vh0","trd_vh0_sel","new_vh0_sel"]):
    def fetch_monthly_data():
        import glob
        files = glob.glob("outputs/cluster_tags/*%s*%d.csv"%(month, year))
        files.sort()
        o = pd.DataFrame()
        for i, f in enumerate(files):
            o = pd.concat([o, pd.read_csv(f)])
        return o
    
    d = pd.DataFrame()
    for gtype in gtypes:
        for gkind in gkinds:
            for vh in vhs:
                fname = "outputs/VhRa/%s-%d.%s-%d.%s.csv"%(month,year,gtype,gkind,vh)
                print(fname)
                if not os.path.exists(fname):
                    if len(d)==0: d = fetch_monthly_data()
                    du = d[d[gtype]==gkind]
                    du["binsrang"] = du.srange.apply(lambda x: 45*int(x/45))
                    du["bin"+vh] = du[vh].apply(lambda x: 10*int(x/10))
                    du = du.groupby( ["binsrang", "bin"+vh] ).size().reset_index(name="Size")
                    du = du[ ["binsrang", "bin"+vh, "Size"] ].pivot( "binsrang", "bin"+vh )
                    du.to_csv(fname)
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
                #process_monthly_ElRa_files(year=y, month=m, th=50)
                #process_1D_histogram_dataset(year=y, month=m)
                pass
        plot_comp_plots(vh="comp", th=50)
#         plot_ElRa_2Dhist(gtype="trad_gsflg",gkind=0,vh=0.5)
#         plot_ElRa_2Dhist(gtype="trad_gsflg",gkind=1,vh=0.5)
#         plot_ElRa_2Dhist(gtype="ribiero_gflg",gkind=0,vh=0.5)
#         plot_ElRa_2Dhist(gtype="ribiero_gflg",gkind=1,vh=0.5) 
#         #plot_1D_velocity_histogram()
#         plot_VhRa_2Dhist(gtype="trad_gsflg",gkind=0,vh="vh0")
#         plot_VhRa_2Dhist(gtype="trad_gsflg",gkind=1,vh="vh0")
#         plot_VhRa_2Dhist(gtype="trad_gsflg",gkind=0,vh="trd_vh0_sel")
#         plot_VhRa_2Dhist(gtype="trad_gsflg",gkind=1,vh="trd_vh0_sel")
#         plot_VhRa_2Dhist(gtype="ribiero_gflg",gkind=0,vh="new_vh0_sel")
#         plot_VhRa_2Dhist(gtype="ribiero_gflg",gkind=1,vh="new_vh0_sel")