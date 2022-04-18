#!/usr/bin/env python

"""plot_lib.py: module is dedicated to plot and create the movies."""

__author__ = "Chakraborty, S."
__copyright__ = "Copyright 2020, SuperDARN@VT"
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "shibaji7@vt.edu"
__status__ = "Research"


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

import numpy as np

def get_gridded_parameters(q, xparam="beam", yparam="slist", zparam="v", r=0, rounding=True):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    if rounding:
        plotParamDF.loc[:, xparam] = np.round(plotParamDF[xparam].tolist(), r)
        plotParamDF.loc[:, yparam] = np.round(plotParamDF[yparam].tolist(), r)
    else:
        plotParamDF[xparam] = plotParamDF[xparam].tolist()
        plotParamDF[yparam] = plotParamDF[yparam].tolist()
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).mean().reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z


CLUSTER_CMAP = plt.cm.gist_rainbow

def get_cluster_cmap(n_clusters, plot_noise=False):
    cmap = CLUSTER_CMAP
    cmaplist = [cmap(i) for i in range(cmap.N)]
    while len(cmaplist) < n_clusters:
        cmaplist.extend([cmap(i) for i in range(cmap.N)])
    cmaplist = np.array(cmaplist)
    r = np.array(range(len(cmaplist)))
    random.seed(10)
    random.shuffle(r)
    cmaplist = cmaplist[r]
    if plot_noise:
        cmaplist[0] = (0, 0, 0, 1.0)    # black for noise
    rand_cmap = cmap.from_list("Cluster cmap", cmaplist, len(cmaplist))
    return rand_cmap
    
class RangeTimeIntervalPlot(object):
    """
    Create plots for velocity, width, power, elevation angle, etc.
    """
    
    def __init__(self, nrang, unique_times, fig_title="", num_subplots=5):
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=150) # Size for website
        plt.suptitle(fig_title, x=0.5, y=0.9, ha="center", va="center", fontweight="bold", fontsize=15)
        matplotlib.rcParams.update({"font.size": 10})
        return
    
    def addParamPlot(self, df, beam, title, p_max=40, p_min=0, p_step=8, xlabel="Time UT", zparam="elv0",
                    label=r"Elevation $[^o]$"):
        ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = get_gridded_parameters(df, xparam="mdates", yparam="slist", zparam=zparam, rounding=False)
        bounds = list(range(p_min, p_max+1, p_step))
        cmap = plt.cm.Spectral_r
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
        hours = mdates.HourLocator(byhour=range(0, 24, 4))
        ax.xaxis.set_major_locator(hours)
        ax.set_xlabel(xlabel, fontdict={"size":12, "fontweight": "bold"})
        ax.set_xlim([self.unique_times[0], self.unique_times[-1]])
        ax.set_ylim([0, self.nrang])
        ax.set_ylabel("Range gate", fontdict={"size":12, "fontweight": "bold"})
        im = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap, vmax=p_max, vmin=p_min)
        cb = self.fig.colorbar(im, ax=ax, shrink=0.7)
        cb.set_label(label)
        #self._add_colorbar(self.fig, ax, bounds, cmap, label=label)
        ax.set_title(title, loc="left", fontdict={"fontweight": "bold"})
        return ax, im
    
    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        return ax
    
    def _add_colorbar(self, fig, ax, bounds, colormap, label=""):
        """
        Add a colorbar to the right of an axis.
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + 0.025, pos.y0 + 0.0125,
                0.015, pos.height * 0.9]                # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        norm = mpl.colors.BoundaryNorm(bounds, colormap.N)
        cb2 = mpl.colorbar.ColorbarBase(cax, cmap=colormap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing="uniform",
                                        orientation="vertical")
        cb2.set_label(label)
        return
    
    def save(self, filepath):
        print(f"Save RTI figure to : {filepath}")
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return