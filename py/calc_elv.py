""" calc_elv.py
   ==============
   Author: S. Chakraborty
   This file mimicks the operations described in elevation_v2.c of RST
"""

import numpy as np
from scipy import constants as scicon
from loguru import logger

def caclulate_elevation_angle(lag0, bmnum, tfreq, hdw, tdiff=None):
    """
    Parameters
    ----------
    lag0: Lag 0
    hwd: Radar Hardware configuration
    tdiff: External TDiff
    """
    tdiff = hdw.tdiff if tdiff is None else tdiff
    logger.info(f" Tdiff : {tdiff}")
    C = scicon.c
    
    # calculate the values that don't change if this hasn't already been done. 
    X, Y, Z = hdw.interferometer_offset[0], hdw.interferometer_offset[1], hdw.interferometer_offset[2]
    d = np.sqrt(X**2+Y**2+Z**2)
    
    # SGS: 20180926
    #
    # There is still some question as to exactly what the phidiff parameter in
    # the hdw.dat files means. The note in the hdw.dat files, presumably written
    # by Ray is:
    # 12) Phase sign (Cabling errors can lead to a 180 degree shift of the
    #     interferometry phase measurement. +1 indicates that the sign is
    #     correct, -1 indicates that it must be flipped.)
    # The _only_ hdw.dat file that has this value set to -1 is GBR during the
    # time period: 19870508 - 19921203
    #
    # To my knowlege there is no data available prior to 1993, so dealing with
    # this parameter is no longer necessary. For this reason I am simply
    # removing it from this algorithm.
   
    sgn = -1 if Y < 0 else 1
    boff = hdw.beams / 2.0 - 0.5
    phi0 = hdw.beam_separation * (bmnum - boff) * np.pi / 180.0
    cp0  = np.cos(phi0)
    sp0  = np.sin(phi0)
    
    # Phase delay [radians] due to electrical path difference.                
    #   If the path length (cable and electronics) to the interferometer is   
    #  shorter than that to the main antenna array, then the time for the    
    #  to transit the interferometer electrical path is shorter: tdiff < 0   
    psi_ele = -2.0 * np.pi * tfreq * tdiff * 1.0e-3
    
    # Determine elevation angle (a0) where psi (phase difference) is maximum; 
    #   which occurs when k and d are anti-parallel. Using calculus of        
    #  variations to compute the value: d(psi)/d(a) = 0                      
    a0 = np.arcsin(sgn * Z * cp0 / np.sqrt(Y**2 + Z**2))
    
    # Note: we are assuming that negative elevation angles are unphysical.    
    #  The act of setting a0 = 0 _only_ has the effect to change psi_max     
    #  (which is used to compute the correct number of 2pi factors and map   
    #  the observed phase to the actual phase. The _only_ elevation angles   
    #  that are affected are the small range from [-a0, 0]. Instead of these 
    #  being mapped to negative elevation they are mapped to very small      
    #  range just below the maximum.                                         

    # Note that it is possible in some cases with sloping ground that extends 
    #  far in front of the radar, that negative elevation angles might exist.
    #  However, since elevation angles near the maximum "share" this phase   
    #  [-pi,pi] it is perhaps more likely that the higher elevation angles   
    #  are actually what is being observed.                                  

    #In either case, one must decide which angle to chose (just as with all  
    #  the aliased angles). Here we decide (unless the keyword 'negative' is 
    #  set) that negative elevation angles are unphysical and map them to    
    #  the upper end.    
    a0[a0 < 0.] = 0.
    ca0 = np.cos(a0)
    sa0 = np.sin(a0)
    psi_obs = lag0
    
    # maximum phase = psi_ele + psi_geo(a0)
    psi_max = psi_ele + 2.0 * np.pi * tfreq *\
                1.0e3 / C * (X * sp0 + Y * np.sqrt(ca0*ca0 - sp0*sp0) + Z * sa0)
    
    # compute the number of 2pi factors necessary to map to correct region
    dpsi = (psi_max - psi_obs)
    n2pi = np.floor(dpsi / (2.0 * np.pi)) if Y > 0 else np.ceil(dpsi / (2.0 * np.pi))
    
    # map observed phase to correct extended phase
    d2pi = n2pi * 2.0 * np.pi
    psi_obs += d2pi
    
    # now solve for the elevation angle: alpha
    E = (psi_obs / (2.0*np.pi*tfreq*1.0e3) + tdiff*1e-6) * C - X * sp0
    alpha = np.arcsin((E*Z + np.sqrt(E*E * Z*Z - (Y*Y + Z*Z)*(E*E - Y*Y*cp0*cp0))) / (Y*Y + Z*Z))
    return (180.0 * alpha / np.pi)

