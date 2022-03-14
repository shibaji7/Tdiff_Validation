""" calc_vheight.py
   ==============
   Author: S. Chakraborty
   This file is a v_Height estimator
"""

def calculate_vHeight(r, elv, hop, Re = 6371.0):
    """
    Parameters
    ----------
    r: slant range in km
    elv: elevation angle in degree
    hop: back scatter hop (0.5,1,1.5, etc.)
    Re: Earth radius in km
    """
   
    h = np.sqrt(Re**2 + (r/(2*N))**2 + (r/N)*Re*np.sin(np.radians(elv))) - Re
      
    return h
