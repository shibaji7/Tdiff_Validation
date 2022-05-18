""" calc_vheight.py
   ==============
   Author: S. Chakraborty
   This file is a v_Height estimator
"""
import numpy as np

def calculate_vHeight(r, elv, hop, Re = 6371.0):
    """
    Parameters
    ----------
    r: slant range in km
    elv: elevation angle in degree
    hop: back scatter hop (0.5,1,1.5, etc.)
    Re: Earth radius in km
    """
   
    h = np.sqrt(Re**2 + (r/(2*hop))**2 + (r/hop)*Re*np.sin(np.radians(elv))) - Re
      
    return h

def chisham(target_range: float, **kwargs):
    """
    Mapping ionospheric backscatter measured by the SuperDARN HF
    radars – Part 1: A new empirical virtual height model by
    G. Chisham 2008 (https://doi.org/10.5194/angeo-26-823-2008)
    Parameters
    ----------
    target_range: float
        is the range from radar to the target (echos)
        sometimes known as slant range [km]
    kwargs: is only needed to avoid key item errors
    Returns
    -------
    altered target_range (slant range) [km]
    """
    # Model constants
    A_const = (108.974, 384.416, 1098.28)
    B_const = (0.0191271, -0.178640, -0.354557)
    C_const = (6.68283e-5, 1.81405e-4, 9.39961e-5)

    # determine which region of ionosphere the gate
    if target_range < 115:
        return (target_range / 115.0) * 112.0
    elif target_range < 787.5:
        return A_const[0] + B_const[0] * target_range + C_const[0] *\
                 target_range**2
    elif target_range <= 2137.5:
        return A_const[1] + B_const[1] * target_range + C_const[1] *\
                 target_range**2
    else:
        return A_const[2] + B_const[2] * target_range + C_const[2] *\
                 target_range**2

def thomas(target_range: float, sctr_type: int):
    """
    Mapping ionospheric backscatter measured by the SuperDARN HF
    radars – Part 1: A new empirical virtual height model by
    E. Thomas 2022 
    Parameters
    ----------
    target_range: float
        is the range from radar to the target (echos)
        sometimes known as slant range [km]
    Returns
    -------
    altered target_range (slant range) [km]
    """
    u = np.array([1, target_range, target_range**2])
    # Model constants
    # GS
    oneE = np.array([111.393, -1.65773e-4, 4.26675e-5])
    oneF = np.array([378.022, -0.14738, 6.99712e-5])
    twoF = np.array([-76.2406, 0.06854, 1.23078e-5])
    # IS
    hafE = np.array([108.873, -0.01444, 1.57806e-4])
    hafF = np.array([341.005, -0.17484, 1.99144e-4])
    ohF = np.array([92.9665, 0.03967, 1.59501e-5])

    h = np.nan
    # determine which region of ionosphere the gate
    if sctr_type == 1:
        if (target_range >= 560) and (target_range < 1140): h = np.sum(oneE * u)
        elif (target_range >= 1140) and (target_range < 3265): h = np.sum(oneF * u)
        elif (target_range >= 3265): h = np.sum(twoF * u)
        else: h = np.nan
    else:
        if (target_range < 675): h = np.sum(hafE * u)
        elif (target_range >= 675) and (target_range < 2275): h = np.sum(hafF * u)
        else: h = np.sum(ohF * u)
    return h

def standard_virtual_height(target_range: float, cell_height: int = 300,
                            **kwargs):
    """
    cell_height, target_range and x_height are in km
    Default values set in virtual height model described
    Mapping ionospheric backscatter measured by the SuperDARN HF
    radars – Part 1: A new empirical virtual height model by
    G. Chisham 2008
    Equation (1) in the paper
    < 150 km climbing into the E region
    150 - 600 km E region scatter
    (Note in the paper 400 km is the edge of the E region)
    600 - 800 km is F region
    Parameters
    ----------
    target_range: float
        is the range from radar to the target (echos)
        sometimes known as slant range [km]
    cell_height: int
        the default height of the echo if the target_range
        is within a certain range
    kwargs: is only needed to avoid key item errors
    Returns
    -------
    altered target_range (slant range) [km]
    """
    # TODO: why 115?
    # map everything into the E region
    if cell_height <= 150 and target_range > 150:
        return cell_height
    # virtual height equation (1) from the above paper
    elif target_range < 150:
        return (target_range / 150.0) * 115
    elif target_range >= 150 and target_range <= 600:
        return 115
    elif target_range > 600 and target_range < 800:
        return (target_range - 600) / 200 * (cell_height - 115) + 115
    # higher than 800 km
    else:
        return cell_height