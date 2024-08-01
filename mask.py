import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D

def occulterLocation(name_input):
    """Return the x and y locations for the default occulting positions.
    Input:
        name_input -- name of the occulter, allowed inputs: ['BAR5', 'BAR10', 
                    'WEDGEA0.6', 'WEDGEA1.0', 'WEDGEA1.8', 'WEDGEA2.0', 'WEDGEA2.5', 'WEDGEA2.8', 
                    'WEDGEB1.0', 'WEDGEB1.8', 'WEDGEB2.0', 'WEDGEB2.5', 'WEDGEB2.8']
    Output:
        x, y -- values of the locations
    """
    #The array below contains emperical postions from the public
    # STIS coronagraphic archive, measured by Bin Ren from ADS: 2017SPIE10400E..21R
    #
    #The only difference is that the values are the median of the centers
    #that are determined by fitting two lines to the diffraction spikes, rather
    #than using Radon transfrom in 2017SPIE10400E..21R
    array_name_values = np.array([
                        ['WEDGEA2.0', 311.48, 613.74],
                        ['WEDGEA1.8', 309.43, 534.17],
                        ['BAR10',     624.73, 844.17],
                        ['WEDGEB2.5', 802.73, 302.46],
                        ['WEDGEA1.0', 309.65, 213.33],
                        ['WEDGEA0.6', 307.98,  67.59],
                        ['WEDGEA2.8', 309.51, 933.82],
                        ['WEDGEB1.8', 528.05, 303.68],
                        ['WEDGEA2.5', 309.03, 813.59],
                        ['BAR5',      969.73, 697.81],
                        ['WEDGEB1.0', 214.37, 305.08],
                        ['WEDGEB2.8', 917.71, 303.47],
                        ['WEDGEB2.0', 606.84, 303.63]
                        ]) 

    names_occulters = array_name_values[:, 0]
    name = name_input.upper()
    if name not in names_occulters:
        raise ValueError(name_input + " is not a supported location by STIS.")
    else:
        return float(array_name_values[np.where(names_occulters == name), 1][0][0]), float(array_name_values[np.where(names_occulters == name), 2][0][0])
    
def create_mask(width_y, width_x, occulter, hw_spikes = 1, cen_x = None, cen_y = None):
    """Make all 1's array except the diagonals where are 0's.
    Input:
        width_y: width in the y-direction. Default is None, i.e., width_y = width;
        width_x: width in the x-direction. Default is None, i.e., width_x = width;
        occulter: occulter position name, e.g. 'WEDGEA1.0';
        hw_spikes: half width of the diagonals;
        cen_x: center of the spikes in the x-direction. Default is None, i.e., cen_x = (width_x-1)/2.0;
        cen_y: center of the spikes in the y-direction. Default is None, i.e., cen_y = (width_y-1)/2.0.
    Output:
        diagonals: 2D array with properties described above.
    """   
    mask = fits.getdata('mask_STIS_coron.fits')
    f = Cutout2D(mask, position=occulterLocation(occulter), size=(width_y,width_x), wcs=None)
    spikes = np.ones((width_y, width_x))
    
    if hw_spikes == 0:
        return spikes
    
    if cen_x is None:
        cen_x = (width_x - 1)/2.0
    if cen_y is None:
        cen_y = (width_y - 1)/2.0

    for y in range(width_y):
        for x in range(width_x):
            if np.abs(np.abs(x-cen_x)-np.abs(y-cen_y)) <= hw_spikes:
                spikes[y, x] = 0
    return spikes*f.data