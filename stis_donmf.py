import os
import math
import warnings
import radonCenteru
import numpy as np
import configparser
import pandas as pd
import scipy.ndimage
from multiprocess import Pool
from mask import create_mask
from badrefs import badrefs
from astropy import wcs
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.exceptions import AstropyWarning

config = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')] if len(x) > 0 else []})
config.read('config.ini')

n_components = config.getint('nmffacs', 'n_components')
maxiters = config.getint('nmffacs', 'maxiters')

