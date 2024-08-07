import os
import math
import warnings
import radonCenter
import numpy as np
import pandas as pd
import scipy.ndimage
from multiprocess import Pool
from mask import create_mask
from badrefs import badrefs
from astropy import wcs
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.exceptions import AstropyWarning

trgdir = 'testfiles/' #Directory where the files corresponding to the target are located
refdir = 'testfiles/' #Directory where the files corresponding to the reference are located
trgname = 'LKCA-15' #Name of target, as noted in the 'TARGNAME' header of the fits file. 
trgfiles = [] 

for i in os.listdir(trgdir):
    path = trgdir+i
    hdul = fits.open(path)
    tname = hdul[0].header['TARGNAME']
    if tname == trgname:
        trgfiles.append(i) #By default, this selects every file in the given target directory that corresponds to the given target name. This can easily be changed to a list of files if one wants to look at specific epochs etc.

reffiles = [f for f in os.listdir(refdir) if f not in trgfiles] #Selects every file in the reference directory making sure to not accidentally select target files. Can also be manually configured as a list.
reffiles = [f for f in reffiles if f not in badrefs] #Filters out specified bad reference files.

ncores = 94 #Set number of cpu cores to be used.

#Processes target and reference frames. It is written to be parallelizable. Follows 
#"Post-processing of the HST STIS coronagraphic observations" by Ren et al. closely. Changes not advised. 
def prep_frames_parallel(filename):
    
    try:
        trgs = []
        refs = []
        PAs = []
        fnames = []
        
        image_data_sc = []
        image_data_dq = []
        
        if filename in trgfiles:
            path = os.path.join(trgdir, filename)
        elif filename in reffiles:
            path = os.path.join(refdir, filename)
            
        with fits.open(path) as hdul:
            exptimes = []
            CRPIX1 = hdul['SCI'].header['CRPIX1']
            CRPIX2 = hdul['SCI'].header['CRPIX2']
            NAXIS1 = hdul['SCI'].header['NAXIS1']
            NAXIS2 = hdul['SCI'].header['NAXIS2']
            CCDGAIN = hdul[0].header['CCDGAIN']
            wcsO = wcs.WCS(hdul[1].header)
            rot_angle = np.rad2deg(math.atan2(wcsO.wcs.cd[1][0], wcsO.wcs.cd[0][0]))
            wcspa = 180*np.sign(rot_angle) - rot_angle
            
            for i in range(1,len(hdul),3):
                exptimes.append(hdul[i].header['EXPTIME'])
                
            exptimes = np.array(exptimes)
            medexp = np.median(exptimes)
            
            for i in range(1,len(hdul),3):
                EXPTIME = hdul[i].header['EXPTIME']
                if EXPTIME > medexp:
                    continue
                image_data_sc.append(hdul[i].data)
                image_data_dq.append(hdul[i+2].data)
                
            CRPIX1a = np.zeros(len(image_data_dq))
            CRPIX2a = np.zeros(len(image_data_dq))
            
        for x in range(len(image_data_dq)): 
            
            cutframesc = Cutout2D(image_data_sc[x],(CRPIX1,CRPIX2),(500,500),wcs=None)
            cutframedq = Cutout2D(image_data_dq[x],(CRPIX1,CRPIX2),(500,500),wcs=None)
            image_data_sc[x] = cutframesc.data
            image_data_dq[x] = cutframedq.data
            CRPIX1a[x],CRPIX2a[x] = cutframesc.position_cutout
            
            dq01 = np.where(image_data_dq[x]==(16 or 256 or 8192),0,1)
            dq10 = np.where(image_data_dq[x]==(16 or 256 or 8192),1,0)
            medsci = scipy.ndimage.median_filter(image_data_sc[x],size=3)
            multsciadd = np.multiply(dq10,medsci)
            multscisub = np.multiply(dq01,image_data_sc[x])
            image_data_sc[x] = np.add(multscisub,multsciadd)
                
            for i in range(len(image_data_dq[x])):
                for j in range(len(image_data_dq[x][i])):
                    vdist  = i - CRPIX2a[x]
                    hdist  = j - CRPIX1a[x]
                    radius = math.sqrt(pow(vdist, 2) + pow(hdist, 2))
                    image_data_sc[x][i][j] = image_data_sc[x][i][j]*pow(radius, 0.5)
                
            (x_cen, y_cen) = radonCenter.searchCenter(image_data_sc[x], CRPIX1a[x], CRPIX2a[x], size_window = math.floor(NAXIS2/2),size_cost=7,theta=[45, 135]) 
        
            image_data_sc[x] = image_data_sc[x]/EXPTIME
            
            voff = image_data_sc[x].shape[0]/2 - y_cen 
            hoff = image_data_sc[x].shape[1]/2 - x_cen 
            
            shiftimage_data_sc = scipy.ndimage.shift(image_data_sc[x], np.array([voff, hoff]))  
            
            lh = 201 #for A1.0
            lw = 201 #for A1.0
        
            sc_data_2d = Cutout2D(shiftimage_data_sc, position=(shiftimage_data_sc.shape[1]/2,shiftimage_data_sc.shape[0]/2), size=(lh,lw), wcs=None)
        
            if filename in trgfiles:
                trgs.append(sc_data_2d.data)
                PAs.append(wcspa)
                fnames.append(filename)
            elif filename in reffiles:
                refs.append(sc_data_2d.data)
                PAs.append(wcspa)
                fnames.append(filename)
                
    except Exception as e:
        print(e, filename)
        
    return trgs, refs, PAs, fnames
 

warnings.simplefilter('ignore', AstropyWarning)
with Pool(ncores) as p:
    results1 = list(p.map(prep_frames_parallel, trgfiles))
    results2 = list(p.map(prep_frames_parallel, reffiles))

trgs = []
refs = []
PAs = []
fnames = []
fnames2 = []

for i in range(len(results1)):
    for j in range(len(results1[i][0])):
        tr = results1[i][0][j]
        trgs.append(tr)
        
for i in range(len(results1)):
    for j in range(len(results1[i][2])):
        tr = results1[i][4][j]
        PAs.append(tr)
        
for i in range(len(results1)):
    for j in range(len(results1[i][3])):
        tr = results1[i][5][j]
        fnames.append(tr)

for i in range(len(results2)):
    for j in range(len(results2[i][3])):
        tr = results2[i][5][j]
        fnames2.append(tr)
        
for i in range(len(results2)):
    for j in range(len(results2[i][1])):
        tr = results2[i][2][j]
        refs.append(tr)

trgs = np.array(trgs)
refs = np.array(refs)
fnames = np.array(fnames)
fnames2 = np.array(fnames2)
PAs = np.array(PAs)
mask = create_mask(refs.shape[1], refs.shape[2], hw_spikes = 5, occulter='WEDGEA1.0')

savefiles = ['testfiles/'+trgname+'refs.npy',  
             'testfiles/'+trgname+'trgs.npy', 
             'testfiles/'+trgname+'fnames.npy',
             'testfiles/'+trgname+'pas.npy',  
             'testfiles/'+trgname+'mask.npy']

with open(savefiles[0], 'wb') as f:
	np.save(f,refs)

with open(savefiles[2], 'wb') as f:
    np.save(f,trgs)

with open(savefiles[4], 'wb') as f:
    np.save(f,fnames)

with open(savefiles[5], 'wb') as f:
    np.save(f,PAs)

with open(savefiles[6], 'wb') as f:
    np.save(f,mask)

print("Save done")
