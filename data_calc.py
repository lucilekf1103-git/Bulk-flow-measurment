# usefull libraries:

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lapack
from CosmoFunc import *
from BF_OptMC  import *
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import os
import time

start = time.time() # for mesuring the time of execution of the code

# path of the data file :
Output_dir  = '/renoir/fromenti/Documents/codes_Bulk_flow/results_V5_data/'   # change here if you whant to change the path of output
data_path_mardec   = '/datadec/desi/pv/combinedpv/Y1/PV_clustering_data_v5_v13.fits' # path of the data file on mardec

estimator = 'wMLE'  # Fit_tech can be 'etaMLE', 'wMLE' or 'tMLE'

# cosmological parameters:
OmegaM= 0.3151  
OmegaA= 1.0-OmegaM  
Hub   = 100.

def moment_data_calculator(data, Fit_tech):


    CFtype  ='BQ' # 'BQ' is to calculate all the moments and 'BF' is to calculate only the bulk flow
    QzzType ='fix'
    NmcSamp = [500,2000]  
    
    
    # if the file end with .fits
    if data.endswith('.fits'):
        # Format FITS
        with fits.open(data) as hfile:
            infile = hfile[1].data
        rav = infile['RA']
        decv = infile['DEC']
        zv = infile['Z']
        logd = infile['LOGDIST']
        elogd = infile['LOGDIST_ERR']
        fp_flag = infile['FP_FLAG'] 
        
        # if we want to select only the Tully-Fisher sample or just the Fundamental Plane sample for the DESI data
        mask_selected = (fp_flag == 0) # id =0 just the Tully_fisher sample and id =1 is the Fundamental Plane sample for the DESI data
        # if we want all the data set :
        mask_selected = (fp_flag == 0) | (fp_flag == 1) # to select all the data set 

    
        # Apply mask to all arrays
        rav = rav[mask_selected]
        decv = decv[mask_selected]
        zv = zv[mask_selected]
        logd = logd[mask_selected]
        elogd = elogd[mask_selected]
        N = len(rav)  # nombre de galaxies après sélection
        
    else:
        # if the file is a txt file :
        file_data = np.loadtxt(data, skiprows=1)
        rav = file_data[:, 1]      # RA
        decv = file_data[:, 2]     # DEC
        zv = file_data[:, 5] / 299792.458  # Z = cz / c
        logd = file_data[:, 6]     # LOGDIST
        elogd = file_data[:, 7]    # LOGDIST_ERR
        N = len(rav)  

    # conversion in galactic coordinates (l, b) the initial coordinates are in equatorial coordinates (RA, DEC)
    c = SkyCoord(ra=rav, dec=decv, unit=(u.degree, u.degree), frame='icrs')
    galactic = c.galactic
    l = galactic.l.value    # galactic longitude in degrees
    b = galactic.b.value    # galactic latitude in degrees

    
    if (Fit_tech == 'etaMLE'):
        
        (B_data,Q_data, tmp, *rest) = Opt_etaMLE(l, b, zv, logd, elogd, False, OmegaM, OmegaA, Hub, CFtype, QzzType, False)
        (B_MC,Q_MC,Sv_MC,Um_Cov,Q_cov,Tot_cov,eQzz,SX,SY,SZ,*rest) = MC_etaMLE(l, b, zv, logd, elogd, False, OmegaM, OmegaA, Hub, NmcSamp, CFtype, QzzType, False)
        
    if (Fit_tech == 'wMLE'):
        Vpec  = Vpec_Fun_wat(zv, logd, OmegaM, OmegaA, Hub)
        eVpec = Vpec_Fun_wat(zv, elogd, OmegaM, OmegaA, Hub)
        B_data,Q_data, tmp = Opt_wMLE(l, b, Vpec, eVpec, zv, Hub, CFtype, QzzType)
        (B_MC,Q_MC,Sv_MC,Um_Cov,Q_cov,Tot_cov,eQzz,SX,SY,SZ,*rest) = MC_wMLE(l, b, Vpec, eVpec, zv, Hub, NmcSamp, CFtype, QzzType)
        
    if (Fit_tech == 'tMLE'):
        Vpec  = Vpec_Fun_tra(zv, logd, OmegaM, OmegaA, Hub)
        eVpec = Vpec_Fun_wat(zv, elogd, OmegaM, OmegaA, Hub)
        B_data,Q_data, tmp = Opt_tMLE(l, b, Vpec, eVpec, zv, Hub, CFtype, QzzType)
        (B_MC,Q_MC,Sv_MC,Um_Cov,Q_cov,Tot_cov,eQzz,SX,SY,SZ,*rest) = MC_tMLE(l, b, Vpec, eVpec, zv, Hub, NmcSamp, CFtype, QzzType)
        
        
    return { "B_data": B_data,
    "Q_data": Q_data,
    "Q_MC": Q_MC,
    "Q_cov": Q_cov,
    "Tot_cov": Tot_cov,
    "eQzz": eQzz,
    "SigV": Sv_MC,
    "N": N}
    
    
result = moment_data_calculator(data_path_mardec, Fit_tech = estimator)

B_data = result["B_data"] # calcuated bulk flow with the estimator
Q_data = result["Q_data"] # calcuated shear moment with the estimator 
Q_MC   = result["Q_MC"]  # shear moment calculated for plot, errors bars and contours
Q_cov  = result["Q_cov"] # total covariance matrix of all the moments
Q_tot  = result["Tot_cov"]  # total covariance matrix of the shear moment
eQzz   = result["eQzz"] # error of the Qzz element of the shear moment tensor
SigV   = result["SigV"] # velocity dispersion
N      = result["N"]   

end = time.time() # for mesuring the time of execution of the code

# We calculate the error of the bulk flow estimation for the data using the covariance matrix

Cov_Q_all = Q_cov
err_Bx = np.sqrt(Q_tot[0,0])
err_By = np.sqrt(Q_tot[1,1])
err_Bz = np.sqrt(Q_tot[2,2])
errQ = np.zeros(5)
for i in range(5):
    errQ[i] = np.sqrt(Cov_Q_all[i, i])

# the definition of the Qzz element of this tensor is just Qzz = -Qxx - Qyy because the shear moment tensor is traceless:
Qzz =  - Q_data[0] - Q_data[3]
err_Qzz = np.sqrt(errQ[0]**2  + errQ[3]**2)

# calculating the B amplitude :
B_amp = np.sqrt(B_data[0]**2 + B_data[1]**2 + B_data[2]**2)
err_B_ampl = (np.abs(B_data[0]*err_Bx) + np.abs(B_data[1]*err_By) + np.abs(B_data[2]*err_Bz)) / B_amp

# Saving our results in a txt file:
os.makedirs(Output_dir, exist_ok=True)
with open(Output_dir + 'all_moments_data_l_b' + estimator + '.txt', 'w') as outfile:
    
    outfile.write(f"Time taken to calculate = {end - start:.4f} seconds\n\n")
    outfile.write("Estimator : " + estimator + "\n\n")
    
    outfile.write("==== Bulk Flow Results ====\n\n")
    outfile.write("Component    Value        Error\n")
    outfile.write("---------------------------------\n")
    outfile.write(f" |B|       {B_amp}       {err_B_ampl}\n")
    outfile.write(f" Bx        {B_data[0]}     {err_Bx}\n")
    outfile.write(f" By        {B_data[1]}     {err_By}\n")
    outfile.write(f" Bz        {B_data[2]}     {err_Bz}\n\n")
    
    outfile.write("==== Shear moment Results ====\n\n")

    outfile.write("Component    Value        Error\n")
    outfile.write("---------------------------------\n")

    outfile.write(f"Qxx       {Q_data[0]:.4f}     {errQ[0]:.4f}\n")
    outfile.write(f"Qxy       {Q_data[1]:.4f}     {errQ[1]:.4f}\n")
    outfile.write(f"Qxz       {Q_data[2]:.4f}     {errQ[2]:.4f}\n")    
    outfile.write(f"Qyy       {Q_data[3]:.4f}     {errQ[3]:.4f}\n")
    outfile.write(f"Qyz       {Q_data[4]:.4f}     {errQ[4]:.4f}\n")
    outfile.write(f"Qzz       {Qzz:.4f}       {err_Qzz:.4f}\n\n")
    
    outfile.write(f"SigV (Velocity Dispersion) = {SigV:.4f} km/s\n\n")
    
    outfile.write("Covariance Matrix Estimation:\n\n")
    for row in Q_tot:
        outfile.write(" ".join(f"{val:.6f}" for val in row) + "\n")
    outfile.write(f" with {N} galaxies in the data file\n")
        

print("File saved at:", Output_dir + 'all_moments_data_l_b' + estimator + '.txt')