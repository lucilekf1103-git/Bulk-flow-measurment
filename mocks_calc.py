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

n_mocks = 675  # number of mocks : in total 675 for DESI sample
n_moment = 5  # Qxx, Qxy, Qxz, Qyy, Qyz (we don't store Qzz because it's not independent, Qzz = -Qxx - Qyy)

Q_estimated_tot = np.zeros((n_mocks, n_moment))  
Q_true_tot = np.zeros((n_mocks, n_moment))

estimator = 'etaMLE' # can also be 'wMLE' or 'tMLE' depending on which estimator we want to use

# cosmological parameters:
OmegaM= 0.3151  
OmegaA= 1.0-OmegaM  
Hub   = 100.

data_path  = '/renoir/fromenti/Documents/data_DESI/combinedpv/Y1/mocks/' # you can change this path to your data path 

def all_mocks_calculator(mock_number, Fit_tech):  
    
    CFtype  ='BQ' # if you want just the bulk flow estimation, you can change this to 'BF'
    QzzType ='fix'
    NmcSamp = [500,2000]

    # we have 25 simulation boxes, each box has been divided into 27 sub-cubes to create the mocks. Therefore we have 25*27 = 675 mocks in total.
    Nsub = 27
    ibox = mock_number// Nsub  # index of the box (0 to 24)
    isub = mock_number % Nsub # index of the sub-box (0 to 26)
    
    
    # 1. the output of optimization and MCMC
    Q_opt = np.zeros(3)
    Qt = np.zeros(3)
    
    # 2. read in the mock and extract the moments
    
    file_name = data_path + 'Combined_AbacusSummit_clustering_c000_ph' + \
        '{:0>3}'.format(ibox) + '_r' + '{:0>3}'.format(isub) + '.fits'  # the name of the file containing the mock data
        
    with fits.open(file_name) as hfile:  # we extract the data from the fits file if it is a .fits
        infile=hfile[1].data
        
    hfile.close()
    rav      = infile['RA'] 
    decv     = infile['DEC']
    zv       = infile['Z'] 
    logdt    = infile['LOGDIST_TRUE']
    logd     = infile['LOGDIST'] 
    elogd    = infile['LOGDIST_ERR']  
    
    
    # Conversion into galactic coordinates(l, b)
    c = SkyCoord(ra=rav, dec=decv, unit=(u.degree, u.degree), frame='icrs')
    galactic = c.galactic
    l = galactic.l.value    # galactic longitude in degrees
    b = galactic.b.value    # galactic latitude in degrees


    # 3. calculate cosmicflow of one mock:
    if Fit_tech == 'etaMLE':
        Bt,Qt, tmp = Opt_etaMLE(l, b, zv, logdt, elogd, False, OmegaM, OmegaA, Hub, CFtype, QzzType, False)
        
    if Fit_tech == 'wMLE':
        Vpec  = Vpec_Fun_wat(zv, logdt, OmegaM, OmegaA, Hub)
        eVpec = Vpec_Fun_wat(zv, elogd, OmegaM, OmegaA, Hub)
        Bt, Qt, tmp = Opt_wMLE(l, b, Vpec, eVpec, zv, Hub, CFtype, QzzType)
        
        
    # 4 . calculate the estimated moments for one mock:    
    if Fit_tech == 'etaMLE':
        BF_opt, Q_opt, tmp = Opt_etaMLE(l, b, zv, logd, elogd, False, OmegaM, OmegaA, Hub, CFtype, QzzType, False)
        
    if Fit_tech == 'wMLE':
        Vpec  = Vpec_Fun_wat(zv, logd, OmegaM, OmegaA, Hub)
        eVpec = Vpec_Fun_wat(zv, elogd, OmegaM, OmegaA, Hub)
        BF_opt, Q_opt, tmp = Opt_wMLE(l, b, Vpec, eVpec, zv, Hub, CFtype, QzzType)
        
    if Fit_tech == 'tMLE':
        Vpec  = Vpec_Fun_tra(zv, logd, OmegaM, OmegaA, Hub)
        eVpec = Vpec_Fun_wat(zv, elogd, OmegaM, OmegaA, Hub)
        BF_opt, Q_opt, tmp = Opt_tMLE(l, b, Vpec, eVpec, zv, Hub, CFtype, QzzType)


    return Q_opt, Qt, Bt , BF_opt



Output_dir = '/renoir/fromenti/Documents/codes_Bulk_flow/results_V5_data/' # you can change this path to your output path
os.makedirs(Output_dir, exist_ok=True)

with open(Output_dir + 'all_moments_all_mocks_l_b_' + estimator + '.txt', 'w') as outfile:
    
    outfile.write("# Mock    Qxx       Qxy       Qxz       Qyy       Qyz       Qzz     Qxx_true    Qxy_true     Qxz_true     Qyy_true     Qyz_true     Qzz_true     Bx_opt     By_opt     Bz_opt     Bx_true    By_true    Bz_true\n")

    for mock_number in range(n_mocks):
        Q_estimated, Q_true, B_opt, B_true = all_mocks_calculator(mock_number, Fit_tech=estimator)
        
        # Store the values in the total arrays
        Q_estimated_tot[mock_number] = Q_estimated
        Q_true_tot[mock_number] = Q_true
        
        # Calculation of Qzz for the estimated and true values : 
        # Q_estimated = [Qxx, Qxy, Qxz, Qyy, Qyz]
        Qxx_est = Q_estimated[0]
        Qyy_est = Q_estimated[3]
        Qzz_est = -Qxx_est - Qyy_est  # Qzz estimated
        
        # Q_true = [Qxx_true, Qxy_true, Qxz_true, Qyy_true, Qyz_true]
        Qxx_true_val = Q_true[0]
        Qyy_true_val = Q_true[3]
        Qzz_true_val = -Qxx_true_val - Qyy_true_val  # Qzz true


        outfile.write(
        f"{mock_number:4d}   {Q_estimated[0]:8.3f}   {Q_estimated[1]:8.3f}   {Q_estimated[2]:8.3f}  "
        f"{Q_estimated[3]:8.3f}   {Q_estimated[4]:8.3f}   {Qzz_est:8.3f} "
        f"{Q_true[0]:8.3f}   {Q_true[1]:8.3f}    {Q_true[2]:8.3f}    {Q_true[3]:8.3f}    {Q_true[4]:8.3f}    {Qzz_true_val:8.3f} "
        f"{B_opt[0]:8.3f}    {B_opt[1]:8.3f}   {B_opt[2]:8.3f} "
        f"{B_true[0]:8.3f}     {B_true[1]:8.3f}     {B_true[2]:8.3f}\n")