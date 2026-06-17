import logging
logging.disable(logging.CRITICAL) 
from generator_flip import *
from CosmoFunc import *
import numpy as np
import camb
from astropy.io import fits
from camb import model, initialpower
import time
import math
import matplotlib.pyplot as plt
from scipy import special
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.integrate import trapezoid
from numba import jit, prange


################################################################################################
# 1. paramètres cosmo
################################################################################################

OmegaM= 0.3151
OmegaLambda = 1.0-OmegaM
Omegarad = 10**(-5)
c = 299792.458 # speed of light in km/s
wo = -1 # for lambdaCDM
wa = 0 # for lambdaCDM
H0 = 100. 
littleh=0.6727


##################################################################################################
# 2. power spectrum avec CAMB
##################################################################################################


pars = camb.CAMBparams()
pars.set_cosmology(100*littleh, ombh2=0.022, omch2=0.1198)
pars.set_dark_energy()
pars.InitPower.set_params(As=2.114940245149156e-09, ns=0.9645)
pars.set_matter_power(redshifts=[0.], kmax=10)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)

khs, z, pks = results.get_matter_power_spectrum(minkh=1e-4, maxkh=0.4, npoints=300)

k_linear = khs
Pk_linear = pks[0, :]
power_spectrum_list = [[k_linear, Pk_linear]]  # ← Liste de listes


####################################################################################################
# 3. Données DESI
#####################################################################################################

data_fix = '/renoir/fromenti/Documents/data_DESI/combinedpv/Y1/PV_clustering_data_v5_v13.fits'
data_mardec_desi = '/datadec/desi/pv/combinedpv/Y1/PV_clustering_data_v5_v13.fits'  # you can change here
data = data_fix

def load_data(data, mode, N):
    
    if mode == "random":

        with fits.open(data) as hfile:
            infile = hfile[1].data
            n_galaxies = len(infile)
            tab = np.array(infile)
            indices = np.random.choice(len(tab), size=min(N, len(tab)), replace=False)
            elements = tab[indices]
    
            ra = elements['RA']
            dec = elements['DEC']
            rsf = elements['Z']
            rsf_err = elements['ZCMB']
            logdist = elements['LOGDIST']
            err_logdist = elements['LOGDIST_ERR']
            N = len(rsf)
            coordinates = np.array([ra, dec, logdist])
        
    elif mode == "all":
        with fits.open(data) as hfile:
            infile = hfile[1].data
            tab = np.array(infile)
            ra = tab['RA']
            dec = tab['DEC']
            rsf = tab['Z']
            rsf_err = tab['ZCMB']
            logdist = tab['LOGDIST']
            err_logdist = tab['LOGDIST_ERR']
            N = len(rsf)
            coordinates = np.array([ra, dec, logdist])
            
    return ra, dec, rsf, rsf_err, logdist, err_logdist, N, coordinates

mode = "random"  # ou "all" , "random"
Ng = 3000  # nombre de galaxies si mode random

ra, dec, rsf, rsf_err, logdist, err_logdist, N, coordinates = load_data(data, mode, Ng)
    
    
coordinates_eq = np.array([ra, dec, logdist])


#################################################################################################
# 4. Calcul de la covariance de flip C_ij
#################################################################################################


start = time.time()
batches = compute_coordinates(
    "vv",
    coordinates_density=None,
    coordinates_velocity=coordinates_eq,
    size_batch=10_000,
    los_definition="bisector",)

correlation = compute_coeficient(
    batches,  
    "carreres23",
    "vv",
    power_spectrum_list,
    number_worker=16,
    hankel=True,)

end  = time.time()


print("Correlation matrix: ok ")
print(np.shape(correlation))
print(f"Time taken for {N} galaxies : ", end - start, " seconds")


print("\n=== RECONSTRUCTION DE LA MATRICE C_ij ===")

# Extraire le vecteur des covariances (premier terme du modèle)
cov_vector = correlation[0]  # shape (n_pairs,)

# Nombre de paires attendu pour N galaxies (auto-corrélation)
# Pour FLIP: n_pairs = N*(N+1)//2 - N + 1 = N*(N-1)//2 + 1
n_pairs_expected = N * (N - 1) // 2 + 1
print(f"N = {N}, cov_vector length = {len(cov_vector)}, attendu = {n_pairs_expected}")

if len(cov_vector) != n_pairs_expected:
    print(f"⚠ Attention: taille inattendue, ajustement...")
    n_pairs_actual = len(cov_vector)
else:
    n_pairs_actual = n_pairs_expected

# Créer la matrice C_ij (N×N)
C_matrix = np.zeros((N, N))

# La première valeur est la variance théorique (identique pour toutes les galaxies)
variance_theorique = cov_vector[0]

# Mettre la variance sur toute la diagonale
for i in range(N):
    C_matrix[i, i] = variance_theorique

# Remplir les termes hors-diagonale (covariances entre galaxies différentes)
# Les valeurs cov_vector[1:] correspondent aux paires (i,j) avec i < j
idx = 1
for i in range(N):
    for j in range(i+1, N):
        if idx < len(cov_vector):
            cov_value = cov_vector[idx]
            C_matrix[i, j] = cov_value
            C_matrix[j, i] = cov_value
            idx += 1

print(f"Matrice C_ij reconstruite: shape {C_matrix.shape}")
        
        

#######################################################################################
# 5.  Calcul des weights 
#######################################################################################

# Now we want to extract the comoving distances of the galaxie assuming a model from the redshift:
d_comoving = []
for i in range(N):
    d_comoving.append(DistDc(rsf[i], OmegaM, OmegaLambda, Omegarad, H0, wo, wa, ap=1)) # in Mpc ap does not matter for the distance

d_rsf = []
for i in range(N):
    d_rsf.append(DistDc(rsf[i], OmegaM, OmegaLambda, Omegarad, H0, wo, wa, ap=1)) # in Mpc ap does not matter for the distance
d_comoving = d_rsf*10.**(-logdist)   

# Convertissez en coordonnées galactiques :
sky = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='icrs')
galactic = sky.galactic
l = galactic.l.value    # longitude galactique en degrés
b = galactic.b.value    # latitude galactique en degrés

# Puis utilisez l et b pour les coordonnées cartésiennes :
x_coord = np.cos(b/180*np.pi) * np.cos(l/180*np.pi)
y_coord = np.cos(b/180*np.pi) * np.sin(l/180*np.pi)
z_coord = np.sin(b/180*np.pi)

# We stack the x,y,z in a single array:
r_hat = np.vstack((x_coord, y_coord, z_coord)).T
d_comoving = np.array(d_comoving) # the comoving distance in Mpc must be an array to be able to convert it in a [N,1] array to multiply it with r_hat [N,3]
r = r_hat * d_comoving[:, None]


alpha_n = np.log(10) * c * rsf * err_logdist / (1+rsf) # error on the peculiar velocity in km/s
alpha_star = 300 # in km/s is a estimation of the typical value of the error of the peculiar velocity of the galaxies

# Lets calculate the weights with the formula from the paper:

# calculating the g function for each galaxy:

g = np.column_stack([x_coord,                       # x
                     y_coord,                       # y
                     z_coord,                       # z 
                     d_comoving*x_coord**2,         # xx
                     2*d_comoving * x_coord*y_coord,# xy
                     2*d_comoving * x_coord*z_coord,# xz
                     d_comoving * y_coord**2,       # yy
                     2*d_comoving * y_coord*z_coord,# yz
                     d_comoving * z_coord**2])      # zz

# calculating the A matrix:
alpha_tot2 = alpha_n**2 + alpha_star**2  # (N,)
A = g.T @ (g / alpha_tot2[:, None])

# calculating the weights:
A_inv = np.linalg.inv(A)
w = 1 / (alpha_n**2 + alpha_star**2)
weights_calc = A_inv @ (g * w[:, None]).T
print ("Weights calculated: ok : with  shape : ", np.shape(weights_calc))



######################################################################################
# 6. Calcul de la matrice Rpq directement à partir de la covariance de flip et des weights 
######################################################################################

weights_norm = weights_calc * N 
Rpq =  weights_calc @  C_matrix @ weights_calc.T

print("\n" + "="*70)
print("MATRICE DE COVARIANCE DES MOMENTS R_pq (9×9)")
print("="*70)
print(Rpq)


###############################################################################################################
# results saves : 
###############################################################################################################


# you can change here where does the .txt file goes
output_file = f'/renoir/fromenti/Documents/codes_Bulk_flow/results_flip/Rpq_{N}_l_b_10.txt'

with open(output_file, "w") as f:

    f.write(f"N = {N}\n")
    f.write(f"Time = {end - start:.4f} seconds\n\n")

    for row in Rpq:
        f.write("  ".join(f"{x:10.4f}" for x in row) + "\n")

print(f"Results saved in: {output_file}")



