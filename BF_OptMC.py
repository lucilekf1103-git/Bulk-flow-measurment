import math
import numpy as np
import emcee
import scipy as sp
from itertools import chain
from CosmoFunc import *
from scipy.interpolate import splev, splrep



Opt_method = 'Nelder-Mead' 
'''
'Nelder-Mead' 
'Powell'    
'CG'  
'BFGS'  
'Newton-CG'  
'L-BFGS-B'  
'TNC'  
'COBYLA'  
'SLSQP'  
'trust-constr' 
'dogleg'  
'trust-ncg'  
'trust-exact'  
'trust-krylov'  
'''

# Distance and velocity estimator:---------------------------------------------
def DRcon(xdt,types,OmegaMs,OmegaAs,Hubs):
    x=np.linspace(-1.,100.,1000)
    y=np.zeros(len(x))
    for i in range(len(x)): 
        y[i]=DistDc(x[i],OmegaMs,OmegaAs, 0.0,Hubs,-1.0, 0.0, 0.0)
    spl_z2d = splrep(x, y, s=0)
    spl_d2z = splrep(y, x, s=0)
    if(types=='z2d'):
        Distc        = splev(xdt, spl_z2d)
        return Distc
    if(types=='d2z'):
        RSFs        = splev(xdt, spl_d2z)
        return RSFs
def Vpec_Fun_tra(Rsf,Logd,OmegaM,OmegaA, Hub):
    dz=DRcon(Rsf,'z2d',OmegaM,OmegaA, Hub)
    dh=dz*10.0**(-Logd)
    zh=DRcon(dh,'d2z',OmegaM,OmegaA, Hub)
    v=LightSpeed *( Rsf - zh )/( 1.0 + zh )
    return v
def Vpec_Fun_wat(Rsf,Logd,OmegaM,OmegaA, Hub):
    deccel = 3.0*OmegaM/2.0 - 1.0
    Vmod   = Rsf*LightSpeed*(1.0 + 0.5*(1.0 - deccel)*Rsf - (2.0 - deccel - 3.0*deccel*deccel)*Rsf*Rsf/6.0)
    vpec   = math.log(10.)*Vmod/(1.0+Vmod/LightSpeed) * Logd
    return vpec
def Aij_inv(ERRs2,dist,HatR):
    #ERRs2=sigma_Vpec*sigma_Vpec+sigma_star*sigma_star
    A_ij=np.zeros((9,9))
    A_ij[0,0]=np.sum(   HatR[0,:]*HatR[0,:]               /ERRs2      )
    A_ij[0,1]=np.sum(   HatR[0,:]*HatR[1,:]               /ERRs2      )
    A_ij[0,2]=np.sum(   HatR[0,:]*HatR[2,:]               /ERRs2      )
    A_ij[0,3]=np.sum(   HatR[0,:]*dist*HatR[0,:]*HatR[0,:]  /ERRs2      )
    A_ij[0,4]=np.sum(   2.*HatR[0,:]*dist*HatR[0,:]*HatR[1,:]  /ERRs2      )
    A_ij[0,5]=np.sum(   2.*HatR[0,:]*dist*HatR[0,:]*HatR[2,:]  /ERRs2      )
    A_ij[0,6]=np.sum(   HatR[0,:]*dist*HatR[1,:]*HatR[1,:]  /ERRs2      )
    A_ij[0,7]=np.sum(   2.*HatR[0,:]*dist*HatR[1,:]*HatR[2,:]  /ERRs2      )
    A_ij[0,8]=np.sum(   HatR[0,:]*dist*HatR[2,:]*HatR[2,:]  /ERRs2      )
    #print(np.sum(   2.*HatR[0,:]*dist*HatR[1,:]*HatR[2,:]  /ERRs2      ),np.sum(   2.*HatR[0,:]*dist*HatR[2,:]*HatR[1,:]  /ERRs2      )    )

    A_ij[1,1]=np.sum(   HatR[1,:]*HatR[1,:]               /ERRs2      )
    A_ij[1,2]=np.sum(   HatR[1,:]*HatR[2,:]               /ERRs2      )
    A_ij[1,3]=np.sum(   HatR[1,:]*dist*HatR[0,:]*HatR[0,:]  /ERRs2      )
    A_ij[1,4]=np.sum(   2.*HatR[1,:]*dist*HatR[0,:]*HatR[1,:]  /ERRs2      )
    A_ij[1,5]=np.sum(   2.*HatR[1,:]*dist*HatR[0,:]*HatR[2,:]  /ERRs2      )
    A_ij[1,6]=np.sum(   HatR[1,:]*dist*HatR[1,:]*HatR[1,:]  /ERRs2      )
    A_ij[1,7]=np.sum(   2.*HatR[1,:]*dist*HatR[1,:]*HatR[2,:]  /ERRs2      )
    A_ij[1,8]=np.sum(   HatR[1,:]*dist*HatR[2,:]*HatR[2,:]  /ERRs2      )

    A_ij[2,2]=np.sum(   HatR[2,:]*HatR[2,:]               /ERRs2      )
    A_ij[2,3]=np.sum(   HatR[2,:]*dist*HatR[0,:]*HatR[0,:]  /ERRs2      )
    A_ij[2,4]=np.sum(   2.*HatR[2,:]*dist*HatR[0,:]*HatR[1,:]  /ERRs2      )
    A_ij[2,5]=np.sum(   2.*HatR[2,:]*dist*HatR[0,:]*HatR[2,:]  /ERRs2      )
    A_ij[2,6]=np.sum(   HatR[2,:]*dist*HatR[1,:]*HatR[1,:]  /ERRs2      )
    A_ij[2,7]=np.sum(   2.*HatR[2,:]*dist*HatR[1,:]*HatR[2,:]  /ERRs2      )
    A_ij[2,8]=np.sum(   HatR[2,:]*dist*HatR[2,:]*HatR[2,:]  /ERRs2      )

    A_ij[3,3]=np.sum(   dist*dist*HatR[0,:]*HatR[0,:]*HatR[0,:]*HatR[0,:]  /ERRs2      )
    A_ij[3,4]=np.sum(   2.*dist*dist*HatR[0,:]*HatR[0,:]*HatR[0,:]*HatR[1,:]  /ERRs2      )
    A_ij[3,5]=np.sum(   2.*dist*dist*HatR[0,:]*HatR[0,:]*HatR[0,:]*HatR[2,:]  /ERRs2      )
    A_ij[3,6]=np.sum(   dist*dist*HatR[0,:]*HatR[0,:]*HatR[1,:]*HatR[1,:]  /ERRs2      )
    A_ij[3,7]=np.sum(   2.*dist*dist*HatR[0,:]*HatR[0,:]*HatR[1,:]*HatR[2,:]  /ERRs2      )
    A_ij[3,8]=np.sum(   dist*dist*HatR[0,:]*HatR[0,:]*HatR[2,:]*HatR[2,:]  /ERRs2      )

    A_ij[4,4]=np.sum(   2.*2.*dist*dist*HatR[0,:]*HatR[1,:]*HatR[0,:]*HatR[1,:]  /ERRs2      )
    A_ij[4,5]=np.sum(   2.*2.*dist*dist*HatR[0,:]*HatR[1,:]*HatR[0,:]*HatR[2,:]  /ERRs2      )
    A_ij[4,6]=np.sum(   2.*dist*dist*HatR[0,:]*HatR[1,:]*HatR[1,:]*HatR[1,:]  /ERRs2      )
    A_ij[4,7]=np.sum(   2.*2.*dist*dist*HatR[0,:]*HatR[1,:]*HatR[1,:]*HatR[2,:]  /ERRs2      )
    A_ij[4,8]=np.sum(   2.*dist*dist*HatR[0,:]*HatR[1,:]*HatR[2,:]*HatR[2,:]  /ERRs2      )

    A_ij[5,5]=np.sum(   2.*2.*dist*dist*HatR[0,:]*HatR[2,:]*HatR[0,:]*HatR[2,:]  /ERRs2      )
    A_ij[5,6]=np.sum(   2.*dist*dist*HatR[0,:]*HatR[2,:]*HatR[1,:]*HatR[1,:]  /ERRs2      )
    A_ij[5,7]=np.sum(   2.*2.*dist*dist*HatR[0,:]*HatR[2,:]*HatR[1,:]*HatR[2,:]  /ERRs2      )
    A_ij[5,8]=np.sum(   2.*dist*dist*HatR[0,:]*HatR[2,:]*HatR[2,:]*HatR[2,:]  /ERRs2      )

    A_ij[6,6]=np.sum(   dist*dist*HatR[1,:]*HatR[1,:]*HatR[1,:]*HatR[1,:]  /ERRs2      )
    A_ij[6,7]=np.sum(   2.*dist*dist*HatR[1,:]*HatR[1,:]*HatR[1,:]*HatR[2,:]  /ERRs2      )
    A_ij[6,8]=np.sum(   dist*dist*HatR[1,:]*HatR[1,:]*HatR[2,:]*HatR[2,:]  /ERRs2      )

    A_ij[7,7]=np.sum(   2.*2.*dist*dist*HatR[1,:]*HatR[2,:]*HatR[1,:]*HatR[2,:]  /ERRs2      )
    A_ij[7,8]=np.sum(   2.*dist*dist*HatR[1,:]*HatR[2,:]*HatR[2,:]*HatR[2,:]  /ERRs2      )

    A_ij[8,8]=np.sum(   dist*dist*HatR[2,:]*HatR[2,:]*HatR[2,:]*HatR[2,:]  /ERRs2      )

    for ia in range(9):
        for ja in range(9):
          if (ja>ia):  
            A_ij[ ja,ia ]=A_ij[ ia,ja ]
    
    Ain_ij=np.linalg.inv(A_ij)
     
    weightss=np.zeros((9,len(dist)))
    for ia in range(9):
      for n in range(len(dist)):
        weightss[ia,n] =(    Ain_ij[ia,0]*HatR[0,n]  +Ain_ij[ia,1]*HatR[1,n] +Ain_ij[ia,2]*HatR[2,n] +Ain_ij[ia,3]*dist[n]*HatR[0,n]*HatR[0,n] +2.*Ain_ij[ia,4]*dist[n]*HatR[0,n]*HatR[1,n] +2.*Ain_ij[ia,5]*dist[n]*HatR[0,n]*HatR[2,n]  +Ain_ij[ia,6]*dist[n]*HatR[1,n]*HatR[1,n]   +2.*Ain_ij[ia,7]*dist[n]*HatR[1,n]*HatR[2,n]  +Ain_ij[ia,8]*dist[n]*HatR[2,n]*HatR[2,n]    ) /ERRs2[n]
    return weightss,A_ij
def skewnormPar(alpha,mea,errs):
    varn = errs**2
    delt = alpha/np.sqrt(1.+alpha**2)
    muz  = np.sqrt(2./np.pi) * delt
    sigz = np.sqrt(1.-muz**2)
    gam  = (4.-np.pi)/2. * (   delt*np.sqrt(2./np.pi)   )**3/(    1.-2.*delt**2/np.pi    )**(3./2.)
    m0   = muz-gam*sigz/2.-np.sign(alpha)/2.*np.exp(-2.*np.pi/np.abs(alpha))
    omg  = np.sqrt(varn/(1.-2.*delt**2/np.pi))
    xi   = mea-omg*delt*np.sqrt(2./np.pi)
    return xi,omg











# Prior PDF:-------------------------------------------------------------------
def lnprior(params):
    Bx, By, Bz,sigma_star_vp = params    
    interval_v = 1200.0 # must greater than 0.
    # Flat prior for Bx
    if (-interval_v <= Bx <= interval_v):
        Bx_prior = 1.0/(2.0*interval_v)
    else:
        return -np.inf   
    # Flat prior for By
    if (-interval_v <= By <= interval_v):
        By_prior = 1.0/(2.0*interval_v)
    else:
        return -np.inf   
    # Flat prior for Bz
    if (-interval_v <= Bz <= interval_v):
        Bz_prior = 1.0/(2.0*interval_v)
    else:
        return -np.inf    
    # Flat prior for sigma_star_vp
    if (0.0 <= sigma_star_vp <= interval_v):
        sigma_star_vp_prior = 1.0/interval_v
    else:
        return -np.inf
    return 0.0
def lnpriorBQ(params,QzzTyp):
    if(QzzTyp=='free'):
        Bx, By, Bz, qxx, qxy, qxz, qyy, qyz,  qzz, sigma_star_vp = params
    if(QzzTyp=='fix'):    
        Bx, By, Bz, qxx, qxy, qxz, qyy, qyz,       sigma_star_vp = params  
    interval_v = 1200.0 
    interval_q = 100.0
    if (-interval_v <= Bx <= interval_v):
        Bx_prior = 1.0/(2.0*interval_v)
    else:
        return -np.inf   
    if (-interval_v <= By <= interval_v):
        By_prior = 1.0/(2.0*interval_v)
    else:
        return -np.inf   
    if (-interval_v <= Bz <= interval_v):
        Bz_prior = 1.0/(2.0*interval_v)
    else:
        return -np.inf  
    if (-interval_q <= qxx <= interval_q):
        qxx_prior = 1.0/(2.0*interval_q)
    else:
        return -np.inf
    if (-interval_q <= qxy <= interval_q):
        qxy_prior = 1.0/(2.0*interval_q)
    else:
        return -np.inf    
    if (-interval_q <= qxz <= interval_q):
        qxz_prior = 1.0/(2.0*interval_q)
    else:
        return -np.inf
    if (-interval_q <= qyy <= interval_q):
        qyy_prior = 1.0/(2.0*interval_q)
    else:
        return -np.inf   
    if (-interval_q <= qyz <= interval_q):
        qyz_prior = 1.0/(2.0*interval_q)
    else:
        return -np.inf
    if(QzzTyp=='free'):
        if (-interval_q <= qzz <= interval_q):
            qzz_prior = 1.0/(2.0*interval_q)
        else:
            return -np.inf
    if (0.0 <= sigma_star_vp <= interval_v):
        sigma_star_vp_prior = 1.0/interval_v
    else:
        return -np.inf
    return 0.0












# 2: Qin MLE:------------------------------------------------------------------
def lnlike_qMLE(params,hatr,vpec,EV,nvs,detls,z,Hub,sigma_star_adj,typs,CFtyp,QzzTyp):
    if(CFtyp=='BF'):
        Bx, By, Bz, sigma_star_vp = params
        Vps       = Bx* hatr[0,:]  +By*hatr[1,:]  +Bz*hatr[2,:]
        XC        = Vps+detls 
        #if(XC<0):
        #    objval=-1.*np.infty
        #    if(typs=='MCMC'):
        #        return objval
        #    if(typs=='Opt'):
        #        return -1.*objval        
        #if(XC>=0):
        #    Vp      = (XC**nvs-1.0) / nvs  
        Vp = np.zeros(len(Vps))      
        Vp[XC>=0.]= (XC[XC>=0.]**nvs[XC>=0.]-1.0) / nvs[XC>=0.]
        Vp[XC< 0.]= (-np.abs(XC[XC< 0.])**nvs[XC< 0.]-1.0) / nvs[XC< 0.] 
        # Notice: The BC-tansform of sigma_star_vp should be:
        # ( ( sigma_star_vp + detls )**nvs - 1.0 ) / nvs
        # But due to the fact that the best estimated sigma_star_vp only 
        # need to keep (sigma_star_vp + detls) >=0, so estimated sigma_star_vp
        # may smaller than 0. However, in the 'lnpriorBQ' and 'lnprior',  
        # sigma_star_vp is set to be greater than 0, so we set detls=0 
        # in the BC-tansform of sigma_star_vp. Be aware that the output
        # sigma_star_vp may be vary large, since the out put sigma_star_vp,
        # in fact, represents 'sigma_star_vp+delt' rather than 'sigma_star_vp'.
        # Using evpec2  = EV**2+(((   sigma_star_vp + 0.0   )**nvs-1.0)/nvs)**2 
        # can increasing the intrinsic scatter of bulkflow.
        #evpec2  = EV**2+(((   sigma_star_vp + 0.0   )**nvs-1.0)/nvs)**2 
        sigma_star_vp=sigma_star_vp*sigma_star_adj ; evpec2  = EV**2+sigma_star_vp**2
        LnPeta  = -np.log(np.sqrt(2.*math.pi*evpec2))-(Vp-vpec)*(Vp-vpec)/(2.*evpec2)
        objval  = np.nansum(LnPeta)
    if(CFtyp=='BQ'):
        if(QzzTyp=='free'):
            Bx, By, Bz, qxx, qxy, qxz, qyy, qyz, qzz, sigma_star_vp = params
        if(QzzTyp=='fix'):
            Bx, By, Bz, qxx, qxy, qxz, qyy, qyz, sigma_star_vp = params
            qzz=-qxx-qyy
        cl = LightSpeed
        Br = Bx* hatr[0,:]  +By*hatr[1,:]  +Bz*hatr[2,:]
        rQr= (hatr[0,:]*qxx +hatr[1,:]*qxy +hatr[2,:]*qxz)*hatr[0,:]+ (hatr[0,:]*qxy +hatr[1,:]*qyy +hatr[2,:]*qyz)*hatr[1,:]+ (hatr[0,:]*qxz +hatr[1,:]*qyz +hatr[2,:]*qzz)*hatr[2,:]
        dh = (-Br*Hub - cl*Hub -cl*rQr+ np.sqrt((Br*Hub+cl*Hub+cl*rQr)**2+4.*Hub*rQr*(-Br*cl+cl**2*z))) / (2.*Hub*rQr)
        Vps = Bx* hatr[0,:]  +By*hatr[1,:]  +Bz*hatr[2,:] + dh*(hatr[0,:]*qxx +hatr[1,:]*qxy +hatr[2,:]*qxz)*hatr[0,:] + dh*(hatr[0,:]*qxy +hatr[1,:]*qyy +hatr[2,:]*qyz)*hatr[1,:] + dh*(hatr[0,:]*qxz +hatr[1,:]*qyz +hatr[2,:]*qzz)*hatr[2,:]
        XC        = Vps+detls 
        Vp = np.zeros(len(Vps))      
        Vp[XC>=0.]= (XC[XC>=0.]**nvs[XC>=0.]-1.0) / nvs[XC>=0.]
        Vp[XC< 0.]= (-np.abs(XC[XC< 0.])**nvs[XC< 0.]-1.0) / nvs[XC< 0.]
        #evpec2  = EV**2+(((   sigma_star_vp + 0.0   )**nvs-1.0)/nvs)**2 
        sigma_star_vp=sigma_star_vp*sigma_star_adj ; evpec2  = EV**2+sigma_star_vp**2
        LnPeta  = -np.log(np.sqrt(2.*math.pi*evpec2))-(Vp-vpec)*(Vp-vpec)/(2.*evpec2)
        objval  = np.nansum(LnPeta)                
    if(typs=='MCMC'):
        return objval
    if(typs=='Opt'):
        return -1.*objval
def lnpost_qMLE(params,hatr,vpec,EV,nvs,detls,z,Hub,sigma_star_adj,typs,CFtyp,QzzTyp):
    if(CFtyp=='BF'):
        prior = lnprior(params)
    if(CFtyp=='BQ'):
        prior = lnpriorBQ(params,QzzTyp)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike_qMLE(params,hatr,vpec,EV,nvs,detls,z,Hub,sigma_star_adj,typs,CFtyp,QzzTyp)
    return prior + like












# 3: Eta MLE:------------------------------------------------------------------
def lnlike_etaMLE(params,hatr,logd,elogd,z,dz,EZ,Hub,dist_spline,typs,CFtyp,QzzTyp,sketamle,alpha):    
    if(CFtyp=='BF'):
        Bx, By, Bz, sigma_star_vp = params
        Vp = Bx* hatr[0,:]  +By*hatr[1,:]  +Bz*hatr[2,:]
        zh    =  (z-Vp/LightSpeed)  / (1. + Vp/LightSpeed)
        dh    =  sp.interpolate.splev(zh,dist_spline)   
        ETA= np.log10(dz/dh)
        elogd2 = elogd**2 + (sigma_star_vp/np.log(10.) * (1.+z)/(Hub*EZ*dz))**2
        if(sketamle):
            LnPeta=-np.log(np.sqrt(2.*math.pi*elogd2))-(ETA-logd)*(ETA-logd)/(2.*elogd2)+np.log(1.+ special.erf( alpha*(ETA-logd)/(np.sqrt(elogd2)*np.sqrt(2.0)) )  )
        else:
            LnPeta=-np.log(np.sqrt(2.*math.pi*elogd2))-(ETA-logd)*(ETA-logd)/(2.*elogd2)
        #inds    = np.where(np.isnan(LnPeta))
        #LnPeta[inds]=np.max(LnPeta)*10**(10)
        objval=np.nansum(LnPeta)   
    if(CFtyp=='BQ'):  
        if(QzzTyp=='free'):
            Bx, By, Bz, qxx, qxy, qxz, qyy, qyz, qzz, sigma_star_vp = params
        if(QzzTyp=='fix'):
            Bx, By, Bz, qxx, qxy, qxz, qyy, qyz, sigma_star_vp = params
            qzz=-qxx-qyy
        cz = z*LightSpeed
        Br = Bx* hatr[0,:]  +By*hatr[1,:]  +Bz*hatr[2,:]
        rQr= (hatr[0,:]*qxx +hatr[1,:]*qxy +hatr[2,:]*qxz)*hatr[0,:]+ (hatr[0,:]*qxy +hatr[1,:]*qyy +hatr[2,:]*qyz)*hatr[1,:]+ (hatr[0,:]*qxz +hatr[1,:]*qyz +hatr[2,:]*qzz)*hatr[2,:]
        dh = (-Br*Hub - LightSpeed*Hub -LightSpeed*rQr+ np.sqrt((Br*Hub+LightSpeed*Hub+LightSpeed*rQr)**2+4.*Hub*rQr*(-Br*LightSpeed+LightSpeed**2*z))) / (2.*Hub*rQr)
        Vp = Bx* hatr[0,:]  +By*hatr[1,:]  +Bz*hatr[2,:] + dh*(hatr[0,:]*qxx +hatr[1,:]*qxy +hatr[2,:]*qxz)*hatr[0,:] + dh*(hatr[0,:]*qxy +hatr[1,:]*qyy +hatr[2,:]*qyz)*hatr[1,:] + dh*(hatr[0,:]*qxz +hatr[1,:]*qyz +hatr[2,:]*qzz)*hatr[2,:]
        zh    =  (z-Vp/LightSpeed)  / (1. + Vp/LightSpeed)
        Dh    =  sp.interpolate.splev(zh,dist_spline)
        ETA= np.log10(dz/Dh)
        elogd2 = elogd**2 + (sigma_star_vp/np.log(10.) * (1.+z)/(Hub*EZ*dz))**2
        if(sketamle):
            LnPeta=-np.log(np.sqrt(2.*math.pi*elogd2))-(ETA-logd)*(ETA-logd)/(2.*elogd2)+np.log(1.+ special.erf( alpha*(ETA-logd)/(np.sqrt(elogd2)*np.sqrt(2.0)) )  )
        else:
            LnPeta=-np.log(np.sqrt(2.*math.pi*elogd2))-(ETA-logd)*(ETA-logd)/(2.*elogd2)     
        objval=np.nansum(LnPeta)   
    if(typs=='MCMC'):
        return objval
    if(typs=='Opt'):
        return -1.*objval
def lnpost_etaMLE(params,hatr,logd,elogd,z,dz,EZ,Hub,dist_spline,typs,CFtyp,QzzTyp,sketamle,alpha):
    if(CFtyp=='BF'):
        prior = lnprior(params)
    if(CFtyp=='BQ'):
        prior = lnpriorBQ(params,QzzTyp)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike_etaMLE(params,hatr,logd,elogd,z,dz,EZ,Hub,dist_spline,typs,CFtyp,QzzTyp,sketamle,alpha)
    return prior + like
    











# 4: wat MLE:------------------------------------------------------------------
def lnlike_wMLE(params,hatr,vpec,EV,z,Hub,typs,CFtyp,QzzTyp): 
    if(CFtyp=='BF'):
        Bx, By, Bz, sigma_star_vp = params
        Vp = Bx* hatr[0,:]  +By*hatr[1,:]  +Bz*hatr[2,:]
        evpec2 = EV**2+sigma_star_vp**2
        LnPeta=-np.log(np.sqrt(2.*math.pi*evpec2))-(Vp-vpec)*(Vp-vpec)/(2.*evpec2)
        objval=np.sum(LnPeta)
    if(CFtyp=='BQ'):
        if(QzzTyp=='free'):
            Bx, By, Bz, qxx, qxy, qxz, qyy, qyz, qzz, sigma_star_vp = params
        if(QzzTyp=='fix'):
            Bx, By, Bz, qxx, qxy, qxz, qyy, qyz, sigma_star_vp = params
            qzz=-qxx-qyy
        cl = LightSpeed
        Br = Bx* hatr[0,:]  +By*hatr[1,:]  +Bz*hatr[2,:]
        rQr= (hatr[0,:]*qxx +hatr[1,:]*qxy +hatr[2,:]*qxz)*hatr[0,:]+ (hatr[0,:]*qxy +hatr[1,:]*qyy +hatr[2,:]*qyz)*hatr[1,:]+ (hatr[0,:]*qxz +hatr[1,:]*qyz +hatr[2,:]*qzz)*hatr[2,:]
        dh = (-Br*Hub - cl*Hub -cl*rQr+ np.sqrt((Br*Hub+cl*Hub+cl*rQr)**2+4.*Hub*rQr*(-Br*cl+cl**2*z))) / (2.*Hub*rQr)
        Vp = Bx* hatr[0,:]  +By*hatr[1,:]  +Bz*hatr[2,:] + dh*(hatr[0,:]*qxx +hatr[1,:]*qxy +hatr[2,:]*qxz)*hatr[0,:] + dh*(hatr[0,:]*qxy +hatr[1,:]*qyy +hatr[2,:]*qyz)*hatr[1,:] + dh*(hatr[0,:]*qxz +hatr[1,:]*qyz +hatr[2,:]*qzz)*hatr[2,:]
        evpec2 = EV**2+sigma_star_vp**2
        LnPeta=-np.log(np.sqrt(2.*math.pi*evpec2))-(Vp-vpec)*(Vp-vpec)/(2.*evpec2)
        objval=np.nansum(LnPeta)
    if(typs=='MCMC'):
        return objval
    if(typs=='Opt'):
        return -1.*objval
def lnpost_wMLE(params,hatr,vpec,EV,z,Hub,typs,CFtyp,QzzTyp):
    if(CFtyp=='BF'):
        prior = lnprior(params)
    if(CFtyp=='BQ'):
        prior = lnpriorBQ(params,QzzTyp)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike_wMLE(params,hatr,vpec,EV,z,Hub,typs,CFtyp,QzzTyp)
    return prior + like
    












# 5 Optimize:------------------------------------------------------------------
def hatr_Fun(ra,dec):
    hatr      = np.array([[0.]*len(ra)]*3)
    hatr[0,:] = np.cos(math.pi*dec/180.0)*np.cos(math.pi*ra/180.0)
    hatr[1,:] = np.cos(math.pi*dec/180.0)*np.sin(math.pi*ra/180.0)
    hatr[2,:] = np.sin(math.pi*dec/180.0)
    return hatr
    
def Opt_qMLE(ra,dec,vpec,EV,nvs,detls,z,Hub,sigma_star_adj,CFtyp,QzzTyp): 
    typs      = 'Opt'
    hatr      = hatr_Fun(ra,dec)
    if(CFtyp=='BF'):
        QzzTyp    = 'fix'
        outpf     = sp.optimize.minimize(lnlike_qMLE,[200.,200.,200.,300.],args=(hatr,vpec,EV,nvs,detls,z,Hub,sigma_star_adj,typs,CFtyp,QzzTyp),method=Opt_method,tol=10.**(-11))
        Ux        = outpf.x[0]  ;  Uy = outpf.x[1]  ;  Uz = outpf.x[2]
        sigv      = outpf.x[3]
        return   np.asarray([Ux,Uy,Uz  ]) ,sigv 
    if(CFtyp=='BQ'):    
        if(QzzTyp== 'fix'):
            outpf     = sp.optimize.minimize(lnlike_qMLE,[200.,200.,200.,2.5,2.5,2.5,2.5,2.5,300.],args=(hatr,vpec,EV,nvs,detls,z,Hub,sigma_star_adj,typs,CFtyp,QzzTyp),method=Opt_method,tol=10.**(-11))
            Ux        = outpf.x[0]  ;  Uy = outpf.x[1]  ;  Uz = outpf.x[2]
            qxx       = outpf.x[3]  ;  qxy= outpf.x[4]  ;  qxz= outpf.x[5]  ; qyy= outpf.x[6]  ; qyz= outpf.x[7]   
            sigv      = outpf.x[8]    
            return    np.asarray([Ux,Uy,Uz]), np.asarray([qxx,qxy,qxz,qyy,qyz]), sigv
        if(QzzTyp== 'free'):
            outpf     = sp.optimize.minimize(lnlike_qMLE,[200.,200.,200.,2.5,2.5,2.5,2.5,2.5,2.5,300.],args=(hatr,vpec,EV,nvs,detls,z,Hub,sigma_star_adj,typs,CFtyp,QzzTyp),method=Opt_method,tol=10.**(-11))
            Ux        = outpf.x[0]  ;  Uy = outpf.x[1]  ;  Uz = outpf.x[2]
            qxx       = outpf.x[3]  ;  qxy= outpf.x[4]  ;  qxz= outpf.x[5]  ; qyy= outpf.x[6]  ; qyz= outpf.x[7]   
            qzz       = outpf.x[8]  ;  sigv      = outpf.x[9]    
            return    np.asarray([Ux,Uy,Uz]), np.asarray([qxx,qxy,qxz,qyy,qyz,qzz]), sigv


def Opt_etaMLE(ra,dec,z,logd,elogd,alpha,OmegaM,OmegaA,Hub,CFtyp,QzzTyp,sketamle): 
    if(not sketamle):
        alpha=np.nan
    if(sketamle):
        logd,elogd= skewnormPar(alpha,logd,elogd)
    dz = np.zeros(len(z));    EZ    = np.zeros(len(z)); 
    for j in range(len(z)):
        EZ[j] = Ez(z[j], OmegaM,OmegaA, 0.0, -1.0, 0.0, 0.0)
        dz[j] = DistDc(z[j],OmegaM,OmegaA, 0.0,Hub,-1.0, 0.0, 0.0)
    dist = np.empty(2000);red = np.empty(2000)
    for j in range(2000):
        red[j] = j*0.4/2000
        dist[j] = DistDc(red[j],OmegaM,OmegaA, 0.0,Hub,-1.0, 0.0, 0.0)
    dist_spline = sp.interpolate.splrep(red, dist, s=0)    
    typs      = 'Opt'
    hatr      = hatr_Fun(ra,dec)
    if(CFtyp=='BF'):
        QzzTyp   = 'fix'
        outpf     = sp.optimize.minimize(lnlike_etaMLE,[200.,200.,200.,300.],args=(hatr,logd,elogd,z,dz,EZ,Hub,dist_spline,typs,CFtyp,QzzTyp,sketamle,alpha),method=Opt_method,tol=10.**(-11))
        Ux        = outpf.x[0]  ;  Uy = outpf.x[1]  ;  Uz = outpf.x[2]
        sigv      = outpf.x[3]
        return   np.asarray([Ux,Uy,Uz  ]) ,sigv 
    if(CFtyp=='BQ'):    
        if(QzzTyp== 'fix'):
            outpf     = sp.optimize.minimize(lnlike_etaMLE,[200.,200.,200.,2.5,2.5,2.5,2.5,2.5,300.],args=(hatr,logd,elogd,z,dz,EZ,Hub,dist_spline,typs,CFtyp,QzzTyp,sketamle,alpha),method=Opt_method,tol=10.**(-11))
            Ux        = outpf.x[0]  ;  Uy = outpf.x[1]  ;  Uz = outpf.x[2]
            qxx       = outpf.x[3]  ;  qxy= outpf.x[4]  ;  qxz= outpf.x[5]  ; qyy= outpf.x[6]  ; qyz= outpf.x[7]   
            sigv      = outpf.x[8]    
            return    np.asarray([Ux,Uy,Uz]), np.asarray([qxx,qxy,qxz,qyy,qyz]), sigv
        if(QzzTyp== 'free'):
            outpf     = sp.optimize.minimize(lnlike_etaMLE,[200.,200.,200.,2.5,2.5,2.5,2.5,2.5,2.5,300.],args=(hatr,logd,elogd,z,dz,EZ,Hub,dist_spline,typs,CFtyp,QzzTyp,sketamle,alpha),method=Opt_method,tol=10.**(-11))
            Ux        = outpf.x[0]  ;  Uy = outpf.x[1]  ;  Uz = outpf.x[2]
            qxx       = outpf.x[3]  ;  qxy= outpf.x[4]  ;  qxz= outpf.x[5]  ; qyy= outpf.x[6]  ; qyz= outpf.x[7]   
            qzz       = outpf.x[8]  ;  sigv      = outpf.x[9]    
            return    np.asarray([Ux,Uy,Uz]), np.asarray([qxx,qxy,qxz,qyy,qyz,qzz]), sigv
    
def Opt_wMLE(ra,dec,vpec,Ev,z,Hub,CFtyp,QzzTyp): 
    typs      = 'Opt'
    hatr      = hatr_Fun(ra,dec)
    if(CFtyp=='BF'):
        QzzTyp='fix'
        outpf     = sp.optimize.minimize(lnlike_wMLE,[200.,200.,200.,300.],args=(hatr,vpec,Ev,z,Hub,typs,CFtyp,QzzTyp),method=Opt_method,tol=10.**(-11))
        Ux        = outpf.x[0]  ;  Uy = outpf.x[1]  ;  Uz = outpf.x[2]
        sigv      = outpf.x[3]
        return   np.asarray([Ux,Uy,Uz]), sigv 
    if(CFtyp=='BQ'):    
        if(QzzTyp== 'fix'):
            outpf     = sp.optimize.minimize(lnlike_wMLE,[200.,200.,200.,2.5,2.5,2.5,2.5,2.5,300.],args=(hatr,vpec,Ev,z,Hub,typs,CFtyp,QzzTyp),method=Opt_method,tol=10.**(-11))
            Ux        = outpf.x[0]  ;  Uy = outpf.x[1]  ;  Uz = outpf.x[2]
            qxx       = outpf.x[3]  ;  qxy= outpf.x[4]  ;  qxz= outpf.x[5]  ; qyy= outpf.x[6]  ; qyz= outpf.x[7]   
            sigv      = outpf.x[8]    
            return    np.asarray([Ux,Uy,Uz]), np.asarray([qxx,qxy,qxz,qyy,qyz]), sigv
        if(QzzTyp== 'free'):
            outpf     = sp.optimize.minimize(lnlike_wMLE,[200.,200.,200.,2.5,2.5,2.5,2.5,2.5,2.5,300.],args=(hatr,vpec,Ev,z,Hub,typs,CFtyp,QzzTyp),method=Opt_method,tol=10.**(-11))
            Ux        = outpf.x[0]  ;  Uy = outpf.x[1]  ;  Uz = outpf.x[2]
            qxx       = outpf.x[3]  ;  qxy= outpf.x[4]  ;  qxz= outpf.x[5]  ; qyy= outpf.x[6]  ; qyz= outpf.x[7]   
            qzz       = outpf.x[8]  ;  sigv      = outpf.x[9]    
            return    np.asarray([Ux,Uy,Uz]), np.asarray([qxx,qxy,qxz,qyy,qyz,qzz]), sigv

def Opt_tMLE(ra,dec,vpec,Ev,z,Hub,CFtyp,QzzTyp):
    if(CFtyp=='BF'):
        a,b       = Opt_wMLE(ra,dec,vpec,Ev,z,Hub,CFtyp,QzzTyp)
        return   a,b      
    if(CFtyp=='BQ'):
        a,b,c     = Opt_wMLE(ra,dec,vpec,Ev,z,Hub,CFtyp,QzzTyp)
        return   a,b,c   
    
    
    
    
    
    
    
    
    
    
    
    
    
# 6 MCMC:---------------------------------------------------------------------- 
def MCparm(ndim, nwalkers,CFtyp,QzzTyp):
    if(CFtyp=='BF'):
        begin = [[200.0*np.random.rand()-100.0,
                  200.0*np.random.rand()-100.0,
                  200.0*np.random.rand()-100.0,
                  200.0*np.random.rand()+200.0] for k in range(nwalkers)]#
    if(CFtyp=='BQ'):
      if(QzzTyp== 'fix'): 
        begin = [[200.0*np.random.rand()-100.0,
                  200.0*np.random.rand()-100.0,
                  200.0*np.random.rand()-100.0,
                  5.0*np.random.rand()-2.5,
                  5.0*np.random.rand()-2.5,
                  5.0*np.random.rand()-2.5,
                  5.0*np.random.rand()-2.5,
                  5.0*np.random.rand()-2.5,
                  200.0*np.random.rand()+200.0] for k in range(nwalkers)]#
      if(QzzTyp== 'free'): 
        begin = [[200.0*np.random.rand()-100.0,
                  200.0*np.random.rand()-100.0,
                  200.0*np.random.rand()-100.0,
                  5.0*np.random.rand()-2.5,
                  5.0*np.random.rand()-2.5,
                  5.0*np.random.rand()-2.5,
                  5.0*np.random.rand()-2.5,
                  5.0*np.random.rand()-2.5,
                  5.0*np.random.rand()-2.5,
                  200.0*np.random.rand()+200.0] for k in range(nwalkers)]#
    return begin
    
def MC_qMLE(ra,dec,vpec,EV,nvs,detls,z,Hub,NmcSamp,sigma_star_adj,CFtyp,QzzTyp): 
    typs      = 'MCMC'
    hatr      = hatr_Fun(ra,dec)
    if(CFtyp=='BF'):
        ndim=4 ; nwalkers=24 ; begin = MCparm(4, 24, 'BF' ,'fix')
    if(CFtyp=='BQ'):
        if(QzzTyp== 'fix'):
            ndim=9 ; nwalkers=24 ; begin = MCparm(9, 24, 'BQ','fix')
        if(QzzTyp== 'free'):
            ndim=10 ; nwalkers=24 ; begin = MCparm(10, 24, 'BQ','free')
    sampler   = emcee.EnsembleSampler(nwalkers, ndim, lnpost_qMLE, args=[hatr,vpec,EV,nvs,detls,z,Hub,sigma_star_adj,typs,CFtyp,QzzTyp])
    pos, prob, state = sampler.run_mcmc(begin, NmcSamp[0])
    sampler.reset()
    sampler.run_mcmc(pos, NmcSamp[1])
    if(CFtyp=='BF')or(CFtyp=='BQ'):
        Ux = np.mean(sampler.flatchain[:,0])
        Uy = np.mean(sampler.flatchain[:,1])
        Uz = np.mean(sampler.flatchain[:,2])
        U_cov = np.cov([sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2]])
        if(CFtyp=='BF'):
            sigv =np.mean(sampler.flatchain[:,3])
            return   np.asarray([Ux,Uy,Uz]) ,sigv , U_cov, sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3]
    if(CFtyp=='BQ'):  
        Qxx=np.mean(sampler.flatchain[:,3])
        Qxy=np.mean(sampler.flatchain[:,4])
        Qxz=np.mean(sampler.flatchain[:,5])
        Qyy=np.mean(sampler.flatchain[:,6])
        Qyz=np.mean(sampler.flatchain[:,7])
        if(QzzTyp== 'fix'):
            eQzz=np.std(-sampler.flatchain[:,3]-sampler.flatchain[:,6])
            Q_cov = np.cov([sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7]])      
            Tot_cov=np.cov([sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7]])
            sigv =np.mean(sampler.flatchain[:,8])
            return   np.asarray([Ux,Uy,Uz]),np.asarray([Qxx,Qxy,Qxz,Qyy,Qyz]),sigv,U_cov,Q_cov,Tot_cov,eQzz,sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8]
        if(QzzTyp== 'free'):
            Qzz=np.mean(sampler.flatchain[:,8])
            Q_cov = np.cov([sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8]])      
            Tot_cov=np.cov([sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8]])
            sigv =np.mean(sampler.flatchain[:,9])
            return   np.asarray([Ux,Uy,Uz]),np.asarray([Qxx,Qxy,Qxz,Qyy,Qyz,Qzz]),sigv,U_cov,Q_cov,Tot_cov,sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8],sampler.flatchain[:,9]



def MC_etaMLE(ra,dec,z,logd,elogd,alpha,OmegaM,OmegaA,Hub,NmcSamp,CFtyp,QzzTyp,sketamle): 
    if(not sketamle):
        alpha=np.nan
    if(sketamle):
        logd,elogd= skewnormPar(alpha,logd,elogd)
    dz = np.zeros(len(z));    EZ    = np.zeros(len(z)); 
    for j in range(len(z)):
        EZ[j] = Ez(z[j], OmegaM,OmegaA, 0.0, -1.0, 0.0, 0.0)
        dz[j] = DistDc(z[j],OmegaM,OmegaA, 0.0,Hub,-1.0, 0.0, 0.0)
    dist = np.empty(2000);red = np.empty(2000)
    for j in range(2000):
        red[j] = j*0.4/2000
        dist[j] = DistDc(red[j],OmegaM,OmegaA, 0.0,Hub,-1.0, 0.0, 0.0)
    dist_spline = sp.interpolate.splrep(red, dist, s=0)
    typs      = 'MCMC'
    hatr      = hatr_Fun(ra,dec)
    if(CFtyp=='BF'):
        ndim=4 ; nwalkers=24 ; begin = MCparm(4, 24, 'BF' ,'fix')
    if(CFtyp=='BQ'):
        if(QzzTyp== 'fix'):
            ndim=9 ; nwalkers=24 ; begin = MCparm(9, 24, 'BQ','fix')
        if(QzzTyp== 'free'):
            ndim=10 ; nwalkers=24 ; begin = MCparm(10, 24, 'BQ','free')
    sampler   = emcee.EnsembleSampler(nwalkers, ndim, lnpost_etaMLE, args=[hatr,logd,elogd,z,dz,EZ,Hub,dist_spline,typs,CFtyp,QzzTyp,sketamle,alpha])
    pos, prob, state = sampler.run_mcmc(begin, NmcSamp[0])
    sampler.reset()
    sampler.run_mcmc(pos, NmcSamp[1])
    if(CFtyp=='BF')or(CFtyp=='BQ'):
        Ux = np.mean(sampler.flatchain[:,0])
        Uy = np.mean(sampler.flatchain[:,1])
        Uz = np.mean(sampler.flatchain[:,2])
        U_cov = np.cov([sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2]])
        if(CFtyp=='BF'):
            sigv =np.mean(sampler.flatchain[:,3])
            return   np.asarray([Ux,Uy,Uz]) ,sigv , U_cov, sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3]
    if(CFtyp=='BQ'):  
        Qxx=np.mean(sampler.flatchain[:,3])
        Qxy=np.mean(sampler.flatchain[:,4])
        Qxz=np.mean(sampler.flatchain[:,5])
        Qyy=np.mean(sampler.flatchain[:,6])
        Qyz=np.mean(sampler.flatchain[:,7])
        if(QzzTyp== 'fix'):
            eQzz=np.std(-sampler.flatchain[:,3]-sampler.flatchain[:,6])
            Q_cov = np.cov([sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7]])      
            Tot_cov=np.cov([sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7]])
            sigv =np.mean(sampler.flatchain[:,8])
            return   np.asarray([Ux,Uy,Uz]),np.asarray([Qxx,Qxy,Qxz,Qyy,Qyz]),sigv,U_cov,Q_cov,Tot_cov,eQzz,sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8]
        if(QzzTyp== 'free'):
            Qzz=np.mean(sampler.flatchain[:,8])
            Q_cov = np.cov([sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8]])      
            Tot_cov=np.cov([sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8]])
            sigv =np.mean(sampler.flatchain[:,9])
            return   np.asarray([Ux,Uy,Uz]),np.asarray([Qxx,Qxy,Qxz,Qyy,Qyz,Qzz]),sigv,U_cov,Q_cov,Tot_cov,sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8],sampler.flatchain[:,9]

def MC_wMLE(ra,dec,vpec,EV,z,Hub,NmcSamp,CFtyp,QzzTyp): 
    typs      = 'MCMC'
    hatr      = hatr_Fun(ra,dec)
    if(CFtyp=='BF'):
        ndim=4 ; nwalkers=24 ; begin = MCparm(4, 24, 'BF' ,'fix')
    if(CFtyp=='BQ'):
        if(QzzTyp== 'fix'):
            ndim=9 ; nwalkers=24 ; begin = MCparm(9, 24, 'BQ','fix')
        if(QzzTyp== 'free'):
            ndim=10 ; nwalkers=24 ; begin = MCparm(10, 24, 'BQ','free')
    sampler   = emcee.EnsembleSampler(nwalkers, ndim, lnpost_wMLE, args=[hatr,vpec,EV,z,Hub,typs,CFtyp,QzzTyp])
    pos, prob, state = sampler.run_mcmc(begin, NmcSamp[0])
    sampler.reset()
    sampler.run_mcmc(pos, NmcSamp[1])
    if(CFtyp=='BF')or(CFtyp=='BQ'):
        Ux = np.mean(sampler.flatchain[:,0])
        Uy = np.mean(sampler.flatchain[:,1])
        Uz = np.mean(sampler.flatchain[:,2])
        U_cov = np.cov([sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2]])
        if(CFtyp=='BF'):
            sigv =np.mean(sampler.flatchain[:,3])
            return   np.asarray([Ux,Uy,Uz]) ,sigv , U_cov, sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3]
    if(CFtyp=='BQ'):  
        Qxx=np.mean(sampler.flatchain[:,3])
        Qxy=np.mean(sampler.flatchain[:,4])
        Qxz=np.mean(sampler.flatchain[:,5])
        Qyy=np.mean(sampler.flatchain[:,6])
        Qyz=np.mean(sampler.flatchain[:,7])
        if(QzzTyp== 'fix'):
            eQzz=np.std(-sampler.flatchain[:,3]-sampler.flatchain[:,6])
            Q_cov = np.cov([sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7]])      
            Tot_cov=np.cov([sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7]])
            sigv =np.mean(sampler.flatchain[:,8])
            return   np.asarray([Ux,Uy,Uz]),np.asarray([Qxx,Qxy,Qxz,Qyy,Qyz]),sigv,U_cov,Q_cov,Tot_cov,eQzz,sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8]
        if(QzzTyp== 'free'):
            Qzz=np.mean(sampler.flatchain[:,8])
            Q_cov = np.cov([sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8]])      
            Tot_cov=np.cov([sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8]])
            sigv =np.mean(sampler.flatchain[:,9])
            return   np.asarray([Ux,Uy,Uz]),np.asarray([Qxx,Qxy,Qxz,Qyy,Qyz,Qzz]),sigv,U_cov,Q_cov,Tot_cov,sampler.flatchain[:,0],sampler.flatchain[:,1],sampler.flatchain[:,2],sampler.flatchain[:,3],sampler.flatchain[:,4],sampler.flatchain[:,5],sampler.flatchain[:,6],sampler.flatchain[:,7],sampler.flatchain[:,8],sampler.flatchain[:,9]

def MC_tMLE(ra,dec,vpec,Ev,z,Hub,NmcSamp,CFtyp,QzzTyp): 
    if(CFtyp=='BF'):
        BF,Siv,CovBF,S0,S1,S2,S3=MC_wMLE(ra,dec,vpec,Ev,z,Hub,NmcSamp,CFtyp,QzzTyp)
        return   BF,Siv,CovBF,S0,S1,S2,S3
    if(CFtyp=='BQ'):
      if(QzzTyp== 'fix'):
        BF,QS,sigv,U_cov,Q_cov,Tot_cov,eQzz,sampler0,sampler1,sampler2,sampler3,sampler4,sampler5,sampler6,sampler7,sampler8=MC_wMLE(ra,dec,vpec,Ev,z,Hub,NmcSamp,CFtyp,QzzTyp)
        return   BF,QS,sigv,U_cov,Q_cov,Tot_cov,eQzz,sampler0,sampler1,sampler2,sampler3,sampler4,sampler5,sampler6,sampler7,sampler8
      if(QzzTyp== 'free'):
        BF,QS,sigv,U_cov,Q_cov,Tot_cov,sampler0,sampler1,sampler2,sampler3,sampler4,sampler5,sampler6,sampler7,sampler8,sampler9=MC_wMLE(ra,dec,vpec,Ev,z,Hub,NmcSamp,CFtyp,QzzTyp)
        return  BF,QS,sigv,U_cov,Q_cov,Tot_cov,sampler0,sampler1,sampler2,sampler3,sampler4,sampler5,sampler6,sampler7,sampler8,sampler9











# 7 Jacbian error:------------------------------------------------------------- 
def BK_error(V_bkf,Reij):
    pi=math.pi
    V_bkf_mag=np.linalg.norm(V_bkf)
    Jaco=np.zeros(3) ; Jacop=np.zeros(3)
    Jaco[0] = V_bkf[0]/V_bkf_mag
    Jaco[1] = V_bkf[1]/V_bkf_mag
    Jaco[2] = V_bkf[2]/V_bkf_mag
    for j in range(3):
        Jacop[j] = 0.
        for i in range(3):
            Jacop[j] = Jacop[j]+Jaco[i]*Reij[i,j]
    suma=0.
    for j in range(3):
        suma = suma+Jacop[j]*Jaco[j]
    V_bkf_mag_err= np.sqrt(suma)
    #error of the bulkflow direction
    jaco_dir=np.zeros((2,3))
    jaco_dirpp=np.zeros(3)

    jaco_dir[0,0] = 180./pi*(-V_bkf[1])/(V_bkf[0]*V_bkf[0]*(1.+V_bkf[1]*V_bkf[1]/(V_bkf[0]*V_bkf[0])))
    jaco_dir[0,1] = 180./pi*1./(V_bkf[0]*(1.+V_bkf[1]*V_bkf[1]/(V_bkf[0]*V_bkf[0])))
    jaco_dir[0,2] = 0.
    jaco_dir[1,0] = 180./pi*(-V_bkf[2]*V_bkf[0])/(V_bkf_mag*V_bkf_mag*V_bkf_mag    * np.sqrt(1.-V_bkf[2]*V_bkf[2]/(V_bkf_mag*V_bkf_mag)))
    jaco_dir[1,1] = 180./pi*(-V_bkf[2]*V_bkf[1])/(V_bkf_mag*V_bkf_mag*V_bkf_mag    * np.sqrt(1.-V_bkf[2]*V_bkf[2]/(V_bkf_mag*V_bkf_mag)))
    jaco_dir[1,2] = 180./pi*(1./np.sqrt(1.-V_bkf[2]*V_bkf[2]/(V_bkf_mag*V_bkf_mag)))  * (V_bkf[2]*V_bkf[2]/(V_bkf_mag*V_bkf_mag*V_bkf_mag)-1./V_bkf_mag) #*(-1.)
 
    for j in range(3):
        jaco_dirpp[j]=0.
        for i in range(3):
            jaco_dirpp[j]=jaco_dirpp[j]+jaco_dir[ 0,i ]*Reij[ i,j ]
            
    suma=0.
    for j in range(3):
        suma = suma+jaco_dirpp[j]*jaco_dir[ 0,j ]
 
    longitude_bkf_err=np.sqrt(suma)

    for j in range(3):
        jaco_dirpp[j]=0.
        for i in range( 3):
            jaco_dirpp[j]=jaco_dirpp[j]+jaco_dir[ 1,i ]*Reij[ i,j ]

    suma=0.
    for j in range(3):
        suma = suma+jaco_dirpp[j]*jaco_dir[ 1,j ]
    latitude_bkf_err=np.sqrt(suma)

    return V_bkf_mag_err,longitude_bkf_err,latitude_bkf_err

###########################################################################
