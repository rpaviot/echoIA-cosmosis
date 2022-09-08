import numpy as np
import treecorr
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import treecorr
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as colors

"""Code to compute gamma_t and wg+ for lightcones with treecorr"""
class lensingPCF:
    def __init__(self,*args,**qwargs):
        

        """Init : First catalog: clustering catalog, second catalog source catalog"""
        self.Om =  qwargs['Om0']
        self.cosmo = FlatLambdaCDM(Om0=self.Om,H0=100)
        
        self.computation = qwargs['computation']
        self.units = qwargs['units']
        self.npatch = qwargs['npatch']
        
        if self.npatch == 1:
            self.var_method = "shot"
        else:
            self.var_method = "jackknife"
 
        
        self.z = None
        self.z2 = None
        self.rand1 = None
        self.rand2 = None
        self.cov = None
        
        ra =  qwargs['RA']
        dec =  qwargs['DEC']
        w =  qwargs['W']
        len1 = len(ra)
        
        ra2 =  qwargs['RA2']
        dec2 =  qwargs['DEC2']
        w2 =  qwargs['W2'] 
        g1 = qwargs['g1']
        g2 = qwargs['g2']
        len2 = len(ra2)
        
        if np.array_equal(ra,ra2):
            self.corr = "auto"
        else :
            self.corr = "cross"
    
        if 'Z' in qwargs:
            self.z =  qwargs['Z']
            dc = self.cosmo.comoving_distance(self.z).value
            self.data1 = treecorr.Catalog(ra=ra,dec=dec,r=dc,w=w,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch)
        else :
            self.data1 = treecorr.Catalog(ra=ra,dec=dec,w=w,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch)

        
        if 'Z2' in qwargs:
            self.z2 =  qwargs['Z2']
            dc2 = self.cosmo.comoving_distance(self.z2).value
            self.data2 = treecorr.Catalog(ra=ra2,dec=dec2,r=dc2,w=w2,g1=g1,g2=g2,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch,patch_centers=self.data1.patch_centers)
        else :
            self.data2 = treecorr.Catalog(ra=ra2,dec=dec2,w=w2,g1=g1,g2=g2,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch,patch_centers=self.data1.patch_centers)
            

        self.varg = treecorr.calculateVarG(self.data2)
            

            
            
    def set_random(self,*args,**qwargs):
        ra_r =  qwargs['RA_r']
        dec_r =  qwargs['DEC_r']
        w_r =  qwargs['W_r']        
        if 'Z_r' in qwargs:
            z_r =  qwargs['Z_r']
            dc_r = self.cosmo.comoving_distance(z_r).value
            self.rand1 = treecorr.Catalog(ra=ra_r,dec=dec_r,r=dc_r,w=w_r,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch,patch_centers=self.data1.patch_centers,is_rand=1)
        else :
            self.rand1 = treecorr.Catalog(ra=ra_r,dec=dec_r, w=w_r,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch,patch_centers=self.data1.patch_centers,is_rand=1)

        if 'RA_r2' in qwargs:
            ra_r2 =  qwargs['RA_r2']
            dec_r2 =  qwargs['DEC_r2']
            w_r2 =  qwargs['W_r2']        
            if 'Z_r2' in qwargs:
                z_r2 =  qwargs['Z_r2']
                dc_r2 = self.cosmo.comoving_distance(z_r2).value
                self.rand2 = treecorr.Catalog(ra=ra_r2,dec=dec_r2,r=dc_r2,w=w_r2,ra_units=self.units,\
                    dec_units=self.units,npatch=1,patch_centers=self.data1.patch_centers,is_rand=1)
            else :
                self.rand2 = treecorr.Catalog(ra=ra_r2,dec=dec_r2,w=w_r2,ra_units=self.units,\
                    dec_units=self.units,npatch=1,patch_centers=self.data1.patch_centers,is_rand=1)


    def compute_norm(self):
        
        if self.rand1 is None:
            raise ValueError("You must provide at least a random catalog")    
        
        if self.corr == "IA" and self.corr =="cross" and self.rand2 is None:
            raise ValueError("You must provide at least two random catalogs for wg+ cross estimation.")

        self.rgnorm = np.zeros(self.npatch)
        self.rrnorm = np.zeros(self.npatch)
        self.ngnorm = np.zeros(self.npatch)
        patchD = np.unique(self.data1.patch)

        if self.npatch == 1:
            self.rgnorm = self.rand1.sumw*self.data2.sumw
            if self.corr == "auto":
                self.ngnorm = (self.data1.sumw)**2 - np.sum((self.data1.w)**2)
                self.rrnorm = (self.rand1.sumw)**2 - np.sum((self.rand1.w)**2)
            elif self.corr == "cross":
                self.ngnorm = self.data1.sumw*self.data2.sumw
                if self.computation=="GG":
                    self.rrnorm = (self.rand1.sumw)**2 - np.sum((self.rand1.w)**2)
                elif self.computation=="IA":
                    self.rrnorm = self.rand1.sumw*self.rand2.sumw
        
        elif self.npatch > 1:
            for i in range(0,len(patchD)):
                cond1 = self.data1.patch == patchD[i]
                cond2 = self.data2.patch == patchD[i]
                cond3 = self.rand1.patch == patchD[i]
                wd1 = self.data1.w[~cond1]
                wd2 = self.data2.w[~cond2]
                wr1 = self.rand1.w[~cond3]
                #self.rgnorm[i] = self.rand1.sumw*self.data2.sumw
                self.rgnorm[i]=np.sum(wr1)*np.sum(wd2)
                if self.corr =="auto":
                    #self.ngnorm[i] = (self.data1.sumw)**2 - np.sum((self.data1.w)**2)
                    #self.rrnorm[i] = (self.rand1.sumw)**2 - np.sum((self.rand1.w)**2)
                    self.ngnorm[i]= (np.sum(wd1)**2 - np.sum(wd1**2))
                    self.rrnorm[i]=(np.sum(wr1)**2 - np.sum(wr1**2))
                elif self.corr == "cross":
                    self.ngnorm[i]= np.sum(wd1)*np.sum(wd2)                   
                    if self.computation=="GG":
                        self.rrnorm[i]=(np.sum(wr1)**2 - np.sum(wr1**2))
                    elif self.computation=="IA":
                        cond4 = np.where(self.rand2.patch == patchD[i])[0]
                        wr2 = self.rand2.w[cond4]
                        self.rrnorm[i] = np.sum(wr1)*np.sum(wr2)

    """Routines to compute wg+/gammat/wgg for a single patch"""

    def combine_pairs_DS(self,corrs):
        return corrs[0].xi*(corrs[0].weight/corrs[1].weight)*(self.rgnorm/self.ngnorm) - corrs[1].xi

    def combine_pairs_RS(self,corrs):
        return corrs[0].xi*(corrs[0].weight/corrs[2].weight)*(self.rrnorm/self.ngnorm) - \
            corrs[1].xi*(corrs[1].weight/corrs[2].weight)*(self.rrnorm/self.rgnorm)
    
    def combine_pairs_RS_clustering(self,corrs):
        return (corrs[0].npairs/corrs[2].npairs)*(self.rrnorm/self.ngnorm) - \
            2*(corrs[1].npairs/corrs[2].npairs)*(self.rrnorm/self.rgnorm) + 1.
    

    def combine_pairs_RS_proj(self,corrs):
        xirppi_t = np.zeros((len(self.pi)-1,self.nbins))
        ng = corrs[0:int(len(corrs)/3)]
        rg = corrs[int(len(corrs)/3):2*int(len(corrs)/3)]
        rr = corrs[2*int(len(corrs)/3):len(corrs)]
        for i in range(0,len(self.pi)-1):
            corrs = [ng[i],rg[i],rr[i]]
            xirppi_t[i] = self.combine_pairs_RS(corrs)
        xirppi_t = np.sum(xirppi_t*self.dpi,axis=0)
        return xirppi_t
    
    

    def combine_pairs_RS_proj_clustering(self,corrs):
        xirppi_t = np.zeros((len(self.pi)-1,self.nbins))
        ng = corrs[0:int(len(corrs)/3)]
        rg = corrs[int(len(corrs)/3):2*int(len(corrs)/3)]
        rr = corrs[2*int(len(corrs)/3):len(corrs)]
        for i in range(0,len(self.pi)-1):
            corrs = [ng[i],rg[i],rr[i]]
            xirppi_t[i] = self.combine_pairs_RS_clustering(corrs)
        xirppi_t = 2*np.sum(xirppi_t*self.dpi,axis=0)
        return xirppi_t
    
    
    
    def combine_pairs_DS_proj(self,corrs):
        xirppi_t = np.zeros((len(self.pi)-1,self.nbins))
        ng = corrs[0:int(len(corrs)/3)]
        rg = corrs[int(len(corrs)/3):2*int(len(corrs)/3)]
        for i in range(0,len(self.pi)-1):
            corrs = [ng[i],rg[i]]
            xirppi_t[i] = self.combine_pairs_DS(corrs)
        xirppi_t = np.sum(xirppi_t*self.dpi,axis=0)
        return xirppi_t
    
    """Routines to compute wg+/gammat for multiple patch"""


    def get_rppi_pairs(self,corrs,rand=False):
        xi_2d = np.zeros((self.npatch,len(self.pi)-1,self.nbins))
        w_2d = np.zeros((self.npatch,len(self.pi)-1,self.nbins))

        plist = [c._jackknife_pairs() for c in corrs]
        plist = list(zip(*plist))
        for row, pairs in enumerate(plist):
            k = 0
            for c, cpairs in zip(corrs, pairs):
                cpairs = list(cpairs)
                c._calculate_xi_from_pairs(cpairs)
                if rand is True:
                    w_2d[row][k] = c.weight
                else:
                    xi_2d[row][k] = c.xi
                    w_2d[row][k] = c.weight
                k = k+1

        return xi_2d,w_2d
    
    def get_rppi_pairs_clustering(self,corrs):
        xi_2d = np.zeros((self.npatch,len(self.pi)-1,self.nbins))

        plist = [c._jackknife_pairs() for c in corrs]
        plist = list(zip(*plist))
        for row, pairs in enumerate(plist):
            k = 0
            for c, cpairs in zip(corrs, pairs):
                xi_2d[row][k] = c.npairs
                k = k+1
        return xi_2d
    
    
    def get_rp_pairs(self,corrs):
        pairs_1d = np.zeros((self.npatch,self.nbins))
        w_1d = np.zeros((self.npatch,self.nbins))

        plist = [c._jackknife_pairs() for c in corrs]
        plist = list(zip(*plist))
        for row, pairs in enumerate(plist):
            for c, cpairs in zip(corrs, pairs):
                cpairs = list(cpairs)
                c._calculate_xi_from_pairs(cpairs)
    
            pairs_1d[row] = c.xi
            w_1d[row] = c.weight

        return pairs_1d,w_1d
    

    
    
    def combine_jack_pairs_rppi(self,NG,RG,wNG,wRG,wRR):
        for i in range(0,self.npatch):
            NG[i] = NG[i]/self.ngnorm[i]
            RG[i] = RG[i]/self.rgnorm[i]
            wRR[i] = wRR[i]/self.rrnorm[i]

        xirppi = (NG*wNG - RG*wRG)/wRR
        wgp = np.sum(xirppi*self.dpi,axis=1)
        wgp_mean = np.mean(wgp,axis=0)
        wgp = wgp - wgp_mean
        C = (1.-1./self.npatch)*np.dot(wgp.T,wgp)
        return wgp_mean,C
    
    
    def combine_jack_pairs_rppi_clustering(self,NG,RG,RR):
        for i in range(0,self.npatch):
            NG[i] = NG[i]/self.ngnorm[i]
            RG[i] = RG[i]/self.rgnorm[i]
            RR[i] = RR[i]/self.rrnorm[i]

        xirppi = NG/RR - 2*RG/RR + 1.
        wgp = 2*np.sum(xirppi*self.dpi,axis=1)
        wgp_mean = np.mean(wgp,axis=0)
        wgp = wgp - wgp_mean
        C = (1.-1./self.npatch)*np.dot(wgp.T,wgp)
        return wgp_mean,C
    
    def combine_jack_pairs(self,NG,RG,wNG,wRG):
        for i in range(0,self.npatch):
            NG[i] = NG[i]/self.ngnorm[i]
            RG[i] = RG[i]/self.rgnorm[i]
            wRG[i] = wRG[i]/self.rgnorm[i]

        xi = (NG*wNG - RG*wRG)/wRG
        xi_mean = np.mean(xi,axis=0)
        xi = xi - xi_mean
        C = (1.-1./self.npatch)*np.dot(xi.T,xi)
        return xi_mean,C
    

    def gammat(self,minr,maxr,nbins,sep_units=None,min_rpar=0):
        min_rpar = min_rpar
        sep_units = sep_units
        self.nbins = nbins
        self.compute_norm()
        """Distances sources > Distances lens + x Mpc """
     
        if self.z is None and self.rand1 is not None:
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,max_sep=maxr,\
                sep_units=sep_units,var_method=self.var_method)

            rg = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,max_sep=maxr,\
                sep_units=sep_units,var_method=self.var_method)

            ng.process(self.data1,self.data2)
            rg.process(self.rand1,self.data2)
            corrs=[ng,rg]
            
            if self.var_method =="shot":
                xi = self.combine_pairs_DS(corrs)
                err = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm)) 
            else:
                NG,wNG = self.get_rp_pairs([ng])
                RG,wRG = self.get_rp_pairs([rg])
                xi,cov = self.combine_jack_pairs(NG,RG,wNG,wRG)
                self.cov = cov
                err = np.sqrt(np.diag(cov))
            
                
            rnorm = ng.rnom
            meanr = rg.meanr
            meanlogr =  rg.meanlogr
            
            
        elif self.z is not None and self.rand1 is not None:
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,
                                          max_sep=maxr,min_rpar=min_rpar,metric="Rperp",var_method=self.var_method)
            
            rg = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=minr,
                                          max_sep=maxr,min_rpar=min_rpar,metric="Rperp",var_method=self.var_method)
            
            ng.process(self.data1,self.data2)
            rg.process(self.rand1,self.data2)
            
            corrs=[ng,rg]
            if self.var_method =="shot":
                xi = self.combine_pairs_DS(corrs)
                err = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm)) 
            else:
                NG,wNG = self.get_rp_pairs([ng])
                RG,wRG = self.get_rp_pairs([rg])
                xi,cov = self.combine_jack_pairs(NG,RG,wNG,wRG)
                self.cov = cov
                err = np.sqrt(np.diag(cov))
            
        rnorm = ng.rnom
        meanr = rg.meanr
        meanlogr =  rg.meanlogr

        return rg.rnom,meanr,meanlogr,xi,err


    def wgp(self,min_sep,max_sep,nbins,pimax,dpi):
        
        self.compute_norm()
        npt = int(2.*pimax/dpi) + 1.
        """Treecorr provides 2D counts computation but only for linear bins here we compute xi(rp,pi) by looping over radial distance instead"""
        pi = np.linspace(-pimax,pimax,npt)
        self.pi = pi
        self.nbins = nbins
        self.dpi = pi[1] - pi[0]

        dictNG = {}
        dictRG = {}
        dictRR = {}
        
        for i in range(0,len(pi)-1):
            pi_min = pi[i]
            pi_max = pi[i+1]
            dictNG[i] = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",var_method=self.var_method)
            dictRG[i] = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",var_method=self.var_method)
            dictRR[i] = treecorr.NNCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",var_method=self.var_method)

            dictNG[i].process(self.data1,self.data2)
            dictRG[i].process(self.rand1,self.data2)
            if self.rand2 is None:
                dictRR[i].process(self.rand1,self.rand1)
            else :
                dictRR[i].process(self.rand1,self.rand2)
        catNG = list(dictNG.values())
        catRG =  list(dictRG.values())
        catRR = list(dictRR.values())
        corrs = catNG + catRG + catRR
        
        rg = treecorr.NGCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,min_rpar=pi_min,max_rpar=pi_max,metric="Rperp")
        rg.process(self.rand1,self.data2)
        meanr = rg.meanr
        meanlogr =  rg.meanlogr
        
        if self.var_method =="shot":
            xi = self.combine_pairs_RS_proj(corrs)
            err = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm))    
        else:
            NG,wNG = self.get_rppi_pairs(catNG)
            RG,wRG = self.get_rppi_pairs(catRG)
            RR,wRR = self.get_rppi_pairs(catRR,rand=True)
            xi,cov = self.combine_jack_pairs_rppi(NG,RG,wNG,wRG,wRR)
            self.cov = cov
            err = np.sqrt(np.diag(cov))

        return dictNG[0].rnom,meanr,meanlogr,xi,err

    def get_cov(self):
        return self.cov
    
    
    
    def wgg(self,min_sep,max_sep,nbins,pimax,dpi):
        
        self.compute_norm()
        npt = int(pimax/dpi) + 1
        """Treecorr provides 2D counts computation but only for linear bins here we compute xi(rp,pi) by looping over radial distance instead"""
        pi = np.linspace(0,pimax,npt)
        self.pi = pi
        self.nbins = nbins
        self.dpi = pi[1] - pi[0]

        dictNG = {}
        dictRG = {}
        dictRR = {}
        
        for i in range(0,len(pi)-1):
            pi_min = pi[i]
            pi_max = pi[i+1]
            dictNG[i] = treecorr.NNCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",var_method=self.var_method)
            dictRG[i] = treecorr.NNCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",var_method=self.var_method)
            dictRR[i] = treecorr.NNCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",var_method=self.var_method)

            dictNG[i].process(self.data1,self.data2)
            dictRG[i].process(self.rand1,self.data2)
            if self.rand2 is None:
                dictRR[i].process(self.rand1,self.rand1)
            else :
                dictRR[i].process(self.rand1,self.rand2)
        catNG = list(dictNG.values())
        catRG =  list(dictRG.values())
        catRR = list(dictRR.values())
        corrs = catNG + catRG + catRR
        
        rg = treecorr.NNCorrelation(bin_type='Log',nbins=nbins,min_sep=min_sep,max_sep=max_sep,min_rpar=pi_min,max_rpar=pi_max,metric="Rperp")
        rg.process(self.rand1,self.data2)
        meanr = rg.meanr
        meanlogr =  rg.meanlogr
        
        if self.var_method =="shot":
            xi = self.combine_pairs_RS_proj_clustering(corrs)
            err = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm))    
        else:
            NG = self.get_rppi_pairs_clustering(catNG)
            RG = self.get_rppi_pairs_clustering(catRG)
            RR = self.get_rppi_pairs_clustering(catRR)
            xi,cov = self.combine_jack_pairs_rppi_clustering(NG,RG,RR)
            self.cov = cov
            err = np.sqrt(np.diag(cov))

        return dictNG[0].rnom,meanr,meanlogr,xi,err

    def get_cov(self):
        return self.cov