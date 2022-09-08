import pandas as pd
import numpy as np
from corr2 import lensingPCF
from astropy.table import Table


T = Table.read('/feynman/work/dap/lceg/rp269101/stuff/eBOSS_LRG_flagship.fits')
ra = T['ra']#.values
dec = T['dec']#.values
z = T['z']#.values
w =  T['w']#.values
e1  = T['g1']#.values
e2 = T['g2']#.values


T2 = Table.read('/feynman/work/dap/lceg/rp269101/stuff/eBOSSr_flagship.fits')
ra_rand = T2['ra']#.values
dec_rand = T2['dec']#.values
z_rand = T2['z']#.values
w_rand  =  T2['w']#.values

#corr = lensingPCF(Om0=0.319,RA=ra,DEC=dec,Z=z,W=w,\
#                  RA2=ra,DEC2=dec,Z2=z,W2=w,g1=e1,g2=e2,computation="IA",units="deg",npatch=1)

#corr.set_random(RA_r = ra_rand,DEC_r=dec_rand,Z_r=z_rand,W_r=w_rand)

#r,meanr,meanlogr,xi,err= corr.wgp(1e-1,100,15,100,10)
#np.savetxt("wgp_CMASS_flagshig_v7.dat",np.transpose([r,meanr,meanlogr,xi,err])) 
corr = lensingPCF(Om0=0.319,RA=ra,DEC=dec,Z=z,W=w,\
                  RA2=ra,DEC2=dec,Z2=z,W2=w,g1=e1,g2=e2,computation="IA",units="deg",npatch=200)

corr.set_random(RA_r = ra_rand,DEC_r=dec_rand,Z_r=z_rand,W_r=w_rand)

r,meanr,meanlogr,xi,err= corr.wgp(1e-1,100,21,100,10)
cov = corr.get_cov()
np.savetxt("wgp_LRG_flagship_jack_final_v7.dat",np.transpose([r,meanr,meanlogr,xi,err]))
np.savetxt("cov_wgp_LRG_flagship_jack_final_v7.dat",cov)
