import healpy as hp    
import numpy as np
from numpy import rad2deg
from numpy.random import default_rng
rng = default_rng()
from scipy.interpolate import CubicSpline as CS
from scipy.interpolate import interp1d
from scipy import stats
from astropy.cosmology import FlatLambdaCDM


cosmo = FlatLambdaCDM(Om0=0.31,H0=100)

"""Code to create random catalog for simple geometry."""


class survey_geometry:

    def __init__(self,NSIDE,RA,DEC):

        self.sphere_area = 41253
        self.ra_rand = None

        self.NSIDE = NSIDE
        self.ra = RA
        self.dec = DEC

        ra2 = self.convert_ra(self.ra)
        self.ra_min = np.min(ra2)
        self.ra_max = np.max(ra2)
        self.dec_min = np.min(self.dec)
        self.dec_max = np.max(self.dec)
        self.maskdata = self.mask()

    
    def convert_ra(self,ra):
        ra2 = np.copy(ra)
        cond = ra2 > 180.0
        ra2[cond] = ra2[cond] - 360.0
        return ra2
    
    def convert_ra_back(self,ra):
        ra2 = np.copy(ra)
        cond = ra2 < 0.
        ra2[cond] = ra2[cond] + 360.0
        return ra2
    

    def set_bins(self):
        if self.isredshift == True:
            dx = 0.005
            npt = int((self.xmax - self.xmin)/dx)+ 1
        else :
            dx = 10
            npt = int((self.xmax - self.xmin)/dx) + 1
        self.xbins = np.linspace(self.xmin,self.xmax,npt)

    def set_target_nz(self,**qwargs):
        if 'Z' in qwargs:
            self.xarray = qwargs['Z']
            self.isredshift = True
        else :
            self.xarray = qwargs['Dc']
            self.isredshift = False
        
        weights = qwargs['W']
        
        if 'xmin' in qwargs:  
            self.xmin = qwargs['xmin']
            self.xmax = qwargs['xmax']
        else :
            self.xmin = np.min(self.xarray)
            self.xmax = np.max(self.xarray)
            
        cond = np.where((self.xarray > self.xmin) & (self.xarray < self.xmax))
        self.xarray = self.xarray[cond]
        weights = weights[cond]
        self.set_bins()
        self.dn_data,self.edges = np.histogram(self.xarray, bins=self.xbins,weights=weights)
        self.centerbin = (self.edges[1:] + self.edges[:-1])/2.
        self.dn_data = self.dn_data/np.sum(self.dn_data)
        self.redshift_CDF()

        
    def compute_nz(self):
        dist = cosmo.comoving_distance(self.centerbin).value
        edges_r = cosmo.comoving_distance(self.edges).value
        dr = np.diff(edges_r)
        self.n_z = self.dn_data/(4*np.pi*dist**2*dr*(self.area/self.sphere_area))
        return self.centerbin,self.edges,self.dn_data,self.n_z
        

    """Cumalive distribution"""

    def redshift_CDF(self):
        myPDF = self.dn_data/np.sum(self.dn_data)
        dxc = np.diff(self.xbins);   xc = self.xbins[0:-1] + 0.5*dxc
        self.myCDF = np.zeros(len(self.xbins))
        self.myCDF[1:] = np.cumsum(myPDF)
        self.spline_inv = interp1d(self.myCDF,self.xbins,kind='linear')
        ##self.spline_inv = CS(myCDF,self.xbins)

  
    """Subsample mock to get the same n(z) distribution as the target"""
    def set_mock_nz(self,ra,dec,dc,w):
        ra2 = []
        dec2 = []
        dc2 = []
        w2 = []
        cond = np.where((dc > self.xmin) & (dc < self.xmax))
        ra = ra[cond]
        dec = dec[cond]
        dc = dc[cond]
        w = w[cond]
 
        dn_data2 = self.dn_data
        self.dn_mock,edges,binind= stats.binned_statistic(dc,w,bins=self.xbins,statistic='sum')
        self.dn_mock = self.dn_mock/np.sum(self.dn_mock)
        data = np.column_stack([ra,dec,dc,w,binind])
        data = data[data[:, 4].argsort()]
        databinned = np.split(data, np.unique(data[:,4], return_index = True)[1])[1:]

        div = self.dn_mock/dn_data2
        minh = np.min(div)
        div = div/minh
        

        for i in range(0,len(databinned)):
            databin = databinned[i]
            factor = 1./div[i]
            indices = np.arange(0,len(databin[:,0]))
            size = len(databin[:,0])
            sub = np.random.choice(indices,size=int(factor*size))
            databin = databin[sub]
            ra2.append(databin[:,0])
            dec2.append(databin[:,1])
            dc2.append(databin[:,2])
            w2.append(databin[:,3])
        ra2 = np.hstack(ra2)
        dec2 = np.hstack(dec2)
        dc2 = np.hstack(dc2)
        w2 = np.hstack(w2)
        return ra2,dec2,dc2,w2
              

    def DeclRaToIndex(self,ra,dec):
        return hp.pixelfunc.ang2pix(self.NSIDE,(-dec+90.)*np.pi/180.,ra*np.pi/180.)

    def IndexToDeclRa(self,index):
        theta,phi=hp.pixelfunc.pix2ang(self.NSIDE,index)
        return (180./np.pi*phi,-(180./np.pi*theta-90))


    def radec2thphi(self,ra,dec):
        return (-dec+90.)*np.pi/180.,ra*np.pi/180.

    def thphi2radec(self,theta,phi):
        return 180./np.pi*phi,-(180./np.pi*theta-90)

    def mask(self):
        npix = hp.nside2npix(self.NSIDE)
        p = np.zeros(npix)
        d = 0 
        index = self.DeclRaToIndex(self.ra,self.dec)
        self.index = index
        footprint = np.unique(index)
        self.footprint = footprint
        p[footprint] =+1
        self.frac = np.sum(p)/len(p)
        self.area = self.frac*self.sphere_area
        return p

    def get_mask(self):
        return self.maskdata

    def wmask(self):
        npix = hp.nside2npix(self.NSIDE)
        pixl = np.zeros(npix)
        tot = np.zeros(npix)
        index = self.DeclRaToIndex()
        for i in range(0,len(index)):
            pixl[index[i]] +=  1.*self.w[i]
            tot[index[i]] += 1.
        avg_weight = pixl/tot
        avg_weight[np.isnan(avg_weight)] = 0
        return avg_weight
 

    def create_random(self,size_random,shuf=False):

        size_random = int(size_random/self.frac)
        ra_rand = 360*rng.random(size_random)
        sindec_rand = 2*rng.random(size_random)-1.
        dec_rand = rad2deg(np.arcsin(sindec_rand))
        ra_rand,dec_rand = self.infootprint(ra_rand,dec_rand)
        ra_rand2 = self.convert_ra(ra_rand)
        cond = np.where((ra_rand2 >= self.ra_min) & (ra_rand2 <= self.ra_max) & (dec_rand >= self.dec_min) & \
                        (dec_rand <= self.dec_max))
        ra_rand = ra_rand[cond]
        dec_rand = dec_rand[cond]        
        #indices = np.random.choice(np.arange(0,len(ra_rand)),size=int(len(ra_rand)/10))
        #ra_rand = ra_rand[indices]
        #dec_rand = dec_rand[indices]
        random_numbers = rng.uniform(0,1,size=len(ra_rand))
        if shuf is False:
            z_rand = self.spline_inv(random_numbers)
        else:
            z_rand = np.random.choice(self.xarray,size=len(ra_rand),replace=True)
        w_rand = np.ones(len(z_rand))
        
        return ra_rand,dec_rand,z_rand,w_rand

    
    def create_random_angular_easy(self,size_random):
        ra2 = self.convert_ra(self.ra)
        ra_rand = rng.uniform(min(ra2), max(ra2), size_random)
        sindec_rand = rng.uniform(np.sin(min(self.dec*np.pi/180)), np.sin(max(self.dec*np.pi/180)), size_random)
        self.dec_rand = rad2deg(np.arcsin(sindec_rand))
        self.ra_rand = self.convert_ra_back(ra_rand)        
    
    """Work only for simplest rectangular footprint without masks"""
    def create_random_easy(self,size_random,shuf=False):
        if self.ra_rand is None:
            self.create_random_angular_easy(size_random)
        if shuf is False:
            random_numbers = rng.uniform(0,1,size=len(self.ra_rand))
            z_rand = self.spline_inv(random_numbers)
        else:
            z_rand = np.random.choice(self.xarray,size=len(self.ra_rand),replace=True)
        w_rand = np.ones(len(z_rand))
        
        return self.ra_rand,self.dec_rand,z_rand,w_rand
    
    def infootprint(self,ra,dec):
        index = self.DeclRaToIndex(ra,dec)
        cond = np.where(self.maskdata[index] !=0)
        ra2 = ra[cond]
        dec2 = dec[cond]
        return ra2,dec2

    @staticmethod
    def match(catalog1,catalog2):
        indices = np.where(catalog2.maskdata[catalog1.index] !=0)
        catalog1.ra = catalog1.ra[indices]
        catalog1.dec = catalog1.dec[indices]
        catalog1.z = catalog1.z[indices]
        catalog1.w = catalog1.w[indices]
        catalog1.g1 = catalog1.g1[indices]
        catalog1.g2 = catalog1.g2[indices]
        catalog1.mask()


        indices = np.where(catalog1.maskdata[catalog2.index] !=0)
        catalog2.ra = catalog2.ra[indices]
        catalog2.dec = catalog2.dec[indices]
        catalog2.z = catalog2.z[indices]
        catalog2.w = catalog2.w[indices]
        catalog2.g1 = catalog2.g1[indices]
        catalog2.g2 = catalog2.g2[indices]

        data1 = np.column_stack([catalog1.ra,catalog1.dec,catalog1.z,catalog1.w,catalog1.g1,catalog1.g2])
        data2 = np.column_stack([catalog2.ra,catalog2.dec,catalog2.z,catalog2.w,catalog2.g1,catalog2.g2])

        return data1,data2
    
    
        
