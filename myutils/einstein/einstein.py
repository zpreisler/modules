#!/usr/bin/env python
from myutils import configuration
class data:
    def __init__(self,b):
        from numpy import array
        self.mean=array([(x.data.mean()) for x in b])
        self.var=array([(x.data.var()) for x in b])
        self.min=array([(x.data.min()) for x in b])
        self.max=array([(x.data.max()) for x in b])

class data2:
    def __init__(self,a,b):
        from numpy import array
        self.mean=array([(y.data*(x)).mean() for x,y in zip(a,b)])
        self.var=array([(y.data*(x)).var() for x,y in zip(a,b)])
        self.min=array([(y.data*(x)).min() for x,y in zip(a,b)])
        self.max=array([(y.data*(x)).max() for x,y in zip(a,b)])

class einstein:
    """
    Frenkel-Ladd free energy calculation
    """
    def __init__(self,files):
        from numpy import array,prod
        self.conf=configuration(files)
        self.input=configuration(['frenkel-ladd_gauss.input'])

        self.c=float(*self.input.get_values('c')[0])
        self.lmax=float(*self.input.get_values('lmax')[0])
        self.n=int(*self.input.get_values('n')[0])

        self.conf.data('.en')
        self.conf.data('.ein')
        self.conf.data('.ein_rot')
    
        a,b=self.conf.get_all_pairs('lambda','.en') 
        self.l=array([float(x[0]) for x in a])
        self.en=data(b)

        a,b=self.conf.get_all_pairs('lambda','.ein') 
        self.ein=data(b)
        self.ein2=data2(self.l,b)

        a,b=self.conf.get_all_pairs('lambda','.ein_rot') 
        self.ein_rot=data(b)
        self.ein_rot2=data2(self.l,b)

        self.trans=self.ein2.mean
        self.rot=self.ein_rot2.mean

        self.N=float(*self.conf.get_values('specie')[0])
        self.beta=float(*self.conf.get_values('epsilon')[0])
        b=array(self.conf.get_values('box'),dtype='float64')[0]
        self.V=b[0]*b[1]
        self.energy=self.en.mean.mean()

    def gauss_integrate(self,attr):
        """
        Gaussian integration
        """
        from numpy import log
        from numpy.polynomial.legendre import leggauss
        a,b=log(self.c),log(self.lmax)
        scale=b-a
        t=0.0
        y=getattr(self,attr)
        d=len(y)
        x,w=leggauss(d)
        for i,v in enumerate(y):
            t+=v*w[i]
        fe=-t*scale*0.5
        return fe

    def einstein_solid_fe(self):
        """
        Einstein solid free energy
        """
        from numpy import pi,log,prod
        N,beta,V,lmax=self.N,self.beta,self.V,self.lmax
        return -1.0/N*((N-1.0)*log(pi/(beta*lmax))+log(V))

    def rot_fe(self):
        """
        Rotational free energy
        """
        from numpy import log,exp
        beta,lmax=self.beta,self.lmax
        return -log((1.0-exp(-beta*lmax))/(beta*lmax))

    def energy_fe(self):
        """
        Energy contribution
        """
        return -self.beta*self.energy

    def plot(self,attr,label=r"$\left<\lambda H\right>$"):
        from matplotlib.pyplot import plot,semilogx,figure,xlabel,ylabel,subplots_adjust,errorbar
        from numpy import sqrt
        y=getattr(self,attr)

        semilogx(self.l,y.min,"k-",alpha=0.2,markersize=4,linewidth=1)
        semilogx(self.l,y.max,"k-",alpha=0.2,markersize=4,linewidth=1)
        errorbar(self.l,y.mean,yerr=sqrt(y.var),fmt='.',markersize=3,alpha=0.5)
        xlabel(r"\lambda")
        ylabel(label)
        subplots_adjust(left=0.19,bottom=0.18)
