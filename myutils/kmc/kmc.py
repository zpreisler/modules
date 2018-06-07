#!/usr/bin/env python
def read_results(folder='sq_field_L_',d=[128,256,512,1024,2048,4096],n='results.dat',c=(1,2)):
    from numpy import loadtxt
    t=[[] for i in range(len(c))]
    for i,l in enumerate(d):
        f='sq_field_L_%d/%s'%(l,n)
        print("[reading: \"%s\"]"%f)
        w=loadtxt(f,usecols=c,unpack=True)
        for g,h in zip(t,w):
            g+=[h]
    return t

class data:
    def __init__(self,b):
        from numpy import array
        self.data=array([(x.data) for x in b])
        self.mean=array([(x.data.mean()) for x in b])
        self.var=array([(x.data.var()) for x in b])
        self.min=array([(x.data.min()) for x in b])
        self.max=array([(x.data.max()) for x in b])

class kmc:
    """
    Read KMC output
    """
    def __init__(self,files,delimiter='\t'):
        from myutils import configuration
        from numpy import array
        self.files=files
        self.conf=configuration(files,delimiter=delimiter)

        self.conf.data_key("entropy.dat")

        a,b=self.conf.get_all_pairs('squarewave','entropy.dat',sorted_key=lambda x: float(x[1]))
        self.amplitude=array([float(x[1]) for x in a])
        self.entropy=data(b)

        self.L=float(*self.conf.get_values('length')[0])

    def plot(self,attrx='amplitude',attry='entropy'):
        from matplotlib.pyplot import plot,semilogx,semilogy,figure,xlabel,ylabel,subplots_adjust,errorbar
        from numpy import sqrt,log
        x=getattr(self,attrx)
        y=getattr(self,attry)

        plot(x,y.min,"k-",alpha=0.2,markersize=4,linewidth=1)
        plot(x,y.max,"k-",alpha=0.2,markersize=4,linewidth=1)
        plot(x,y.mean,"-")
        errorbar(x,y.mean,yerr=sqrt(y.var),fmt='.',markersize=3,alpha=0.5)
        xlabel(r"Amplitude \Theta")
        ylabel(r"\sigma")
        subplots_adjust(left=0.19,bottom=0.18)
