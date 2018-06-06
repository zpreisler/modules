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
        self.mean=array([(x.data.mean()) for x in b])
        self.var=array([(x.data.var()) for x in b])
        self.min=array([(x.data.min()) for x in b])
        self.max=array([(x.data.max()) for x in b])

class kmc:
    """
    Read KMC output
    """
    def __init__(self,files):
        from myutils import configuration
        from numpy import array
        self.conf=configuration(files)

        self.conf.data_key("entropy.dat")
