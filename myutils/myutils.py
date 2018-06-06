#!/usr/bin/env python
class data:
    """
    Data class
    """
    def __init__(self,name,binary=True):
        from numpy import fromfile,loadtxt
        try:
            if binary is True:
                self.data=fromfile(name,dtype='float64')
            else:
                self.data=loadtxt(name)
        except FileNotFoundError:
            print('File "%s" not found'%name)
            self.data=[]
            pass

    def __str__(self):
        return self.data.__str__()

    def hist(self):
        """
        Get histogram
        """
        from numpy import histogram
        self.n,self.b=histogram(self.data)
        self.size=self.b[1:]-self.b[:-1]
        self.m=self.b[:-1]+self.size*0.5
        return self.m,self.n

    def fitgauss(self):
        """
        Fit gaussian to the histogram
        """
        from numpy import sqrt,zeros
        from scipy.optimize import curve_fit
        p0=[self.data.mean(),sqrt(self.data.var())]
        coeff,var_matrix=curve_fit(self.gauss,self.m,self.n,p0=p0)
        self.gauss=self.gauss(self.m,*coeff)
        self.mu,self.sigma=coeff

    @staticmethod
    def gauss(x,*p):
        """
        Gaussian
        """
        from numpy import pi,exp,sqrt
        mu,sigma=p
        return exp(-(x-mu)**2.0/(2.0*sigma**2))/(sigma*sqrt(2*pi))

class configuration():
    """
    Configuration class
    """
    def __init__(self,files,delimiter=':'):
        """
        Initialize dictionary
        """
        self.conf={}
        for name in files:
            d={}
            try:
                for line in open(name,"r"):
                    t=line.split('#',1)[0]
                    s=t.find(delimiter)
                    if s is -1:
                        s=t.find(' ')
                    a,b=line[:s],line[s+1:]
                    t=b.split()
                    if s is not -1:
                        d[a]=t
                self.conf[name]=d
            except FileNotFoundError:
                print('File "%s" not found'%name)
                pass

    def __str__(self):
        """
        Print configurations
        """
        return self.conf

    def __get_values(self,key,value,l):
        if isinstance(value,dict):
            k=value.get(key,None)
            if k is not None:
                l+=[value.get(key,None)]
            for k,v in value.items():
                self.__get_values(key,v,l)

    def get_values(self,key,d=None):
        """
        Get list of key values
        """
        w=[]
        if d is None:
            d=self.conf
        self.__get_values(key,d,w)
        return w

    @staticmethod
    def order_list(l,key):
        return sorted(l,key=key)

    def __get_pair(self,d,key,hole,value):
        if isinstance(d,dict):
            v=d.get(key)
            if v:
                if v == value:
                    yield d.get(hole)
            for k,v in d.items():
                for x in self.__get_pair(v,key,hole,value):
                    yield x

    def get_pair(self,d,key,hole,value):
        """
        Get pairs for two keys and a value
        """
        w=[]
        for i in self.__get_pair(d,key,hole,value):
            w+=[i]
        return w

    def filter_dict(self,d,filter_dict):
        """
        Filter dictionary // FIXME generalization for higher order nested lists
        """
        w={}
        for k,v in d.items():
            if filter_dict.items() <= v.items():
                w[k]=v
        return w

    def get_all_pairs(self,key,hole,sorted_key=lambda x: float(x[0]),filter_dict=None):
        """
        Get ordered pairs for two corresponding keys
        """
        if filter_dict:
            d=self.filter_dict(self.conf,filter_dict)
        else:
            d=self.conf

        l=self.get_values(key,d=d)
        l=self.order_list(l,key=sorted_key)

        w=[]
        for v in l:
            w+=self.get_pair(d,key,hole,v)
        return l,w

    def data(self,c):
        """
        Loads data
        """
        from numpy import fromfile
        for d in self.conf.values():
            name=''.join(d.get('name')+[c])
            d[c]=data(name)

    def data_key(self,c,binary=False):
        """
        Loads data using self path
        """
        for d,b in self.conf.items():
            name=''.join([d[:d.rfind('/')+1]]+[c])
            b[c]=data(name,binary=binary)
