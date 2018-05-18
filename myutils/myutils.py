#!/usr/bin/env python
class data:
    """
    Data class
    """
    def __init__(self,name):
        from numpy import fromfile
        try:
            self.data=fromfile(name,dtype='float64')
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
    def __init__(self,files):
        """
        Initialize dictionary
        """
        self.conf={}
        for name in files:
            d={}
            try:
                for line in open(name,"r"):
                    t=line.split('#',1)[0]
                    s=t.find(':')
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
            l+=value.get(key,[])
            for k,v in value.items():
                self.__get_values(key,v,l)

    def get_values(self,key):
        """
        Get list of key values
        """
        w=[]
        self.__get_values(key,self.conf,w)
        return w

    @staticmethod
    def order_list(l,f=float):
        return sorted(l,key=lambda x: f(x))

    def __get_pair(self,d,key,hole,value):
        if isinstance(d,dict):
            v=d.get(key)
            if v:
                if v == [value]:
                    yield d.get(hole)
            for k,v in d.items():
                for x in self.__get_pair(v,key,hole,value):
                    yield x

    def get_pair(self,key,hole,value):
        """
        Get pairs for two keys and a value
        """
        return [i for i in self.__get_pair(self.conf,key,hole,value)]

    def get_all_pairs(self,key,hole):
        """
        Get ordered pairs for two corresponding keys
        """
        l=self.get_values(key)
        l=self.order_list(l)
        w=[]
        for v in l:
            w+=self.get_pair(key,hole,v)
        return l,w

    def data(self,c):
        """
        Loads data
        """
        from numpy import fromfile
        for d in self.conf.values():
            name=''.join(d.get('name')+[c])
            d[c]=data(name)
