#!/usr/bin/env python
class hive:
    """
    Class for reading simulation data and generating input/output for tensorflow
    """
    def __init__(self,files,offset=0):
        from glob import glob
        from numpy import column_stack
        from pprint import pprint
        import h5py
        #from myutils import configuration
        """
        Data extensions 
        """

        self.attr=['en','mu','rho','epsilon','pressure']

        """
        Input files [files can be e.g. '*.conf']
        """
        self.files=glob(files)
        self.__get_attr__()

        self.__get_length__()
        #self.save_h5py()

    def __get_length__(self):
        for a in self.attr:
            attr=self.__name_attr__(a)
            q=getattr(self,attr)
            self.length=len(q)

    def __get_attr__(self):
        from numpy import fromfile,append
        self.__alloc_attr__()
        for f in self.files[:]:
            name=f[:f.rfind('.')+1]
            for a in self.attr:
                self.__read_attr__(name,a)

    def __alloc_attr__(self):
        for a in self.attr:
            attr=self.__name_attr__(a)
            setattr(self,attr,[])

    def __read_attr__(self,name,a):
        from numpy import fromfile,append
        attr=self.__name_attr__(a)
        p=fromfile(name+a,dtype='float32')
        q=getattr(self,attr)
        offset=int(len(p)/3)
        setattr(self,attr,append(q,p[offset:]))

    def save_h5py(self,name='data.h5'):
        import h5py
        self.h5=h5py.File(name,'w')
        for a in self.attr:
            self.h5.create_dataset(a,data=self.__name_attr__(a))

    @staticmethod
    def __name_attr__(a):
        return '_collective_'+a
