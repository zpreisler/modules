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

        self.attr=['en','mu','rho','epsilon']

        """
        Input files [files can be e.g. '*.conf']
        """
        self.files=glob(files)
        self.__get_attr__()

        self.save_h5py()

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
        p=fromfile(name+a)
        q=getattr(self,attr)
        setattr(self,attr,append(q,p))

    def save_h5py(self):
        import h5py
        self.h5=h5py.File('data.h5','w')
        for a in self.attr:
            self.h5.create_dataset(a,data=self.__name_attr__(a))

    @staticmethod
    def __name_attr__(a):
        return '_collective_'+a
