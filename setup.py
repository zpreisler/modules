#!/usr/bin/env python
from distutils.core import setup
setup(name='myutils',
        version='1.0',
        description='utilities',
        author='Zdenek Preisler',
        author_email='z.preisler@gmail.com',
        packages=['myutils','myutils.einstein','myutils.kmc',
            'tensorflow_utils',
            'mof_lattice'],
        scripts=['scripts/fe_einstein_solid.py','scripts/plot_mof.py']
        )
