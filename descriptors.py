#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:21:55 2019

@author: will
"""

#import numpy as np
#from ase.build import molecule
from ase import Atoms
#from dscribe.descriptors import SOAP
#from dscribe.descriptors import CoulombMatrix



#
#
## Setup descriptors
#cm_desc = CoulombMatrix(n_atoms_max=3, permutation="sorted_l2")
#soap_desc = SOAP(species=["C", "H", "O", "N"], rcut=5, nmax=8, lmax=6, crossover=True)
#
## Create descriptors as numpy arrays or scipy sparse matrices
#water = samples[0]
#coulomb_matrix = cm_desc.create(water)
#soap = soap_desc.create(water, positions=[0])
#
## Easy to use also on multiple systems, can be parallelized across processes
#coulomb_matrices = cm_desc.create(samples)
#coulomb_matrices = cm_desc.create(samples, n_jobs=3)
#oxygen_indices = [np.where(x.get_atomic_numbers() == 8)[0] for x in samples]
#oxygen_soap = soap_desc.create(samples, oxygen_indices, n_jobs=3)
#
#


from dscribe.descriptors import ACSF

# Setting up the ACSF descriptor
acsf = ACSF(
    species=["C", "O"],
    rcut=6.0,
    g2_params=[[1, 1], [1, 2], [1, 3]],
    g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
)

d = 1.1
co = Atoms(['C','O'], positions=[(0, 0, 0), (0, 0, d)])
acsf_water = acsf.create(co)