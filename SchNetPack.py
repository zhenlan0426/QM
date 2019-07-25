#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:23:51 2019

@author: will
"""

#import schnetpack.atomistic.output_modules
import schnetpack as spk
import schnetpack.representation as rep
from schnetpack.datasets import QM9
import pandas as pd
import numpy as np
import pickle
from ase import Atoms

# load qm9 dataset and download if necessary
data = QM9("qm9.db", collect_triples=True)
loader = spk.data.AtomsLoader(data, batch_size=1, num_workers=2)
reps = rep.BehlerSFBlock()

# get wACSF feature
reps_dict = {}
for i,x in enumerate(loader):
    reps_dict[data.get_atoms(i).__repr__()] = reps(x).squeeze(0)
    

structures = pd.read_csv('/home/will/Desktop/kaggle/QM/Data/structures.csv')
structures_gb = structures.groupby(['molecule_name'])

atoms_structures = {}
for k,v in structures_gb:
    atom_dict = {'positions':v[['x','y','z']].values.tolist(),
                 'symbols':[i[0] for i in v[['atom']].values.tolist()]
                 }
    atom = Atoms(**atom_dict)
    atoms_structures[k] = atom.__repr__()

final_dict = {}
for k,v in atoms_structures.items():
    final_dict[k] = reps_dict[v].numpy().astype(np.float32)

with open('/home/will/Desktop/kaggle/QM/Data/structures_dict_wACSF.pickle', 'wb') as handle:
    pickle.dump(final_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)    