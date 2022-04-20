import torch
import torch.nn as nn
import numpy as np
import dgl
from ase import Atoms
from ase.io import iread
from ase.io import read

from torch.utils.data import Dataset
from dgl.dataloading import DataLoader

from tqdm import tqdm as tqdm
from sklearn.dummy import DummyRegressor
import time

import wandb

wandb.login()

# Train
print("Training...")
train_e = []
train_xyz = []
for mol in iread('../Carbon_GAP_20/Carbon_GAP_20_Training_Set.xyz'):
    train_e.append(mol.get_potential_energy())  
train_xyz = torch.full((len(train_e),),0)

# Dunn
print("Dunn...")
dunn_e = []
dunn_xyz = []
t = open("../dunn/filenames.txt", "r")
test_files = t.read().split()
for fn in test_files:
    with open('../dunn/' + fn, 'r') as f:
        data = f.read()
    n_data = data.replace("Energy", "energy")
    with open('../dunn/' + fn, 'w') as f:
        f.write(n_data)
    mol = read('../dunn/' + fn)
    dunn_e.append(mol.get_potential_energy())
    with open('../dunn/' + fn, 'w') as f:
        f.write(data)   
dunn_xyz = torch.full((len(dunn_e),),0)    
                                                  
# Modified Total
print("Modified Total...")
total_xyz = []
full_e = []
total_e= []
err = 0
for mol in iread('../Carbon_GAP_20/Carbon_Data_Set_Total.xyz'):
    full_e.append(mol.get_potential_energy())
#print("Length of Original Total Dataset:", len(full_e))

for mol in iread('../Carbon_GAP_20/Carbon_GAP_20_Training_Set.xyz'):
    try:
        idx = full_e.index(mol.get_potential_energy())
        full_e.pop(idx)
    except:
        err += 1
        continue
#print("Length of Modified Total Dataset:", len(full_e))
#print("Num of not matched molecules from training set:", err)

for i in range(0, 100*5, 5):
    total_e.append(full_e[i])
        
total_xyz = torch.full((len(total_e),),0)
#print("Length of final test dataset:", len(total_xyz))                  

# Dummy
dummy = DummyRegressor()

dummy.fit(train_xyz, train_e)
pred = dummy.predict(dunn_xyz)                                                                     
loss = ((pred - dunn_e)**2).sum()
print("\nDummy MSE:")
print(loss/len(pred))

dummy.fit(train_xyz, train_e)
pred = dummy.predict(total_xyz)
loss = ((pred - total_e)**2).sum()
print(loss/len(pred))
