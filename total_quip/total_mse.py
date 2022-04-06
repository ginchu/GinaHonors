import os
import numpy as np
from ase.io import iread
from ase import Atoms

full_xyz = []
full_E = []
E = []
err = 0
for mol in iread('../Carbon_GAP_20/Carbon_Data_Set_Total.xyz'):
    full_xyz.append(mol)
    full_E.append(mol.get_potential_energy())
print("Length of Original Total Dataset:", len(full_xyz))
for mol in iread('../Carbon_GAP_20/Carbon_GAP_20_Training_Set.xyz'):
    try:
        idx = full_E.index(mol.get_potential_energy())
        full_xyz.pop(idx)
        full_E.pop(idx)
    except:
        err += 1
        continue
print("Length of Modified Total Dataset:", len(full_xyz))
print("Num of not matched molecules from training set:", err)
                                              
count=0                                              
for i in range(0, 100*5, 5):
    E.append(full_E[i])
    full_xyz[i].write("xyz/" + str(i) + ".xyz")
    count+=1                                                                
print("Length of final test dataset:", count)                  


for i in range(0, 100*5, 5):
    os.system("quip atoms_filename=xyz/" + str(i) + ".xyz param_filename=../Carbon_GAP_20/Carbon_GAP_20_potential/Carbon_GAP_20.xml E | grep -m1 'Energy=' | sed -n -e 's/^.*Energy=//p' >> quip_e.txt")

q = open("quip_e.txt", "r")

target = np.asarray(E)
pred = np.asarray([float(j) for j in q.read().split()])

loss = ((pred-target)**2).sum()

print(loss/len(pred))
