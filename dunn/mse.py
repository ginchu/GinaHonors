import numpy as np
from ase.io import iread
from ase.io.xyz import read_xyz
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator

a = open("filenames.txt", "r")
aa = a.read().split()

t = open("target_e.txt", "r")
q = open("quip_e.txt", "r")

target = np.asarray([float(i[1:-1]) for i in t.read().split()])
pred = np.asarray([float(j) for j in q.read().split()])

print(aa[0])
ar = read(aa[0])
print(ar)
calc = SinglePointCalculator(ar)
#ar.set_calculator(calc)
print(ar)
print(ar.get_potential_energy())

for mol in iread('../Carbon_GAP_20/Carbon_GAP_20_Training_Set.xyz'):
    print(mol)
    exit()

loss = ((pred-target)**2).sum()

print(loss/len(pred))
