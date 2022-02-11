#wget https://www.repository.cam.ac.uk/bitstream/handle/1810/307452/Carbon_GAP_20.tgz
#tar -xzvf Carbon_GAP_20.tgz

from ase.io import iread
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.spatial import distance_matrix
import numpy as np
import dgl
from ase.calculators.mopac import MOPAC
from ase import Atoms

class MyDataset(Dataset):
    def __init__(self):
        self.xyz = []
        self.E = []
        for mol in iread('Carbon_GAP_20/Carbon_GAP_20_Training_Set.xyz'):
            self.xyz.append(mol.get_positions())
            self.E.append(mol.get_potential_energy())

    def __len__(self):
        return len(self.E)

    def __getitem__(self, idx):
        return self.xyz[idx], self.E[idx]

def nearest_neighbors(g, m, k):
    '''
        g --> (3) one coordinate used as reference point
        m --> (x,3) whole molecule geometry
        k --> (1) number of nearest neighbors to be found
        - assumes g is in m so the first closest neighbor is excluded
        - calculates the actual neighbors, the distances, and their indices in the list of atoms
    '''
    if k >= len(m):
        #print("Error: there are not enough points for desired number of neighbors.")
        #print("Choose a different number less than "+str(len(m))+".")
        #return None
        k = len(m)-1
    dist = distance_matrix([g], m)
    if len(m)==1:
    # if single atom, itself will be its neighbor
        k = 1
        indices = np.argpartition(dist[0], range(k))[:k] 
    else:
        indices = np.argpartition(dist[0], range(k+1))[1:k+1] # excludes first closest neighbor (itself)
    k_nearest = []
    k_dist = []
    for idx in indices:
        k_nearest.append(m[idx])
        k_dist.append(dist[0][idx])
    return k_nearest, k_dist, indices

def xyz_to_graph(molecule, k, node_featurizer, edge_featurizer):
    '''
        molecule --> (x,3) whole molecule geometry
        k --> (1) number of nearest neighbors to be found
        - creates a graph of the molecule where each atom is connected to its k nearest neighbors
        - featurizes the nodes with the energy?? and the edges with distance
    '''    
    #c = Atoms('C', positions=[[0, 0, 0]])
    #print("atoms")
    #c.calc = MOPAC(label='C', task='PM7 1SCF UHF')
    #print("mopac")
    #energy = c.get_potential_energy()
    #print("energy")

    src = []
    dest = []
    ndist = []
    #c_e = []
    for atom in range(len(molecule)):
        nbhd, dist, idx = nearest_neighbors(molecule[atom], molecule, k)
        #c_e.append(energy) 
        for i in range(len(nbhd)):
            src.append(atom)
            dest.append(idx[i])
            ndist.append(dist[i])
    g = dgl.graph((torch.tensor(src), torch.tensor(dest)))
    
    if node_featurizer is True:
        print("no node_featurizer")
        #g.ndata.update({'energy': torch.tensor(c_e)})

    if edge_featurizer is True:
        g.edata.update({'length': torch.tensor(ndist)})
    
    return g

def main():
    dataset = MyDataset()
    dataloader = DataLoader(dataset)

    print(len(dataset)) # total number of molecules
    print(len(dataset[0])) # matrix of both positions and energy
    print(len(dataset[0][0])) # one molecule's geometry/positions
    print(dataset[0][1]) # one molecule's potential energy
    
    test = dataset[3][0]
    print("singular point:",dataset[4782], "eV")
    #print(test)

    print(nearest_neighbors(test[0], test, 5))
    
    # Dummy Test
    # g1 = [0,1,0]
    # g2 = [[0,4,0],[0,1,0],[0,6,0],[0,3,0],[0,2,0],[0,5,0]]
    # nearest_neighbors(g1,g2,3)

    g = xyz_to_graph(test, 3, node_featurizer=False, edge_featurizer=True)
    print(g)
    print(g.edata)
    print(g.ndata)

if __name__ == "__main__":
    main()
