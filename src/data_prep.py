from ase.io import iread
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.spatial import distance_matrix
import numpy as np
import dgl

class GraphDataset(Dataset):
    def __init__(self):
        self.xyz = []
        self.E = []
        self.graph = []
        for mol in iread('../Carbon_GAP_20/Carbon_GAP_20_Training_Set.xyz'):
            self.xyz.append(mol.get_positions())
            self.E.append(mol.get_potential_energy())
        #self.xyz = self.xyz[:10]
        #self.E = self.E[:10]

    def nearest_neighbors(self, g, m, k):
        '''
            g --> (3) one coordinate used as reference point
            m --> (x,3) whole molecule geometry
            k --> (1) number of nearest neighbors to be found
            - assumes g is in m so the first closest neighbor is excluded
            - takes the max amount of neighbors if k is greater than total atoms in a molecule
            - if a molecule is also a single atom, it will be its own neighbor
            - calculates the actual neighbors, the distances, and their indices in the list of atoms
        '''
        if k >= len(m):
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

    def featurize_atoms(self, molecule):
    # featurize atoms
        #c = Atoms('C', positions=[[0, 0, 0]])
        #c.calc = MOPAC(label='C', task='PM7 1SCF UHF')
        #energy = c.get_potential_energy()
        
        c_e = []
        for atom in range(len(molecule)):
            c_e.append(0)
        return {'energy': torch.tensor(c_e).reshape(-1,1).float()}
    
    def xyz_to_graph(self, molecule, k, node_featurizer, edge_featurizer):
        '''
            molecule --> (x,3) whole molecule geometry
            k --> (1) number of nearest neighbors to be found
            - creates a graph of the molecule where each atom is connected to its k nearest neighbors
            - featurizes the nodes with the energy and the edges with distance
        '''    
        src = []
        dest = []
        ndist = []
        for atom in range(len(molecule)):
            nbhd, dist, idx = self.nearest_neighbors(molecule[atom], molecule, k)
            for i in range(len(nbhd)):
                src.append(atom)
                dest.append(idx[i])
                ndist.append(dist[i])
        g = dgl.graph((torch.tensor(src), torch.tensor(dest)))

        if node_featurizer is not None:
            g.ndata.update(node_featurizer(molecule))

        if edge_featurizer is True:
            g.edata.update({'length': torch.tensor(ndist).reshape(-1,1).float()})

        return g
    
    def process(self, k):
    # make graph for each molecule
        print("Start processing dataset...")
        tmp = []
        counter=0
        for xyz in self.xyz:
            if counter%1000 == 0:
                print(counter)
            counter+=1
            tmp.append(self.xyz_to_graph(xyz, k, node_featurizer=self.featurize_atoms, edge_featurizer=True))
        self.graph = tmp
        print("Finished processing dataset\n")
         
    def __len__(self):
        return len(self.E)

    def __getitem__(self, idx):
        return self.graph[idx], self.E[idx]