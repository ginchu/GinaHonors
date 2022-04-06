from ase.io import read
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.spatial import distance_matrix
import numpy as np
import dgl

class DunnGraphDataset(Dataset):
    def __init__(self):
        self.xyz = []
        self.E = []
        self.graph = []
        t = open("../dunn/filenames.txt", "r")
        test_files = t.read().split()
        for fn in test_files:
            with open('../dunn/' + fn, 'r') as f:
                data = f.read()
            n_data = data.replace("Energy", "energy")
            with open('../dunn/' + fn, 'w') as f:
                f.write(n_data)
            mol = read('../dunn/' + fn)
            self.xyz.append(mol.get_positions())
            self.E.append(mol.get_potential_energy())
            with open('../dunn/' + fn, 'w') as f:
                f.write(data)                              
                                                                    
    def nearest_neighbors(self, g, m, k):
        '''
            g --> (3) one coordinate used as reference point
            m --> (x,3) matrix, whole molecule geometry
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
            k_dist.append(dist[0,idx])
        return k_nearest, k_dist, indices.tolist()

    def xyz_to_graph(self, molecule, k, node_featurizer, edge_featurizer, coul_mat):
        '''
            molecule --> (x,3) matrix, whole molecule geometry
            k --> (1) number of nearest neighbors to be found
            - creates a graph of the molecule where each atom is connected to its k nearest neighbors
            - featurizes the nodes with the energy and the edges with distance
        '''    
        src = []
        dest = []
        ndist = []
        for atom in range(len(molecule)):
            nbhd, dist, idx = self.nearest_neighbors(molecule[atom], molecule, k)
            if coul_mat != True:
                src+=[atom]*len(nbhd)
                dest+=idx
                ndist+=dist
            else:
                # Coulomb Matrix
                nbhd = [molecule[atom]] + nbhd
                idx = [atom] + idx
                coulomb = distance_matrix(nbhd, nbhd)
                for i in range(len(nbhd)-1):
                    src += [idx[i]]*(len(nbhd)-(i+1))
                    dest += idx[i+1:]
                    ndist += coulomb[i,i+1:].tolist()
        g = dgl.graph((torch.tensor(src), torch.tensor(dest)))

        if node_featurizer is True:
            c_e = []
            for atom in range(len(molecule)):
                c_e.append(0)
            g.ndata.update({'energy': torch.tensor(c_e).reshape(-1,1).float()})

        if edge_featurizer is True:
            g.edata.update({'length': torch.tensor(ndist).reshape(-1,1).float()})

        return g
    
    def process(self, k, coul_mat=False):
    # make graph for each molecule
        print("Start processing dataset...")
        tmp = []
        counter=0
        for xyz in self.xyz:
            if counter%1000 == 0:
                print(counter)
            counter+=1
            tmp.append(self.xyz_to_graph(xyz, k, node_featurizer=True, edge_featurizer=True, coul_mat=coul_mat))
        self.graph = tmp
        print("Finished processing dataset")

    def __len__(self):
        return len(self.E)

    def __getitem__(self, idx):
        return self.graph[idx], self.E[idx]
