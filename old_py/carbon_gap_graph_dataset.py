from ase.io import iread
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.spatial import distance_matrix
import numpy as np
import dgl
from ase.calculators.mopac import MOPAC
from ase import Atoms

from tqdm import tqdm as tqdm
from dgl.dataloading import GraphDataLoader
from dgllife.model.gnn.mpnn import MPNNGNN
from dgllife.model.readout.mlp_readout import MLPNodeReadout
from dgl.nn.pytorch.conv.cfconv import ShiftedSoftplus

import wandb

wandb.login()

class GraphDataset(Dataset):
    def __init__(self):
        self.xyz = []
        self.E = []
        self.graph = []
        for mol in iread('Carbon_GAP_20/Carbon_GAP_20_Training_Set.xyz'):
            self.xyz.append(mol.get_positions())
            self.E.append(mol.get_potential_energy())
        #self.xyz = self.xyz[:10]
        #self.E = self.E[:10]

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
            k_dist.append(dist[0][idx])
        return k_nearest, k_dist, indices

    def xyz_to_graph(self, molecule, k, node_featurizer, edge_featurizer):
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
            for i in range(len(nbhd)):
                src.append(atom)
                dest.append(idx[i])
                ndist.append(dist[i])
        g = dgl.graph((torch.tensor(src), torch.tensor(dest)))

        if node_featurizer is True:
            c_e = []
            for atom in range(len(molecule)):
                c_e.append(0)
            g.ndata.update({'energy': torch.tensor(c_e).reshape(-1,1).float()})

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
            tmp.append(self.xyz_to_graph(xyz, k, node_featurizer=True, edge_featurizer=True))
        self.graph = tmp
        print("Finished processing dataset\n")
         
    def __len__(self):
        return len(self.E)

    def __getitem__(self, idx):
        return self.graph[idx], self.E[idx]

class Model(nn.Module):
    def __init__(self, 
                 node_in_feats,
                 edge_in_feats,
                 node_out_feats=64,
                 edge_hidden_feats=128,
                 n_tasks=1,
                 num_step_message_passing=6):
        
        super(Model, self).__init__()
        self.gnn = MPNNGNN(node_in_feats=node_in_feats,
                           node_out_feats=node_out_feats,
                           edge_in_feats=edge_in_feats,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        self.readout = MLPNodeReadout(node_feats=node_out_feats, hidden_feats=edge_hidden_feats, graph_feats=node_out_feats)
        #self.readout = MLPNodeReadout(node_feats=node_out_feats, hidden_feats= 1000, graph_feats=1, activation=ShiftedSoftplus())

        self.predict = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats), 
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )
        
    def forward(self, g, nodes, edges):
        node_feats = self.gnn(g, nodes, edges)
        graph_feats = self.readout(g, node_feats)
        return self.predict(graph_feats) # graph_feats

def main():
    batch_size = 32
    epochs = 1000 
    lr = 0.001

    k_near = 3

    config_dict = dict(
        k_neighbors = k_near,
        batch_size = batch_size,
        epochs = epochs,
        learn_rate = lr
    )

    wandb.init(project="GinaHonors", entity="ginac",config=config_dict)

    graph_dataset = GraphDataset()
    graph_dataset.process(k_near)
    print("Length of dataset:",len(graph_dataset))
    
    dataloader = GraphDataLoader(graph_dataset,batch_size=batch_size)
    print("Batch size:", batch_size)
    print("Length of Dataloader or Num of Batches:", len(dataloader))
    model = Model(1,1) #, num_step_message_passing=3)  

    #for batch_x, batch_y in dataloader:
    #    print(batch_x, batch_y)

    print("Number of Epochs:", epochs, "\n")
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            atoms = batch_x.ndata['energy']
            edges = batch_x.edata['length']
            y_pred = model(batch_x, atoms, edges)
            mse = ((y_pred.reshape(-1) - batch_y)**2).sum()
            running_loss += mse.item()
            mse.backward()
            optimizer.step()
            
        running_loss /= len(dataloader)
        print("Train loss: ", running_loss)
        wandb.log({'Epoch Num': epoch+1, 'Train loss': running_loss})

if __name__ == "__main__":
    main()
