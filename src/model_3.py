from ase.io import iread
import torch
import torch.nn as nn
import numpy as np
import dgl

from dgllife.model.gnn.mpnn import MPNNGNN
from dgllife.model.readout.mlp_readout import MLPNodeReadout

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
        
        self.readout = MLPNodeReadout(node_feats=node_out_feats, hidden_feats= 1000, graph_feats=1, activation=nn.ELU())

        self.predict = nn.Sequential(
            nn.Linear(node_out_feats, node_out_feats), 
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(node_out_feats, n_tasks)
        )
        
    def forward(self, g, nodes, edges):
        node_feats = self.gnn(g, nodes, edges)
        graph_feats = self.readout(g, node_feats)
        return graph_feats 
        #return self.predict(graph_feats)
