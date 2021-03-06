import torch
import torch.nn as nn
import numpy as np
import dgl
from ase.io import read
from ase.calculators.mopac import MOPAC
from ase import Atoms

from tqdm import tqdm as tqdm
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset
import time

from data_prep import *
from dunn_prep import *
from total_prep import *
from model import *

import wandb

wandb.login()

def main():
    # Parameters
    lr = 0.001
    mse = True
    mae = False
    test_data = 'dunn'

    knn = 4
    coul_mat = False
    model = 0
    message_pass = 3
    node_out_feats = 64
    edge_hidden_feats = 128
    n_tasks = 1

    # Initialize Model
    model = Model(node_in_feats=1,edge_in_feats=1, node_out_feats=node_out_feats, edge_hidden_feats=edge_hidden_feats, n_tasks=n_tasks, num_step_message_passing=message_pass)  
    
    # Loss Function
    if mse == True:
        loss = lambda a,b : ((b - a)**2).sum()
    elif mae == True:
        loss = lambda a,b : torch.abs(b - a).sum()

    best_score = None

    config_dict = dict(
        k_neighbors = knn,
        coul_mat = coul_mat,
        model = model,
        message_pass = message_pass,
        node_out_feats = node_out_feats,
        edge_hidden_feats = edge_hidden_feats,
        n_tasks = n_tasks,
        test_data = test_data,
        learn_rate = lr,
        mse = mse,
        mae = mae
    )
                                                                        
    #wandb.init(project="GinaHonors", entity="ginac",config=config_dict)
    
    # FOR TEST LOOP
    test_dataset = TotalGraphDataset()
    test_dataset.process(knn, coul_mat=coul_mat)
    testloader = GraphDataLoader(test_dataset,batch_size=len(test_dataset))
    
    # TEST
    model.eval()
    test_loss = 0.

    for batch_x, batch_y in testloader:
        atoms = batch_x.ndata['energy']
        edges = batch_x.edata['length']
        y_pred = model(batch_x, atoms, edges)
        error = loss(batch_y, y_pred.reshape(-1))
        test_loss += error.item()
                
    test_loss /= len(test_dataset)
            
    print("Test loss: ", test_loss)
    #wandb.log({'Epoch Num': epoch+1, 'Test loss': test_loss, 'Best Test Loss': best_score})



if __name__ == "__main__":
    main()

