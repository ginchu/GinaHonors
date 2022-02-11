import torch
import torch.nn as nn
import numpy as np
import dgl
from ase.calculators.mopac import MOPAC
from ase import Atoms

from tqdm import tqdm as tqdm
from dgl.dataloading import GraphDataLoader

from data_prep import *
from model import *

import wandb

wandb.login()

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

