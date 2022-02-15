import torch
import torch.nn as nn
import numpy as np
import dgl
from ase.calculators.mopac import MOPAC
from ase import Atoms

from tqdm import tqdm as tqdm
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset
from data_prep import *
from model import *

import wandb

wandb.login()

def main():
    # Parameters
    data_split = 0.8
    batch_size = 32
    epochs = 1000 
    lr = 0.001

    knn = 3
    atom_e=False
    config_dict = dict(
        k_neighbors = knn,
        atom_e = atom_e,
        data_split = data_split,
        batch_size = batch_size,
        epochs = epochs,
        learn_rate = lr
    )

    wandb.init(project="GinaHonors", entity="ginac",config=config_dict)

    # Creating Dataset
    graph_dataset = GraphDataset()
    graph_dataset.process(knn, atom_e=atom_e)
    print("Length of dataset:",len(graph_dataset))
    
    # Split and Batch Dataset
    train, val, test = split_dataset(graph_dataset,[data_split,0,(1-data_split)])
    trainloader = GraphDataLoader(train,batch_size=batch_size)
    testloader = GraphDataLoader(test,batch_size=batch_size)
    print("Batch size:", batch_size)
    #print("Length of Dataloader or Num of Batches:", len(dataloader))
    
    # Initialize Model
    model = Model(1,1) #, num_step_message_passing=3)  

    print("Number of Epochs:", epochs, "\n")
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    best_score = None

    for epoch in tqdm(range(epochs)):
        # TRAIN
        model.train()
        running_loss = 0.
        for batch_x, batch_y in trainloader:
            optimizer.zero_grad()
            atoms = batch_x.ndata['energy']
            edges = batch_x.edata['length']
            y_pred = model(batch_x, atoms, edges)
            mse = ((y_pred.reshape(-1) - batch_y)**2).sum()
            running_loss += mse.item()
            mse.backward()
            optimizer.step()
            
        running_loss /= len(trainloader)
        print("Train loss: ", running_loss)
        wandb.log({'Epoch Num': epoch+1, 'Train loss': running_loss})

        # TEST
        if epoch%10 == 0:
            model.eval()
            test_loss = 0.
            for batch_x, batch_y in testloader:
                atoms = batch_x.ndata['energy']
                edges = batch_x.edata['length']
                y_pred = model(batch_x, atoms, edges)
                mse = ((y_pred.reshape(-1) - batch_y)**2).sum()
                test_loss += mse.item()
                
            test_loss /= len(testloader)
            
            if not best_score:
                best_score = test_loss
                torch.save(model.state_dict(), 'best-model.pt')
            if test_loss < best_score:
                best_score = test_loss
                torch.save(model.state_dict(), 'best-model.pt')

            print("Test loss: ", test_loss)
            wandb.log({'Epoch Num': epoch+1, 'Test loss': test_loss, 'Best Test Loss': best_score})



if __name__ == "__main__":
    main()

