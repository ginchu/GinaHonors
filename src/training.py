import torch
import torch.nn as nn
import numpy as np
import dgl
from ase.calculators.mopac import MOPAC
from ase import Atoms

from tqdm import tqdm as tqdm
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import split_dataset
import time

from data_prep import *
from model import *

import wandb

wandb.login()

def main():
    # Parameters
    data_split = 0.8
    batch_size = 16
    epochs = 1000 
    lr = 0.001

    knn = 3
    coul_mat=False

    config_dict = dict(
        k_neighbors = knn,
        coul_mat = coul_mat,
        data_split = data_split,
        batch_size = batch_size,
        epochs = epochs,
        learn_rate = lr
    )

    wandb.init(project="GinaHonors", entity="ginac",config=config_dict)

    # Creating Dataset
    graph_dataset = GraphDataset()
    graph_dataset.process(knn, coul_mat=coul_mat)
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
        #t0 = time.time()
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
        #t1 = time.time()
        #print(t1-t0)

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
            
            print("Test loss: ", test_loss)
            wandb.log({'Epoch Num': epoch+1, 'Test loss': test_loss, 'Best Test Loss': best_score})



if __name__ == "__main__":
    main()

