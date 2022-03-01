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
    epochs = 3000 
    lr = 0.001
    mse = True
    mae = False

    knn = 3
    coul_mat=False
    message_pass = 1

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
    model = Model(1,1, num_step_message_passing=message_pass)  

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    # Loss Function
    if mse == True:
        loss = lambda a,b : ((b - a)**2).sum()
    elif mae == True:
        loss = lambda a,b : torch.abs(b - a).sum()

    best_score = None

    config_dict = dict(
        k_neighbors = knn,
        coul_mat = coul_mat,
        message_pass = message_pass,
        dataset_len = len(graph_dataset),
        data_split = data_split,
        batch_size = batch_size,
        epochs = epochs,
        learn_rate = lr
    )
                                                                        
    wandb.init(project="GinaHonors", entity="ginac",config=config_dict)
    
    print("Number of Epochs:", epochs, "\n")
    for epoch in tqdm(range(epochs)):
        # TRAIN
        t0 = time.time()
        model.train()
        running_loss = 0.
        for batch_x, batch_y in trainloader:
            optimizer.zero_grad()
            atoms = batch_x.ndata['energy']
            edges = batch_x.edata['length']
            y_pred = model(batch_x, atoms, edges)
            error = loss(batch_y, y_pred.reshape(-1))
            running_loss += error.item()
            error.backward()
            optimizer.step()
            
        running_loss /= len(trainloader)
        print("Train loss: ", running_loss)
        wandb.log({'Epoch Num': epoch+1, 'Train loss': running_loss})
        t1 = time.time()
        print(t1-t0)
        exit()

        # TEST
        if epoch%10 == 0:
            model.eval()
            test_loss = 0.
            for batch_x, batch_y in testloader:
                atoms = batch_x.ndata['energy']
                edges = batch_x.edata['length']
                y_pred = model(batch_x, atoms, edges)
                error = loss(batch_y, y_pred.reshape(-1))
                test_loss += error.item()
                
            test_loss /= len(testloader)
            
            if not best_score:
                best_score = test_loss
            if test_loss < best_score:
                best_score = test_loss

            print("Test loss: ", test_loss)
            wandb.log({'Epoch Num': epoch+1, 'Test loss': test_loss, 'Best Test Loss': best_score})



if __name__ == "__main__":
    main()

