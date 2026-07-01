import numpy as np
import pandas as pd
import os

from model_bank.PCA import PCA
from model_bank.KPCA import KPCA
from model_bank.SFA import SFA
# from model_bank.ESFA import ESFA
from model_bank.SSA import analytic_ssa as SSA
from model_bank.CCA import CVA

from model_bank.CNN import convAE
from model_bank.AE import MLPAE
from model_bank.VAE import VAE
from model_bank.Transformer import Transformer
from model_bank.GRU import GRU

import argparse
from utils_tools import read_datasets_from_folder,seed_all
import torch

from collections import defaultdict

parser = argparse.ArgumentParser(description="MRS")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

seed_all(3407)

models_list = [
               PCA(0.99,0.75),PCA(0.99,0.8),PCA(0.99,0.85),PCA(0.99,0.9),PCA(0.99,0.95),\
                SFA(0.99,2),SFA(0.99,3),SFA(0.99,4),SFA(0.99,5),SFA(0.99,6),
                SSA(0.99,2),SSA(0.99,3),SSA(0.99,4),SSA(0.99,5),SSA(0.99,6),
                CVA(2,2,0.99),CVA(3,3,0.99),CVA(4,4,0.99),CVA(5,5,0.99),CVA(6,6,0.99),
                MLPAE(args,num_layers=2,growth_rate=0.8,lr=2e-4,epochs=50),
                MLPAE(args,num_layers=3,growth_rate=0.8,lr=2e-4,epochs=50),
                MLPAE(args,num_layers=4,growth_rate=0.8,lr=2e-4,epochs=50),
                MLPAE(args,num_layers=2,growth_rate=0.8,lr=2e-4,epochs=100),
                MLPAE(args,num_layers=3,growth_rate=0.8,lr=2e-4,epochs=100),
                MLPAE(args,num_layers=4,growth_rate=0.8,lr=2e-4,epochs=100),
                VAE(args,num_layers=2,growth_rate=0.8,lr=2e-4,epochs=50),
                VAE(args,num_layers=3,growth_rate=0.8,lr=2e-4,epochs=50),
                VAE(args,num_layers=4,growth_rate=0.8,lr=2e-4,epochs=50),
                VAE(args,num_layers=2,growth_rate=0.8,lr=2e-4,epochs=100),
                VAE(args,num_layers=3,growth_rate=0.8,lr=2e-4,epochs=100),
                VAE(args,num_layers=4,growth_rate=0.8,lr=2e-4,epochs=100),
                convAE(args,conv_depth=1,lr=2e-4,epochs=50),
                convAE(args,conv_depth=2,lr=2e-4,epochs=50),
                convAE(args,conv_depth=3,lr=2e-4,epochs=50),
                convAE(args,conv_depth=1,lr=2e-4,epochs=100),
                convAE(args,conv_depth=2,lr=2e-4,epochs=100),
                convAE(args,conv_depth=3,lr=2e-4,epochs=100),
                GRU(args,latent_dim=8,layers=3,epochs=50),
                GRU(args,latent_dim=16,layers=3,epochs=50),
                GRU(args,latent_dim=32,layers=3,epochs=50),
                GRU(args,latent_dim=8,layers=3,epochs=100),
                GRU(args,latent_dim=16,layers=3,epochs=100),
                GRU(args,latent_dim=32,layers=3,epochs=100),
                Transformer(args,d_model=8,n_heads=8,e_layers=3,d_ff=128,epochs=50),
                Transformer(args,d_model=16,n_heads=8,e_layers=3,d_ff=128,epochs=50),
                Transformer(args,d_model=32,n_heads=8,e_layers=3,d_ff=128,epochs=50),
                Transformer(args,d_model=8,n_heads=8,e_layers=3,d_ff=128,epochs=100),
                Transformer(args,d_model=16,n_heads=8,e_layers=3,d_ff=128,epochs=100),
                Transformer(args,d_model=32,n_heads=8,e_layers=3,d_ff=128,epochs=100),
               ]


model_counters = defaultdict(int)

# Build ordered model name list
model_names = []
for model in models_list:
    class_name = model.__class__.__name__
    # Update per-class counter
    model_counters[class_name] += 1
    # Append numbered model name
    model_name = f"{class_name}{model_counters[class_name]}"
    model_names.append(model_name)

print(model_names)
# import ipdb;ipdb.set_trace()

all_datasets = read_datasets_from_folder("./dataset_bank")
print(all_datasets.keys())
print('The number of datasets is',len(all_datasets.keys()))

import ipdb;ipdb.set_trace()

# Initialize score matrix: rows = datasets, columns = models
score_matrix = np.zeros((len(all_datasets), len(models_list)))

# Train and score each model on every dataset
for j, model in enumerate(models_list):
    # Iterate over all datasets
    for i, (dataset_name, dataset_info) in enumerate(all_datasets.items()):
        train_data = dataset_info['train']
        test_data = dataset_info['test']
        fault_start = dataset_info['fault_start']
        fault_end = dataset_info['fault_end']
        print('Dataset',dataset_name,'Model',j)

        if i == 0 or\
            (dataset_name.split('_')[0] != list(all_datasets.keys())[i-1].split('_')[0]) \
            or ((dataset_name.split('_')[0] != 'TEP') and (dataset_name.split('_')[0] != 'MFP') and (dataset_name.split('_')[0] != 'CutMadeProcess')):
            try:       
                model.fit(train_data,dataset_name)
            except Exception as e:
                print(f"Model {j} on {dataset_name} encounter errors: {e}")
                score_matrix[i, j] = -1
                print('score',score_matrix[i, j])
                continue
            # model.fit(train_data,dataset_name)
        else:
            # if different test sets share the same training set
            print("Model is trained on {}".format(dataset_name.split('_')[0]))
        score = model.score(test_data, fault_start, fault_end,overall=True)
        model.plot('./figs/{}_Model_{}.png'.format(dataset_name,j))
        if np.isnan(score):
            score = 0
        score_matrix[i, j] = score
        print('score',score)
        
    file_name = "output_D{}xM{}.csv".format(score_matrix.shape[0],j)
    if not os.path.exists(file_name):
        df = pd.DataFrame(score_matrix, index=list(all_datasets.keys()), columns=model_names)
        df.to_csv(file_name)
        print(f"Saved to {file_name}")
    else:
        print(f"File {file_name} already exists, skipping.")
# Score matrix output
print("Score Matrix:")
print(score_matrix)

# file_name = "output_D{}xM{}.csv".format(score_matrix.shape[0],score_matrix.shape[1])
# if not os.path.exists(file_name):
#     df = pd.DataFrame(score_matrix, index=list(all_datasets.keys()), columns=model_names)
#     df.to_csv(file_name)
#     print(f"Saved to {file_name}")
# else:
#     print(f"File {file_name} already exists, skipping.")



import ipdb;ipdb.set_trace()