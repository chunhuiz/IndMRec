import numpy as np
import pandas as pd

import argparse
from utils_tools import read_datasets_from_folder,seed_all,plot_with_tsne
import torch

from model_bank.PCA import PCA
from model_bank.KPCA import KPCA
from model_bank.SFA import SFA
# from model_bank.ESFA import ESFA
from model_bank.SSA import analytic_ssa as SSA
from model_bank.CCA import CVA
from collections import defaultdict
from model_bank.CNN import convAE
from model_bank.AE import MLPAE
from model_bank.VAE import VAE
from model_bank.Transformer import Transformer
from model_bank.GRU import GRU

from scipy.stats import rankdata
from scipy import stats

import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor

from get_meta_feature import generate_meta_features
from bls import broadnet
import datetime


parser = argparse.ArgumentParser(description="MRS")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device

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



def dict_to_array(dict,score_matrix):
    traindata = []
    trainlabel = []
    for key in list(dict.keys()):
        traindata.append(dict[key])
        y = score_matrix[score_matrix['Unnamed: 0']==key].iloc[0,1:].values.astype('float')
        trainlabel.append(np.tile(y, (dict[key].shape[0], 1)))
    traindata = np.vstack(traindata)
    trainlabel = np.vstack(trainlabel)

    trainlabel[trainlabel == -1] = 0
    return traindata, trainlabel

    
seed_all(3407)

# Load score matrix
score_matrix = pd.read_csv('output_D69xM49.csv')
score_np_data = score_matrix.iloc[:,1:].values
score_np_data[score_np_data == -1] = 0
# Load meta-feature dictionary
meta_feature = np.load('meta_fea_dict.npy',allow_pickle=True).item()

score_dataset_name = list(score_matrix.iloc[:,0])
model_name = score_matrix.columns.tolist()[1:]
meta_dataset_name = list(meta_feature.keys())

from utils_tools import get_group_indices
group_indices = get_group_indices(model_name)

if score_dataset_name != meta_dataset_name:
    print('Name inconsistency.')

matrix = np.random.rand(3000, 10)
attr_name = generate_meta_features(matrix)[1]

keys = list(meta_feature.keys())

test_size = 10
num_tests = len(keys) // test_size

groups_index_info = [[1, 6, 10, 12, 15, 29, 43, 44, 59, 69],         
[2, 5, 8, 13, 36, 37, 40, 51, 56, 61],                                
[1, 7, 14, 15, 27, 41, 59, 60, 63, 64],               
[8, 19, 20, 21, 23, 31, 32, 50, 52, 53],         
[16, 20, 23, 30, 31, 36, 38, 51, 53, 58],  ]    

f = open('output.txt', 'w')
print(model_names,file=f)
print(attr_name,file=f)

improve_list = []

for i in range(len(groups_index_info)):
    test_key_index = groups_index_info[i]
    test_keys = [keys[j-1] for j in test_key_index]


    train_keys = [item for item in keys if item not in test_keys]
    print(test_key_index)
    print(test_key_index,file=f)
    train_data_dict = {key: meta_feature[key] for key in train_keys}
    test_data_dict = {key: meta_feature[key] for key in test_keys}

    print(f"Validating with {len(test_keys)} test keys.")
    print(test_keys,file=f)

    traindata, trainlabel = dict_to_array(train_data_dict,score_matrix)
    testdata, testlabel = dict_to_array(test_data_dict,score_matrix)
    # import ipdb;ipdb.set_trace()
    if np.any(np.var(traindata, axis=0) == 0):
        print('Some meta feature has no var.')
        print(np.where(np.var(traindata, axis=0) == 0))
        save_column = (np.var(traindata, axis=0) != 0)
    traindata = traindata[:, save_column]
    testdata = testdata[:, save_column]


    traindata = np.nan_to_num(traindata, nan=0.0)   
    testdata = np.nan_to_num(testdata, nan=0.0)   

    starttime = datetime.datetime.now()
    seed_all(2345)
    bls = broadnet(maptimes = 10, enhencetimes = 10, map_function = 'relu', enhence_function = 'relu',batchsize = 128, reg = 0.001)

    bls.fit(traindata,trainlabel)
    endtime = datetime.datetime.now()
    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))
    
    ####################################################
    ####    TEST   #####################################
    ####################################################

    ranks_list = []
    score_list = []
    cur_rank_list = []
    cur_score_list = []
    for key in list(test_data_dict.keys()):
        cur_test_data = test_data_dict[key][:, save_column]
        pred_score = bls.predict(cur_test_data)
        cur_score = score_np_data[np.where(score_matrix['Unnamed: 0']==key)[0][0],:]
        ranks = rankdata(-cur_score, method='dense')

        score_list.append(cur_score)
        ranks_list.append(ranks)

        max_indices = np.argmax(pred_score, axis=1)
        vote_index = stats.mode(max_indices).mode[0]
        cur_ranks = ranks[vote_index]
        cur_rank_list.append(cur_ranks)
        cur_score_list.append(cur_score[vote_index])
        print(f"{key:<{25}} Cur rank {cur_ranks:<{3}} Cur score {cur_score[vote_index]:.3f} Best score {max(cur_score):.3f}")
        print(f"{key:<{25}} Cur rank {cur_ranks:<{3}} Cur score {cur_score[vote_index]:.3f} Best score {max(cur_score):.3f}",file=f)

    score_array = np.vstack(score_list)
    ranks_array = np.vstack(ranks_list)

    grouped_score_max = np.array([score_array[:,indices].mean(0).max() for indices in group_indices.values()]).T
    grouped_rank_min = np.array([ranks_array[:,indices].mean(0).min() for indices in group_indices.values()]).T

    print('Best rank without RS',ranks_array.mean(0).min(),model_name[ranks_array.mean(0).argmin()])
    print('Best score without RS',score_array.mean(0).max(),model_name[score_array.mean(0).argmax()])
    print('Rank with RS',np.mean(cur_rank_list))
    print('Score with RS',np.mean(cur_score_list))

    print(" ".join("&{:.3f}".format(num) for num in grouped_score_max))
    print(" ".join("&{:.1f}".format(num) for num in grouped_rank_min))

    print('Best rank without RS',ranks_array.mean(0).min(),model_name[ranks_array.mean(0).argmin()],file=f)
    print('Best score without RS',score_array.mean(0).max(),model_name[score_array.mean(0).argmax()],file=f)
    # import ipdb;ipdb.set_trace()
    print('Rank with RS {:.1f} ({:.1f})'.format(np.mean(cur_rank_list),np.mean(cur_rank_list)-ranks_array.mean(0).min()),file=f)
    print('Score with RS {:.3f} (+{:.3f})'.format(np.mean(cur_score_list),np.mean(cur_score_list)-score_array.mean(0).max()),file=f)

    print(" ".join("&{:.3f}".format(num) for num in grouped_score_max),file=f)
    print(" ".join("&{:.1f}".format(num) for num in grouped_rank_min),file=f)
    # import ipdb;ipdb.set_trace()
    improve_list.append(np.mean(cur_score_list)-score_array.mean(0).max())



f.close()
import ipdb;ipdb.set_trace()
