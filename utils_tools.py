import numpy as np
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.io import loadmat
import re
import random
import torch
import pandas as pd
from sklearn.impute import SimpleImputer
# import torch
from torch.utils.data import Dataset

def load_note_embedding(folder: str) -> np.ndarray:
    """Load precomputed process-description embedding (384-d, from SentenceTransformer)."""
    path = os.path.join(folder, 'note.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing note.npy in {folder}')
    return np.load(path)

def seed_all(seed):
    # Set NumPy random seed
    np.random.seed(seed)
    # Set Python random module seed
    random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MedianImputerWithVarianceFilter:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.non_zero_variance_columns = None

    def fit(self, X, y=None):
        # Impute missing values with column medians
        X_imputed = self.imputer.fit_transform(X)
        # Per-column variance
        variances = np.var(X_imputed, axis=0)
        # Indices of non-zero-variance columns
        self.non_zero_variance_columns = np.where(variances != 0)[0]
        X_filtered = X_imputed[:, self.non_zero_variance_columns]
        return X_filtered

    def transform(self, X, y=None):
        # Impute with fitted median imputer
        X_imputed = self.imputer.transform(X)
        # Return only non-zero-variance columns
        X_filtered = X_imputed[:, self.non_zero_variance_columns]
        return X_filtered
        
def filter_dataframes_by_columns(file_path, df1, df2): 
    """
    Filter two DataFrames to columns listed in columns.txt.
    Missing column names are skipped; both outputs share the same columns.
    Note: the Time column is already removed.

    Args:
    file_path: path to columns.txt.
    df1: first DataFrame to filter.
    df2: second DataFrame to filter.

    Returns:
    df1_filtered: filtered first DataFrame.
    df2_filtered: filtered second DataFrame.
    """
    # Read column names from file

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        columns_needed = [line.split()[0] for line in lines if line != '\n']  # first token per line

    # Columns present in both DataFrames
    columns_in_both_dfs = [col for col in columns_needed if col in df1.columns and col in df2.columns]

    # Select matching columns
    df1_filtered = df1[columns_in_both_dfs]
    df2_filtered = df2[columns_in_both_dfs]
    
    return df1_filtered, df2_filtered

def calculate_missing_value_ratio(array):
    """
    Compute the fraction of NaN values in a NumPy array.

    Args:
    array: ndarray that may contain NaN values.

    Returns:
    missing_value_ratio: percentage of NaN values in the array.
    """
    total_values_count = array.size
    missing_values_count = np.isnan(array).sum()
    
    missing_value_ratio = 100 * missing_values_count / total_values_count
    # import ipdb;ipdb.set_trace()
    return missing_value_ratio

def clean_and_convert_to_float(array):
    """
    Convert invalid string entries to NaN and cast the array to float.

    Args:
    array: raw ndarray that may contain non-numeric strings.

    Returns:
    cleaned_array: float ndarray with invalid strings replaced by NaN.
    """
    # Use pandas for mixed-type coercion
    df = pd.DataFrame(array)
    
    # Coerce non-numeric values to NaN
    cleaned_df = df.apply(pd.to_numeric, errors='coerce')
    
    # Convert back to NumPy
    cleaned_array = cleaned_df.to_numpy()
    print("Missing rate",calculate_missing_value_ratio(cleaned_array))
    return cleaned_array

def extract_prefix_and_number(s):
    match = re.match(r'(\D*)(\d*)', s)
    prefix, number = match.groups()
    return (prefix, int(number) if number else 0)

def read_datasets_from_folder(folder_path):
    datasets = {}
    # Sort subdirectories for deterministic iteration order
    subdirs = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    subdirs.sort(key=lambda x: extract_prefix_and_number(os.path.basename(x)))
    print(subdirs)
    TN_num = 200
    for subdir in subdirs:
        # if  'ThermalPlantA' in subdir\
        #     or 'ThermalPlantB' in subdir\
        #     or 'ThermalPlantC' in subdir\
        #     or 'GasTurbineGenerationProcess' in subdir\
        #     or 'MFP' in subdir:
        #     # or 'TEP' in subdir:
        #     # or 'CutMade' in subdir:
        #     continue
        if 'ThermalPlantA' in subdir:
            for root, dirs, files in os.walk(subdir):
                # mat_files = [file for file in files if file.endswith('.xlsx')]
                mat_files = [file for file in files if file.endswith('.npy')]
                txt_files = [file for file in files if file.endswith('.txt')]
                # import ipdb;ipdb.set_trace()
                if len(mat_files) >= 2 and '说明' not in root:
                    dataset_name = os.path.basename(root)

                    train_file = next((f for f in mat_files if 'abnormal' not in f and 'note' not in f), None)
                    test_file = next((f for f in mat_files if 'abnormal' in f), None)

                    if train_file and test_file:
                        print(root)
                        ########################################################
                        ###########  first clean ################################
                        ########################################################
                        # train_data = pd.read_excel(os.path.join(root, train_file))
                        # test_data = pd.read_excel(os.path.join(root, test_file))
                        # train_data['TimeStamp'] = pd.to_datetime(train_data['TimeStamp'], errors='coerce')
                        # test_data['TimeStamp'] = pd.to_datetime(test_data['TimeStamp'], errors='coerce')
                        # train_data,test_data = filter_dataframes_by_columns(os.path.join(root, 'columns_save.txt'),train_data,test_data)
                        # np.save(os.path.join(root, train_file.replace('xlsx','npy')), np.array(train_data))
                        # np.save(os.path.join(root, test_file.replace('xlsx','npy')), np.array(test_data))
                        ########################################################
                        ###########  quick load ################################
                        ########################################################
                        train_data = np.load(os.path.join(root, train_file), allow_pickle=True)
                        test_data = np.load(os.path.join(root, test_file), allow_pickle=True)
                        # Remove invalid string outliers
                        train_data = clean_and_convert_to_float(train_data)
                        test_data = clean_and_convert_to_float(test_data)
                        print(train_data.shape)
                        print(test_data.shape)
                        filter = MedianImputerWithVarianceFilter()
                        train_data = filter.fit(train_data)
                        test_data = filter.transform(test_data)
                        print(train_data.shape)
                        print(test_data.shape)
                        embeddings = load_note_embedding(root)
                        datasets['ThermalPlantA_'+dataset_name] = {
                            'train': train_data[:-TN_num],
                            'test': np.vstack((train_data[-TN_num:], test_data)),
                            'fault_start': TN_num,
                            'fault_end': np.vstack((train_data[-TN_num:], test_data)).shape[0],
                            'note':embeddings
                        }
                        # import ipdb;ipdb.set_trace()
            pass
        elif 'ThermalPlantB' in subdir:
            for root, dirs, files in os.walk(subdir):
                mat_files = [file for file in files if file.endswith('.xlsx')]
                # mat_files = [file for file in files if file.endswith('.npy')]
                txt_files = [file for file in files if file.endswith('.txt')]
                info_file = next((f for f in txt_files if 'info' in f), None)
                if info_file is not None:
                    print(root)
                    dataset_name = os.path.basename(root)
                    # data_file = next((f for f in mat_files if '._' not in f and '~' not in f), None)
                    # print(data_file)
                    # all_data = pd.read_excel(os.path.join(root, data_file)).iloc[2:,1:]
                    # filter = MedianImputerWithVarianceFilter()
                    # all_data = filter.fit(clean_and_convert_to_float(all_data))
                    # # Read and parse info.txt
                    # with open(os.path.join(root, txt_files[0]), 'r') as file:
                    #     lines = file.readlines()
                    #     if len(lines) == 1:
                    #         fault_start = int(lines[0].strip())
                    #         train_data = all_data[:fault_start]
                    #         test_data = all_data[fault_start:]
                    #     elif len(lines) == 2:
                    #         fault_start = int(lines[0].strip())
                    #         fault_end = int(lines[1].strip())
                    #         train_data = all_data[:fault_start]
                    #         test_data = all_data[fault_start:fault_end]
                    #     else:
                    #         raise ValueError(f'info.txt in {root} does not have the expected format.')
                    # np.save(os.path.join(root, 'train.npy'), np.array(train_data))
                    # np.save(os.path.join(root, 'test.npy'), np.array(test_data))
                    
                    train_data = np.load(os.path.join(root, 'train.npy'), allow_pickle=True)
                    test_data = np.load(os.path.join(root, 'test.npy'), allow_pickle=True)
                    print(train_data.shape)
                    print(test_data.shape)
                    embeddings = load_note_embedding(root)
                    datasets['ThermalPlantB_'+dataset_name] = {
                            'train': train_data[:-TN_num],
                            'test': np.vstack((train_data[-TN_num:], test_data)),
                            'fault_start': TN_num,
                            'fault_end': np.vstack((train_data[-TN_num:], test_data)).shape[0],
                            'note':embeddings
                        }
            # import ipdb;ipdb.set_trace()
            pass
        elif 'ThermalPlantC' in subdir:
            for root, dirs, files in os.walk(subdir):
                mat_files = [file for file in files if file.endswith('.xls')]
                # mat_files = [file for file in files if file.endswith('.npy')]
                txt_files = [file for file in files if file.endswith('.txt')]
                info_file = next((f for f in txt_files if 'info' in f), None)
                if info_file is not None:
                    print(root)
                    dataset_name = os.path.basename(root)
                    # import ipdb;ipdb.set_trace()
                    # data_file = next((f for f in mat_files if '._' not in f and '~' not in f), None)
                    # print(data_file)
                    # all_data = pd.read_excel(os.path.join(root, data_file)).iloc[:,1:]
                    # # import ipdb;ipdb.set_trace()
                    # filter = MedianImputerWithVarianceFilter()
                    # all_data = filter.fit(clean_and_convert_to_float(all_data))
                    # # Read and parse info.txt
                    # with open(os.path.join(root, txt_files[0]), 'r') as file:
                    #     lines = file.readlines()
                    #     if len(lines) == 1:
                    #         fault_start = int(lines[0].strip())
                    #         train_data = all_data[:fault_start]
                    #         test_data = all_data[fault_start:]
                    #     elif len(lines) == 2:
                    #         fault_start = int(lines[0].strip())
                    #         fault_end = int(lines[1].strip())
                    #         train_data = all_data[:fault_start]
                    #         test_data = all_data[fault_start:fault_end]
                    #     else:
                    #         raise ValueError(f'info.txt in {root} does not have the expected format.')
                    # np.save(os.path.join(root, 'train.npy'), np.array(train_data))
                    # np.save(os.path.join(root, 'test.npy'), np.array(test_data))
                    
                    train_data = np.load(os.path.join(root, 'train.npy'), allow_pickle=True)
                    test_data = np.load(os.path.join(root, 'test.npy'), allow_pickle=True)
                    print(train_data.shape)
                    print(test_data.shape)
                    embeddings = load_note_embedding(root)
                    datasets['ThermalPlantC_'+dataset_name] = {
                            'train': train_data[:-TN_num],
                            'test': np.vstack((train_data[-TN_num:], test_data)),
                            'fault_start': TN_num,
                            'fault_end': np.vstack((train_data[-TN_num:], test_data)).shape[0],
                            'note':embeddings
                        }
                    
            pass
        elif 'CutMadeProcess' in subdir:
            train_data = np.load(subdir+'/data_normal.npy', allow_pickle=True)
            test_data_1 = np.load(subdir+'/data_fault1.npy', allow_pickle=True)
            test_data_2 = np.load(subdir+'/data_fault2.npy', allow_pickle=True)
            train_data = clean_and_convert_to_float(train_data)
            test_data_1 = clean_and_convert_to_float(test_data_1)
            test_data_2 = clean_and_convert_to_float(test_data_2)
            filter = MedianImputerWithVarianceFilter()
            train_data = filter.fit(train_data)
            test_data_1 = filter.transform(test_data_1)
            test_data_2 = filter.transform(test_data_2)
            embeddings = load_note_embedding(subdir)
            datasets['CutMadeProcess_case01'] = {
                            'train': train_data[:-100],
                            'test': np.vstack((train_data[-100:], test_data_1)),
                            'fault_start': 100,
                            'fault_end': np.vstack((train_data[-100:], test_data_1)).shape[0],
                            'note':embeddings
                        }
            datasets['CutMadeProcess_case02'] = {
                            'train': train_data[:-100],
                            'test': np.vstack((train_data[-100:], test_data_2)),
                            'fault_start': 100,
                            'fault_end': np.vstack((train_data[-100:], test_data_2)).shape[0],
                            'note':embeddings
                        }
            
            # import ipdb;ipdb.set_trace()
            pass
        elif 'GasTurbineGenerationProcess' in subdir:
            for root, dirs, files in os.walk(subdir):
                txt_files = [file for file in files if file.endswith('.txt')]
                info_file = next((f for f in txt_files if 'info' in f), None)
                train_npy = os.path.join(root, 'train.npy')
                test_npy = os.path.join(root, 'test.npy')
                if info_file is not None and os.path.exists(train_npy) and os.path.exists(test_npy):
                    print(root)
                    dataset_name = os.path.basename(root)
                    train_data = np.load(train_npy, allow_pickle=True)
                    test_data = np.load(test_npy, allow_pickle=True)
                    with open(os.path.join(root, info_file), 'r') as file:
                        lines = file.readlines()
                        if len(lines) == 3:
                            train_end = int(lines[0].strip())
                            fault_start = int(lines[1].strip())
                        else:
                            raise ValueError(f'info.txt in {root} does not have the expected format.')
                    embeddings = load_note_embedding(root)
                    print(train_data.shape)
                    print(test_data.shape)
                    datasets['GasPlantC_'+dataset_name] = {
                            'train': train_data,
                            'test': test_data,
                            'fault_start': fault_start - train_end,
                            'fault_end': test_data.shape[0],
                            'note':embeddings
                        }
            # import ipdb;ipdb.set_trace()
            pass
        elif 'MFP' in subdir:
            X_1=pd.read_csv(subdir+'/T1.csv')
            X_1.drop('Unnamed: 0',axis = 1, inplace=True)
            X_1 = X_1.values
            X_2=pd.read_csv(subdir+'/T2.csv')
            X_2.drop('Unnamed: 0',axis = 1, inplace=True)
            X_2 = X_2.values
            X_3=pd.read_csv(subdir+'/T3.csv')
            X_3.drop('Unnamed: 0',axis = 1, inplace=True)
            X_3 = X_3.values
            train_data = np.concatenate((X_1,X_2,X_3),axis=0)
            # import ipdb;ipdb.set_trace()
            embeddings = load_note_embedding(subdir)

            # import ipdb;ipdb.set_trace()
            datasets['MFP_fault_1_1'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase1.mat')['Set1_1'],
                        'fault_start': 1566,'fault_end': 5181,
                        'note': embeddings
                    }
            datasets['MFP_fault_1_2'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase1.mat')['Set1_2'],
                        'fault_start': 657,'fault_end': 3777,
                        'note': embeddings
                    }
            datasets['MFP_fault_1_3'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase1.mat')['Set1_3'],
                        'fault_start': 691,'fault_end': 3691,
                        'note': embeddings
                    }
            datasets['MFP_fault_2_1'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase2.mat')['Set2_1'],
                        'fault_start': 2244,'fault_end': 6616,
                        'note': embeddings
                    }
            datasets['MFP_fault_2_2'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase2.mat')['Set2_2'],
                        'fault_start': 476,'fault_end': 2656,
                        'note': embeddings
                    }
            datasets['MFP_fault_2_3'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase2.mat')['Set2_3'],
                        'fault_start': 331,'fault_end': 2467,
                        'note': embeddings
                    }
            datasets['MFP_fault_3_1'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase3.mat')['Set3_1'],
                        'fault_start': 1136,'fault_end': 8352,
                        'note': embeddings
                    }
            datasets['MFP_fault_3_2'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase3.mat')['Set3_2'],
                        'fault_start': 333,'fault_end': 5871,
                        'note': embeddings
                    }
            datasets['MFP_fault_3_3'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase3.mat')['Set3_3'],
                        'fault_start': 596,'fault_end': 9566,
                        'note': embeddings
                    }
            datasets['MFP_fault_4_1'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase4.mat')['Set4_1'],
                        'fault_start': 953,'fault_end': 6294,
                        'note': embeddings
                    }
            datasets['MFP_fault_4_2'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase4.mat')['Set4_2'],
                        'fault_start': 851,'fault_end': 3851,
                        'note': embeddings
                    }
            datasets['MFP_fault_4_3'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase4.mat')['Set4_3'],
                        'fault_start': 241,'fault_end': 3241,
                        'note': embeddings
                    }
            datasets['MFP_fault_5_1'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase5.mat')['Set5_1'][:1772],
                        'fault_start': 686,'fault_end': 1172,
                        'note': embeddings
                    }
            datasets['MFP_fault_5_2'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase5.mat')['Set5_2'][:7031],
                        'fault_start': 1633,'fault_end': 2955,
                        'note': embeddings
                    }
            datasets['MFP_fault_6_1'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase6.mat')['Set6_1'],
                        'fault_start': 1723,'fault_end': 2800,
                        'note': embeddings
                    }
            datasets['MFP_fault_6_2'] = {'train': train_data,
                        'test': loadmat(subdir+'/FaultyCase6.mat')['Set6_2'],
                        'fault_start': 1037,'fault_end': 4830,
                        'note': embeddings
                    }
            
            pass

        elif 'TEP' in subdir:
            for root, dirs, files in os.walk(subdir):

                mat_files = [file for file in files if file.endswith('.mat')]
                txt_files = [file for file in files if file.endswith('.txt')]
                # import ipdb;ipdb.set_trace()
                if len(mat_files) >= 2 and len(txt_files) == 1:
                    dataset_name = os.path.basename(root)

                    # train_file = next((f for f in mat_files if 'te' not in f), None)
                    train_file = 'd00.mat'
                    test_file = next((f for f in mat_files if 'te' in f), None)
                    
                    if train_file and test_file:
                        train_data = loadmat(os.path.join(root, train_file))
                        test_data = loadmat(os.path.join(root, test_file))

                        # Read and parse info.txt
                        with open(os.path.join(root, txt_files[0]), 'r') as file:
                            lines = file.readlines()
                            if len(lines) == 2:
                                fault_start = int(lines[0].strip())
                                fault_end = test_data.shape[0] if lines[1].strip() == '-1' else int(lines[1].strip())
                            else:
                                raise ValueError(f'info.txt in {root} does not have the expected format.')

                        embeddings = load_note_embedding(subdir)
                        datasets[dataset_name] = {
                            'train': train_data['data'].T,
                            'test': test_data['data'].T,
                            'fault_start': fault_start,
                            'fault_end': fault_end,
                            'note':embeddings
                        }
                        # import ipdb;ipdb.set_trace()
    new_keys_list = list(datasets.keys())
    new_keys_list.sort(key=lambda x: extract_prefix_and_number(os.path.basename(x)))
    sorted_dict = {key: datasets[key] for key in new_keys_list}
    # import ipdb;ipdb.set_trace()
    return sorted_dict

def add_missing_values(data, missing_rate=0.3):
    # import ipdb;ipdb.set_trace()
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be between 0 and 1.")

    total_values = data.size
    missing_count = int(total_values * missing_rate)

    flattened_data = data.flatten()
    missing_indices = np.random.choice(total_values, missing_count, replace=False)
    flattened_data[missing_indices] = np.nan

    return flattened_data.reshape(data.shape)

# def find_kde(input, Confidence):

#     """
#     KDE nonparametric estimation computes control limit function
#     :param1 data_original: one-dimensional nadarray data
#     :param2 data_original: The parameter Confidence is the confidence level, which is generally above 0.9
#     :return: Specific value of control limit
#     """
#     plt.figure()
#     # ax = sns.kdeplot(input, kernel='Gaussian', bw=((input.max() - input.min()) / 1000))
#     ax = sns.kdeplot(input, cumulative=True, kernel='gau'
#                      , bw=((input.max() - input.min()) / 1000))
#     line = ax.lines[0]
#     x1, y1 = line.get_data()
#     for i in range(len(y1)):
#         if y1[i] > Confidence:
#             kzx = x1[i - 1] + (x1[i] - x1[i - 1]) * (y1[i] - Confidence) / (y1[i] - y1[i - 1])
#             break
#     plt.close()
#     return kzx

def calculate_statistics(features, cov=None, type='SPE'):
    """
    Compute T2 or SPE monitoring statistics.
    :param1 features: shape samples x features
    :param2 cov: covariance matrix required for T2
    :param3 type: 'T2' or 'SPE'
    :return: statistics vector of shape samples x 1
    """
    if type == 'T2':
        n_samples = features.shape[0]
        statistics = np.zeros(n_samples)
        inv_cov = np.linalg.inv(cov)
        for i in range(n_samples):
            # import ipdb;ipdb.set_trace()
            statistics[i] = np.dot(np.dot(features[i,:], inv_cov), features[i,:])
    else:
        statistics = np.diag(np.dot(features, features.T))
    return statistics


def find_kde(T2, confidence):

    """
    Estimate control limit via KDE.
    :param1 T2: monitoring statistics, shape samples x 1
    :param2 confidence: confidence level, typically > 0.9
    :return: control limit
    """
    # T2 = T2[:,np.newaxis]
    # plt.figure()
    # params = {'bandwidth': np.logspace(-1, 1, 20)}
    # grid = GridSearchCV(KernelDensity(kernel='gaussian'), params)
    # grid.fit(T2)

    # print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    # use the best estimator to compute the kernel density estimate
    ax = sns.kdeplot(data=T2, cumulative=True)
    line = ax.lines[0]
    x1, y1 = line.get_data()
    for i in range(len(y1)):
        if y1[i] > confidence:
            control_limit = x1[i]
            break
    plt.close()
    return control_limit

def plot_data(data, N, title, ylabel):
    """
    Plot process variables.
    data: shape samples x variables
    N: number of variables to plot
    title: figure title
    ylabel: y-axis label prefix or list
    """
    f = plt.figure(figsize=(15,20))
    fz = 14
    f.suptitle(title, fontsize=16, fontweight='bold')
    for i in range(N):
        plt.subplot(int(np.ceil(float(N)/2)), 2, i+1)
        plt.plot(data[:,i])
        if isinstance(ylabel, str):
            plt.ylabel(ylabel + str(i+1), fontsize=fz)
        else:
            plt.ylabel(ylabel[i], fontsize=fz)

def plot_statistics(statistics, control_limits, title, ylabels, fault_start):
    """
    Plot monitoring statistics with control limits.
    statistics: list of statistic series
    control_limits: list of control limits
    title: figure title
    ylabels: y-axis labels
    fault_start: fault onset sample index
    """
    f = plt.figure(figsize=(6,6))
    title_font = {
    'fontsize': 16,
    'fontweight': 'bold',
    # 'fontfamily': 'Arial'
}
    f.suptitle(title, fontsize=16, fontweight='bold')
    nums = len(statistics)
    lw = 1.25
    fz = 14
    for i in range(nums):
        T2 = statistics[i]
        plt.subplot(nums+1,1,i+1)
        plt.plot(T2, linewidth=lw, c='blue')
        # plt.yscale('log')
        plt.axhline(control_limits[i], c='r')
        plt.axvline(fault_start, c='r',linestyle='--')
        # print('FAR T2 {}: {:.3f}'.format(i, np.sum(T2>control_limits[i])/T2.shape[0]))
        # Axis label font size
        ax = plt.gca()
        ax.set_xlabel('Samples', fontsize=fz)
        ax.set_ylabel(ylabels[i], fontsize=fz)
        # Tick label font size
        ax.tick_params(axis='both', labelsize=12)
        ax.spines['left'].set_linewidth(2)  # left spine width
        ax.spines['bottom'].set_linewidth(2)  # bottom spine width

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size=64):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        # Total number of windows
        return len(self.data) - self.window_size + 1

    def __getitem__(self, index):
        # Get the data for one window
        window = self.data[index:index+self.window_size]
        return window
    
class NoWindowDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def get_group_indices(arr):
    # Map group prefix -> list of indices
    groups = {}
    
    # Match alphabetic prefix at start of each name
    pattern = re.compile(r'^\D+')
    
    for index, value in enumerate(arr):
        # Extract prefix with regex
        match = pattern.match(value)
        if match:
            key = match.group()
            if key not in groups:
                groups[key] = []
            groups[key].append(index)
    
    # Return grouped index dict
    return groups