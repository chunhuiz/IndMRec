import numpy as np
from model_bank.PCA import PCA
from model_bank.SFA import SFA
from trash.ESFA import ESFA

from utils_tools import read_datasets_from_folder,seed_all
from get_meta_feature import generate_meta_features

seed_all(3407)

all_datasets = read_datasets_from_folder("./dataset_bank")

def sliding_window(matrix, window_length, number_of_windows):
    if np.any(np.var(matrix, axis=0) == 0):
        print('Some columns are no var, delete them.')
        matrix = matrix[:, np.var(matrix, axis=0) != 0]  # drop zero-variance columns

    # Matrix shape
    num_samples, _ = matrix.shape
    
    # Skip sliding window if window exceeds sample count
    if window_length > num_samples:
        print('window_length > num_samples, early return')
        return generate_meta_features(matrix)[0]

    # Step size so windows cover the full series (minimum step = 1 if non-overlapping)
    step_length = max(1, (num_samples - window_length) // (number_of_windows - 1))
    
    # Effective number of windows
    effective_number_of_windows = min(
        (num_samples - window_length) // step_length + 1,
        number_of_windows
    )

    # Collected window features
    windows = []

    # Slide window over data
    for i in range(0, step_length * effective_number_of_windows, step_length):
        # Keep indices in bounds
        start_index = i
        end_index = i + window_length
        if np.any(np.var(matrix[start_index:end_index, :], axis=0) == 0):
            # Skip window with locally zero-variance columns
            start_index = end_index
            # print('Column no var, continue')
            continue
        if end_index <= num_samples:
            windows.append(generate_meta_features(matrix[start_index:end_index, :])[0])
        else:
            # Handle trailing boundary
            windows.append(generate_meta_features(matrix[-window_length:, :])[0])
            break
    try:
        result = np.vstack(windows)
    except:
        return None
    return result


# import ipdb;ipdb.set_trace()

final_fea_dict = {}
embeddings_list = []
window_num = 100
window_len = 1000

final_fea_dict = np.load("meta_fea_temp.npy",allow_pickle=True).item()
# import ipdb;ipdb.set_trace()
# Iterate over all datasets
for i, (dataset_name, dataset_info) in enumerate(all_datasets.items()):
    train_data = dataset_info['train']
    test_data = dataset_info['test']
    fault_start = dataset_info['fault_start']
    fault_end = dataset_info['fault_end']
    emb = dataset_info['note']

    embeddings_list.append(emb)
    if dataset_name in list(final_fea_dict.keys()):
        print(dataset_name,final_fea_dict[dataset_name].shape)
        continue
    # if 'CutMadeProcess' in dataset_name or  'GasPlantC_case01' in dataset_name or 'GasPlantC_case02' in dataset_name:
    #     continue
    print('#############')
    print(dataset_name)
    print('#############')

    if i == 0 or\
            (dataset_name.split('_')[0] != list(all_datasets.keys())[i-1].split('_')[0]) \
            or ((dataset_name.split('_')[0] != 'TEP') and (dataset_name.split('_')[0] != 'MFP') and (dataset_name.split('_')[0] != 'CutMadeProcess')):

        result1 = sliding_window(train_data, window_len, window_num)
        result2 = sliding_window(test_data, window_len, window_num)
    else:
        print("Train data are processed on {}".format(dataset_name.split('_')[0]))
        result2 = sliding_window(test_data, window_len, window_num)

    if result1 is not None and result2 is not None:
        final_fea_dict[dataset_name] = np.vstack((result1, result2))
        print(final_fea_dict[dataset_name].shape)
    elif result1 is None and result2 is not None:
        final_fea_dict[dataset_name] = result2
    elif result1 is not None and result2 is None:
        final_fea_dict[dataset_name] = result1
    else:
        final_fea_dict[dataset_name] = None

    np.save("meta_fea_temp.npy",np.array(final_fea_dict))
embeddings_array = np.vstack(embeddings_list)

# import ipdb;ipdb.set_trace()
# PCA for BERT embeddings
from sklearn.decomposition import PCA as sklearn_PCA
pca_transformer = sklearn_PCA(n_components=10)
pca_transformer.fit_transform(embeddings_array)

for i, (dataset_name, dataset_info) in enumerate(all_datasets.items()):
    emb = dataset_info['note']
    emb_pca = pca_transformer.transform(emb)

    N, A = final_fea_dict[dataset_name].shape
    
    expanded_emb_pca = np.tile(emb_pca, (N, 1))
    final_fea_dict[dataset_name] = np.hstack((final_fea_dict[dataset_name], expanded_emb_pca))
    print(dataset_name,final_fea_dict[dataset_name].shape)
np.save('meta_fea_dict.npy',np.array(final_fea_dict))
import ipdb;ipdb.set_trace()