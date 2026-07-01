# import pandas as pd
# from sklearn.decomposition import PCA
# from scipy.linalg import expm
# import scipy.linalg
import scipy.stats as stats
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.font_manager import FontProperties
# import math
# from scipy import stats
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics.pairwise import euclidean_distances
from utils_tools import find_kde, calculate_statistics
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['simhei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']

plt.rcParams['axes.unicode_minus']=False

class KPCA:
    def __init__(self,confidence,cumulative_percent_variance):  # no normalization required
        self.confidence = confidence
        self.cumulative_percent_variance = cumulative_percent_variance
        # self.PCs = PCs
        self.p_k = None  # loading matrix
        self.v_I = None # $\Sigma^{-1}$
        self.T_limit = None
        self.SPE_limit = None
        self.test_data = None
        self.data_mean = None
        self.data_std = None

    def construct_kernel_matrix(self,Y,X):
        dists=euclidean_distances(X=Y,Y=X)**2
        K=np.exp(-dists/(2*(self.para**2)))
        return K
    
    def fit(self, data_original,dataset_name):
        self.data_mean = np.mean(data_original,0)
        self.data_std = np.std(data_original,0)

        data_nor = (data_original - self.data_mean)/(self.data_std)
        self.data_train_nor = data_nor

        para=pdist(data_nor)**2
        para=np.sqrt(squareform(para))
        para[para<=0]=float("inf")
        para=np.min(para,axis=0)
        self.para=5*np.mean(para)

        kernel_matrix = self.construct_kernel_matrix(data_nor,self.data_train_nor)
        N = data_original.shape[0]
        one_n=np.ones([N,N])/N

        X=kernel_matrix-one_n.dot(kernel_matrix)-kernel_matrix.dot(one_n)+one_n.dot(kernel_matrix).dot(one_n)
        
        P,v,P_t = np.linalg.svd(X)  # SVD for loading matrix (returns u, s, v; v is u.T)
        Z = np.dot(P,P_t)
        v_sum = np.sum(v)

        k = []
        for x in range(len(v)):
            PE_k = v[x]/v_sum
            if x == 0:
                PE_sum = PE_k 
            else:
                PE_sum = PE_sum + PE_k
            if PE_sum < self.cumulative_percent_variance: 
                pass
            else:
                k.append(x+1)
                # print(k)
                break
        ## principal components
        p_k = P[:,:k[0]]
        v_I = np.diag(1/v[:k[0]])

        self.p_k = p_k
        self.v_I = v_I
        self.get_statistics(data_original,is_train=True)
        return v_I, p_k,v,P,k


    def transform(self, data_original):
        self.test_data = data_original
        data_nor = (data_original - self.data_mean)/(self.data_std)
        data_nor = self.construct_kernel_matrix(data_nor,self.data_train_nor)
        feature = np.dot(data_nor,self.p_k)
        return feature
    
    def get_statistics(self, raw_test_data, is_train=False):
        feature = self.transform(raw_test_data)
        cov = np.cov(feature.T)
        T2 = calculate_statistics(feature,cov, type='T2')
        SPE = calculate_statistics(feature, type='SPE')
        # import ipdb;ipdb.set_trace()
        if is_train:
            # Estimate control limits from training statistics via KDE
            self.T2_kzx = find_kde(T2, self.confidence)
            self.SPE_kzx = find_kde(SPE, self.confidence)
        return [T2, SPE], [self.T2_kzx, self.SPE_kzx]
    
    
    def score(self,test_data,fault_start,fault_end,overall=False):
        T2_fault,_ = self.get_statistics(test_data)
        self.T2_fault = T2_fault
        T2_kzx = [self.T2_kzx, self.SPE_kzx]
        # import ipdb;ipdb.set_trace()
        TP = sum((np.array(T2_fault[0][fault_start:fault_end])>=T2_kzx[0])|(np.array(T2_fault[1][fault_start:fault_end])>=T2_kzx[1]))
        TN = sum((np.array(T2_fault[0][:fault_start])<T2_kzx[0])&(np.array(T2_fault[1][:fault_start])<T2_kzx[1])) + \
                sum((np.array(T2_fault[0][fault_end:])<T2_kzx[0])&(np.array(T2_fault[1][fault_end:])<T2_kzx[1]))
        FP = sum((np.array(T2_fault[0][:fault_start])>=T2_kzx[0])|(np.array(T2_fault[1][:fault_start])>=T2_kzx[1])) + \
                sum((np.array(T2_fault[0][fault_end:])>=T2_kzx[0])|(np.array(T2_fault[1][fault_end:])>=T2_kzx[1]))
        FN = sum((np.array(T2_fault[0][fault_start:fault_end])<T2_kzx[0])&(np.array(T2_fault[1][fault_start:fault_end])<T2_kzx[1]))
        # Accuracy: fraction of correctly classified samples
        if overall:
            # Precision= TP / TP + FP
            # Recall = TP / TP + FN
            # F1 = 2 P * R / P + R
            Pre = TP/(TP+FP)
            Rec = TP/(TP+FN)
            if Pre+Rec == 0:
                return 0
            return  2*Pre*Rec/(Pre+Rec)
        else:
            return TP/(fault_end-fault_start)

    def plot(self,path):
        plt.suptitle('PCA:'+path.split('.png')[0])
        plt.subplot(2,1,1)
        # plt.yscale('log')
        plt.plot(self.T2_fault[0])
        plt.ylabel('T2')
        plt.axhline(self.T2_kzx, c='r')
        plt.subplot(2,1,2)
        # plt.yscale('log')
        plt.plot(self.T2_fault[1])
        plt.ylabel('SPE')
        plt.axhline(self.SPE_kzx, c='r')
        plt.savefig(path,dpi=300)
        plt.close()