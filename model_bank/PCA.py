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
from utils_tools import find_kde, calculate_statistics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['simhei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']

plt.rcParams['axes.unicode_minus']=False

class PCA:
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

    def fit(self, data_original,dataset_name):
        # self.data_mean = np.mean(data_original,0)
        # self.data_std = np.std(data_original,0)
        # data_nor = (data_original - self.data_mean)/(self.data_std)
        self.sc = StandardScaler()
        data_nor = self.sc.fit_transform(data_original)
        X = np.cov(data_nor.T)
        # import ipdb;ipdb.set_trace()
        # print(np.where(np.isnan(X)))
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

        # ## T2 statistic threshold
        # coe = k[0]*(np.shape(data_original)[0]-1)*(np.shape(data_original)[0]+1)/((np.shape(data_original)[0]-k[0])*np.shape(data_original)[0])
        # T_limit = coe*stats.f.ppf(self.confidence,k[0],(np.shape(data_original)[0]-k[0]))
        # ## SPE statistic threshold
        # O1 = np.sum((v[k[0]:])**1)
        # O2 = np.sum((v[k[0]:])**2)
        # O3 = np.sum((v[k[0]:])**3)
        # h0 = 1 - (2*O1*O3)/(3*(O2**2))
        # c_confidence = stats.norm.ppf(self.confidence)
        # SPE_limit = O1*((h0*c_confidence*((2*O2)**0.5)/O1 + 1 + O2*h0*(h0-1)/(O1**2))**(1/h0))
        # self.T2_kzx = T_limit
        # self.SPE_kzx = SPE_limit
        self.p_k = p_k
        self.v_I = v_I
        self.get_statistics(data_original,is_train=True)
        return v_I, p_k,v,P,k

    def transform(self, data_original):
        self.test_data = data_original
        # data_nor = (data_original - self.data_mean)/(self.data_std)
        data_nor = self.sc.transform(data_original)
        feature = np.dot(data_nor,self.p_k)
        return feature
    
    def calculate_T2(self,data_in):
        # test_data_nor = ((data_in - self.data_mean)/self.data_std).reshape(len(data_in),1)
        # import ipdb;ipdb.set_trace()
        test_data_nor = self.sc.transform(data_in.reshape(1,-1)).reshape(-1,1)
        
        T_count = np.dot(np.dot((np.dot((np.dot(test_data_nor.T,self.p_k)), self.v_I)), self.p_k.T),test_data_nor)
        return T_count
    
    def calculate_SPE(self,data_in):
        # test_data_nor = ((data_in - self.data_mean)/self.data_std).reshape(len(data_in),1)
        test_data_nor = self.sc.transform(data_in.reshape(1,-1)).reshape(-1,1)
        I = np.eye(len(data_in))
        Q_count = np.dot(np.dot((I - np.dot(self.p_k, self.p_k.T)), test_data_nor).T,np.dot((I - np.dot(self.p_k, self.p_k.T)), test_data_nor))
        return Q_count

    def get_statistics(self, raw_test_data, is_train=False):

        t_total = []
        q_total = []
        for x in range(np.shape(raw_test_data)[0]):
            data_in = raw_test_data[x,:]
            t = self.calculate_T2(data_in)
            q = self.calculate_SPE(data_in)
            t_total.append(t[0,0])
            q_total.append(q[0,0])
        T2 = np.array(t_total)
        SPE = np.array(q_total)

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