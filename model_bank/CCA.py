import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
# from PCA import PCA
from scipy import stats
from random import choice
from sklearn.preprocessing import StandardScaler
from utils_tools import find_kde, calculate_statistics

class CVA:
    def __init__(self, p, f,confidence, cumulative=0.9):
        self.cov = None
        self.U = None
        self.S = None
        self.confidence = confidence
        self.cumulative = cumulative
        self.p = p
        self.f = f
        self.load = None
        self.Sxx_=None
        self.count = None

    def SVDdecomposition(self, Yp, Yf):
        self.sc_p = StandardScaler()
        self.sc_f = StandardScaler()
        # import ipdb;ipdb.set_trace()
        Yp_norm = self.sc_p.fit_transform(Yp)
        Yf_norm = self.sc_f.fit_transform(Yf)
        # Yp_norm = Yp
        # Yf_norm = Yf

        pn = Yp_norm.shape[0]
        Sxy = Yp_norm.T@Yf_norm/(pn - 1)

        Sxx = np.cov(Yp_norm.T)
        Syy = np.cov(Yf_norm.T)
        # import ipdb;ipdb.set_trace()
        Sxx_ = linalg.sqrtm(np.linalg.inv(Sxx)).real
        Syy_ = linalg.sqrtm(np.linalg.inv(Syy)).real
        self.Sxx_ = Sxx_
        self.Syy_  = Syy_
        M = Sxx_.T.dot(Sxy.dot(Syy_))
        U, S, V = np.linalg.svd(M, full_matrices=True)
        self.U = U
        self.S = S
        self.get_load_matrix()
        self.A = self.load.T@self.Sxx_
        # import ipdb;ipdb.set_trace()
        self.P = (np.eye(self.load.shape[0]) - self.load@self.load.T)@self.Syy_


    def fit(self, data,dataset_name):
        Yp,Yf = self.hankelpf(data,self.p,self.f,is_test=False)
        self.SVDdecomposition(Yp,Yf)

        self.get_statistics(raw_test_data=data,is_train=True)
        # import ipdb;ipdb.set_trace()
        return self.U, self.S
    
    def get_statistics(self, raw_test_data, is_train=False):
        """
        Compute monitoring statistics.
        Input feature: extracted features, shape samples x features
        Input cov: covariance matrix (required for test, optional for train)
        Output statistics: list of statistic vectors, each shape samples x 1
        Output kzx: control limits, shape 2 x 1
        """
        Yp, _ = self.hankelpf(raw_test_data,self.p,self.f,is_test= not is_train)
        data_nor = self.sc_p.transform(Yp)
        # import ipdb;ipdb.set_trace()
        
        feature = data_nor@self.A.T
        e_feature = data_nor@self.P
        Td = calculate_statistics(feature[:, :], type='SPE')
        Te = calculate_statistics(e_feature[:, :], type='SPE')
        # import ipdb;ipdb.set_trace()
        if is_train:
            # Estimate control limits from training statistics via KDE
            self.Td_kzx = find_kde(Td, self.confidence)
            self.Te_kzx = find_kde(Te, self.confidence)
        return [Td, Te], [self.Td_kzx, self.Te_kzx]
    
    def score(self,test_data,fault_start,fault_end,overall=False):
        T2_fault, _ = self.get_statistics(test_data)
        T2_kzx = [self.Td_kzx, self.Te_kzx]
        self.T2_fault = T2_fault
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
        plt.suptitle('CCA:'+path.split('.png')[0])
        plt.subplot(2,1,1)
        # plt.yscale('log')
        plt.plot(self.T2_fault[0])
        plt.ylabel('T2_test')
        plt.axhline(self.Td_kzx, c='r')
        plt.subplot(2,1,2)
        # plt.yscale('log')
        plt.plot(self.T2_fault[1])
        plt.ylabel('SPE_test')
        plt.axhline(self.Te_kzx, c='r')
        plt.savefig(path,dpi=300)
        plt.close()

    def hankelpf(self,y,p,f,is_test=False):
        """
        Create past and future Hankel matrices for given observed data.
        
        Parameters:
        y: 2D NumPy array of observed data with shape (N, ny), where N is the 
        number of observation points and ny is the number of variables.
        p: the number of past observations.
        f: the number of future observations.
        
        Returns:
        Yp: the past Hankel matrix with shape (M, p*ny), where M = N - p - f.
        Yf: the future Hankel matrix with shape (M, f*ny).
        """
        N, ny = y.shape  # Total number of observations and the number of variables

        if is_test == True:
            f = 0

        M = N - p - f  # Determine the number of columns for the Hankel matrices
        
        # Check if p and f are valid
        if M <= 0:
            raise ValueError("The number of past observations (p) and future observations (f) are too large for the size of the observed data.")
        
        # Construct the past and future Hankel matrices
        Yp = np.empty((p*ny, M))
        Yf = np.empty((f*ny, M))
        
        for m in range(M):
            # import ipdb;ipdb.set_trace()
            Yp[:, m] = y[m:m+p].flatten()  # Column-wise flattening
            if not is_test:
                Yf[:, m] = y[m+p:m+p+f].flatten()
        # import ipdb;ipdb.set_trace()

        return Yp.T, Yf.T


    def get_load_matrix(self, num=None):
        limit = self.cumulative
        if num is not None:
            count = num
            self.load = self.U[:, :count]
            self.count = count
            # print('The num of eigen values: ', count)
            # print('The contribution is: ', np.sum(self.S[:count]) / np.sum(self.S))
            return count, self.load.T
        count = 1  # number of eigenvalues to retain
        while np.sum(self.S[:count]) / np.sum(self.S) < limit:
            count = count + 1
        # print(count)
        self.load = self.U[:, :count]
        self.count = count
        # print('The num of eigen values: ', count)
        # print('The contribution is: ', np.sum(self.S[:count]) / np.sum(self.S))
        return count, self.load.T