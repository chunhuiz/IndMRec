# from scipy.linalg import expm
import scipy.linalg
# import scipy.stats as stats
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.font_manager import FontProperties
# import math
# from scipy import stats
from utils_tools import find_kde, calculate_statistics
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['simhei']
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False

class SFA:
    """
    Slow Feature Analysis (SFA).
    """
    def __init__(self,confidence,PCs):  # requires prior normalization
        self.confidence = confidence
        self.sf_num = PCs

        self.EB = None
        self.EZ = None
        self.Diff_Data = None
        self.W = None
        self.eig_values = None
        self.org_data = None

        self.Td_kzx = None
        self.Te_kzx = None
        self.Sd_kzx = None
        self.Se_kzx = None

    def get_EB(self, data):
        """
        Covariance matrix of raw data.
        :param data: raw data, shape samples x variables
        """
        return np.cov(data.T)

    def get_ZB(self, diff_data):
        """
        Covariance matrix of first-order differences.
        :param diff_data: differenced data, shape samples x variables
        """
        return np.cov(diff_data.T)

    def get_diff_data(self, data, axis=0):
        """ 
        First-order difference along axis (default: samples).
        :param data: raw data, shape samples x variables
        :param axis: difference axis, usually 0
        :return: differenced data, shape samples x variables
        """
        diff1 = np.diff(data, axis=axis)
        diff2 = data[0, :] - data[-1, :]
        diff = np.concatenate((diff2[np.newaxis,:], diff1), axis=0)
        return np.array(diff,dtype='float32')

    def decide_slow_or_fast(self, data, feature):
        """Determine the number of slow features.
        Input:
         data (samples x variables): raw data.
         feature: slow features from SFA.
        Output:
         sf_num: number of slow features."""
        data_diff = self.get_diff_data(data, 0)
        delta_data_criterion = np.diag(data_diff.T@data_diff)/np.diag(data.T@data)
        index = int(np.floor(np.size(data,1)*0.9))
        slow_fast_criterion = np.sort(delta_data_criterion)
        slow_fast_criterion = slow_fast_criterion[index]
        T = feature
        S = self.get_diff_data(T, 0)
        feature_value = np.diag(S.T@S)/np.diag(T.T@T)
        self.sf_num = sum(feature_value-slow_fast_criterion<=0)
        return self.sf_num

    def fit(self, data_original,dataset_name):
        """
        Fit SFA on training data.
        :param data_original: training data, shape samples x variables
        :return: sorted eigenvalues; number of slow features
        """
        raw_data = data_original.copy()
        self.data_mean = np.mean(data_original,0)
        self.data_std = np.std(data_original,0)
        data_original = (data_original - self.data_mean)/(self.data_std)

        self.org_data = data_original
        self.EB = self.get_EB(self.org_data)
        self.Diff_Data = self.get_diff_data(self.org_data, axis=0)
        self.EZ = self.get_ZB(self.Diff_Data)
        eig_values, eig_vector = scipy.linalg.eig(self.EZ, self.EB)
        sorted_indices = np.argsort(eig_values)
        self.W = eig_vector[:, sorted_indices] 
        self.W = self.W / np.std(self.org_data@self.W, axis=0)
        feature = data_original@self.W
        # self.decide_slow_or_fast(data_original, feature)
        # self.sf_num
        self.eig_values = eig_values[sorted_indices]
        self.get_statistics(raw_data,is_train=True)  # control limits from training data
        return self.eig_values, self.sf_num


    def transform(self, data_original):
        """
        Project data to slow features.
        :param data_original: raw data, shape samples x variables
        :return: slow features, shape samples x features
        """
        data_nor = (data_original - self.data_mean)/(self.data_std)
        feature = data_nor@self.W
        return feature
    
    def get_statistics(self, raw_test_data, is_train=False):
        """
        Compute monitoring statistics.
        Input feature: extracted features, shape samples x features
        Input cov: covariance matrix (required for test, optional for train)
        Output statistics: list of statistic vectors, each shape samples x 1
        Output kzx: control limits, shape 2 x 1
        """
        data_nor = (raw_test_data - self.data_mean)/(self.data_std)
        feature = data_nor@self.W
        Td = calculate_statistics(feature[:, :self.sf_num], type='SPE')
        Te = calculate_statistics(feature[:, self.sf_num:], type='SPE')

        # diff_feature = self.get_diff_data(feature)
        # cov_diff_feature = np.cov(diff_feature.T)
        # Sd = calculate_statistics(diff_feature[:, :self.sf_num], cov=cov_diff_feature[:self.sf_num,:self.sf_num], type='T2')
        # Se = calculate_statistics(diff_feature[:, self.sf_num:], cov=cov_diff_feature[self.sf_num:,self.sf_num:], type='T2')
        if is_train:
            # Estimate control limits from training statistics via KDE
            self.Td_kzx = find_kde(Td, self.confidence)
            self.Te_kzx = find_kde(Te, self.confidence)
            # self.Sd_kzx = find_kde(Sd, confidence)
            # self.Se_kzx = find_kde(Se, confidence)
        # return [Td, Te, Sd, Se], [self.Td_kzx, self.Te_kzx, self.Sd_kzx, self.Se_kzx]
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
        plt.suptitle('SFA:'+path.split('.png')[0])
        plt.subplot(2,1,1)
        # plt.yscale('log')
        plt.plot(self.T2_fault[0])
        plt.ylabel('Td_test')
        plt.axhline(self.Td_kzx, c='r')
        plt.subplot(2,1,2)
        # plt.yscale('log')
        plt.plot(self.T2_fault[1])
        plt.ylabel('Te_test')
        plt.axhline(self.Te_kzx, c='r')
        plt.savefig(path,dpi=300)
        plt.close()
