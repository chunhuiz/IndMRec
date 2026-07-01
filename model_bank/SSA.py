import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from scipy.io import loadmat
import warnings
from utils_tools import find_kde,calculate_statistics

warnings.filterwarnings("ignore")  # suppress warnings

class analytic_ssa:
    def __init__(self,confidence,PCs,win_num=20):
        self._S = []
        self.confidence = confidence
        self.PCs = PCs
        self.win_num = win_num

    def calculate(self,data):
        N = self.win_num
        sliceMu = np.matrix(np.mean(data))
        if data.shape[0] > 1:
            sliceSigma = np.matrix(data.T.dot(data))/(data.shape[0]-1)
        else:
            sliceSigma = np.matrix(data.T.dot(data))
        # import ipdb;ipdb.set_trace()

        self._S = self._S+(sliceMu.T.dot(sliceMu)+2*sliceSigma.dot((self._totalSigma.I).dot(sliceSigma)))/N
    
    def fit(self,data,dataset_name):
        N = self.win_num
        data_ori = data.copy()
        self.sc = StandardScaler()
        data = self.sc.fit_transform(data)
        data = pd.DataFrame(data)

        # population = slice_sample(data=data,N=N,length=1/N)
        population = data.copy()
        length = np.int(data.shape[0]/N)
        totalMu = np.matrix(np.mean(population))        

        totalSigma = np.matrix(population.T.dot(population))/(population.shape[0]-1)
        # if np.linalg.det(totalSigma) == 0:
        #     import ipdb;ipdb.set_trace()
        self._totalSigma = totalSigma
        self._S =  -np.matrix(totalMu.T.dot(totalMu))-2*totalSigma
        for i in range(N):
            self.calculate(data.iloc[i*length:(i+1)*length,:])
        S = self._S

        self._eigvals, self._eigvecs = eigh(S,totalSigma,eigvals_only=False)
        
        ssa_components = data.dot(self._eigvecs[:,:self.PCs])
        self.get_statistics(data_ori,is_train=True)
        return ssa_components
    
    def transform(self,data):
        data = self.sc.transform(data)
        return data.dot(self._eigvecs[:,:self.PCs])
    
    def get_statistics(self, raw_test_data, is_train=False):
        feature = self.transform(raw_test_data)
        cov = np.cov(feature.T)
        T2 = calculate_statistics(feature[:, :self.PCs],cov, type='T2')

        if is_train:
            # Estimate control limits from training statistics via KDE
            self.T2_kzx = find_kde(T2, self.confidence)

        return [T2], [self.T2_kzx]
    
    def score(self,test_data,fault_start,fault_end,overall=False):
        T2_fault,_ = self.get_statistics(test_data)
        self.T2_fault = T2_fault
        T2_kzx = [self.T2_kzx]
        # import ipdb;ipdb.set_trace()
        TP = sum(np.array(T2_fault[0][fault_start:fault_end])>=T2_kzx[0])
        TN = sum(np.array(T2_fault[0][:fault_start])<T2_kzx[0]) + \
                sum(np.array(T2_fault[0][fault_end:])<T2_kzx[0])
        FP = sum((np.array(T2_fault[0][:fault_start])>=T2_kzx[0])) + \
                sum((np.array(T2_fault[0][fault_end:])>=T2_kzx[0]))
        FN = sum((np.array(T2_fault[0][fault_start:fault_end])<T2_kzx[0]))
        
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
        plt.suptitle('SSA:'+path.split('.png')[0])
        plt.subplot(2,1,1)
        # plt.yscale('log')
        plt.plot(self.T2_fault[0])
        plt.ylabel('T2')
        plt.axhline(self.T2_kzx, c='r')

        plt.savefig(path,dpi=300)
        plt.close()

def slice_sample(data,N,length):    
    data_combine = pd.DataFrame(columns=data.columns)
    for i in range(N):
        data_combine = pd.concat([data_combine,data.sample(frac = length,replace=True,random_state=i)])
    return data_combine
    
if __name__ == "__main__":

    N = 10
    df=train_data = loadmat('../dataset_bank/TEP_fault5/d00.mat')['data'].transpose()
    df_ab=loadmat('../dataset_bank/TEP_fault5/d05_te.mat')['data'].transpose()

    ssa = analytic_ssa(0.99,5)

    s = ssa.fit(pd.DataFrame(df))
    
    s = ssa.sc.transform(df_ab)@ssa._eigvecs

    data =np.matrix(s).T
    x = np.linspace(0,data.shape[1],data.shape[1])
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(8.5, 1*data.shape[0])

    for i in range(data.shape[0]):
        axes = plt.subplot(data.shape[0],1,i+1)
        axes.set_frame_on(False) 
        axes.set_axis_off()
        plt.plot(x,data[i,:].T,color = 'black',linewidth=1)
    plt.savefig('timeseries.pdf')

    import ipdb;ipdb.set_trace()
    