import pandas as pd
import numpy as np
import itertools


from scipy.stats import skew, kurtosis
from scipy.stats import f_oneway
from scipy.stats import entropy

from scipy.stats import moment
from scipy.stats import normaltest


from sklearn.utils import check_array

from sklearn.decomposition import PCA as sklearn_PCA

from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from utils_tools import add_missing_values

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = np.add(array, 0.0000001, casting="unsafe")
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def Diff(li1, li2):
    """Calculate the difference of two list

    Parameters
    ----------
    li1
    li2

    Returns
    -------

    """
    return (list(set(li1) - set(li2)))


def argmaxn(w, nth):
    w = np.asarray(w).ravel()
    t = np.argsort(w)
    return t[-1 * nth]


def flatten_diagonally(x, diags=None):
    diags = np.array(diags)
    if x.shape[1] > x.shape[0]:
        diags += x.shape[1] - x.shape[0]
    n = max(x.shape)
    ndiags = 2 * n - 1
    i, j = np.indices(x.shape)
    d = np.array([])
    for ndi in range(ndiags):
        if diags != None:
            if not ndi in diags:
                continue
        d = np.concatenate((d, x[i == j + (n - 1) - ndi]))
    return d


def list_process(x, r_min=True, r_max=True, r_mean=True, r_std=True,
                 r_skew=True, r_kurtosis=True):
    """Return statistics of a list

    Parameters
    ----------
    x
    r_min
    r_max
    r_mean
    r_std
    r_skew
    r_kurtosis

    Returns
    -------

    """
    x = np.asarray(x).reshape(-1, 1)
    return_list = []

    if r_min:
        return_list.append(np.nanmin(x))

    if r_max:
        return_list.append(np.nanmax(x))

    if r_mean:
        return_list.append(np.nanmean(x))

    if r_std:
        return_list.append(np.nanstd(x))

    if r_skew:
        return_list.append(skew(x, nan_policy='omit')[0])

    if r_kurtosis:
        return_list.append(kurtosis(x, nan_policy='omit')[0])

    return return_list


def list_process_name(var):
    return [var + '_min', var + '_max', var + '_mean', var + '_std',
            var + '_skewness', var + '_kurtosis']


def generate_meta_features(X):
    """Get the meta-features of a datasets X

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        Input array

    Returns
    -------
    meta_features : numpy array of shape (1, 200)
        Meta-feature in dimension of 200

    """
    # outliers_fraction = np.count_nonzero(y) / len(y)
    # outliers_percentage = round(outliers_fraction * 100, ndigits=4)
    X = check_array(X)

    meta_vec = []
    meta_vec_names = []

    # on the sample level
    n_samples, n_features = X.shape[0], X.shape[1]

    meta_vec.append(n_samples)
    meta_vec.append(n_features)

    meta_vec_names.append('n_samples')
    meta_vec_names.append('n_features')

    sample_mean = np.mean(X)
    sample_median = np.median(X)
    sample_var = np.var(X)
    sample_min = np.min(X)
    sample_max = np.max(X)
    sample_std = np.std(X)

    q1, q25, q75, q99 = np.percentile(X, [0.01, 0.25, 0.75, 0.99])
    iqr = q75 - q25

    normalized_mean = sample_mean / sample_max
    normalized_median = sample_median / sample_max
    sample_range = sample_max - sample_min
    sample_gini = gini(X)
    med_abs_dev = np.median(np.absolute(X - sample_median))
    avg_abs_dev = np.mean(np.absolute(X - sample_mean))
    quant_coeff_disp = (q75 - q25) / (q75 + q25)
    coeff_var = sample_var / sample_mean

    outliers_15iqr = np.logical_or(
        X < (q25 - 1.5 * iqr), X > (q75 + 1.5 * iqr))
    outliers_3iqr = np.logical_or(X < (q25 - 3 * iqr), X > (q75 + 3 * iqr))
    outliers_1_99 = np.logical_or(X < q1, X > q99)
    outliers_3std = np.logical_or(X < (sample_mean - 3 * sample_std),
                                  X > (sample_mean + 3 * sample_std))

    percent_outliers_15iqr = np.sum(outliers_15iqr) / len(X)
    percent_outliers_3iqr = np.sum(outliers_3iqr) / len(X)
    percent_outliers_1_99 = np.sum(outliers_1_99) / len(X)
    percent_outliers_3std = np.sum(outliers_3std) / len(X)

    has_outliers_15iqr = np.any(outliers_15iqr).astype(int)
    has_outliers_3iqr = np.any(outliers_3iqr).astype(int)
    has_outliers_1_99 = np.any(outliers_1_99).astype(int)
    has_outliers_3std = np.any(outliers_3std).astype(int)

    meta_vec.extend(
        [sample_mean, sample_median, sample_var, sample_min, sample_max,
         sample_std,
         q1, q25, q75, q99, iqr, normalized_mean, normalized_median,
         sample_range, sample_gini,
         med_abs_dev, avg_abs_dev, quant_coeff_disp, coeff_var,
         percent_outliers_15iqr, percent_outliers_3iqr, percent_outliers_1_99,
         percent_outliers_3std,
         has_outliers_15iqr, has_outliers_3iqr, has_outliers_1_99,
         has_outliers_3std])

    meta_vec_names.extend(
        ['sample_mean', 'sample_median', 'sample_var', 'sample_min',
         'sample_max', 'sample_std',
         'q1', 'q25', 'q75', 'q99', 'iqr', 'normalized_mean',
         'normalized_median', 'sample_range', 'sample_gini',
         'med_abs_dev', 'avg_abs_dev', 'quant_coeff_disp', 'coeff_var',
         'percent_outliers_15iqr', 'percent_outliers_3iqr',
         'percent_outliers_1_99', 'percent_outliers_3std',
         'has_outliers_15iqr', 'has_outliers_3iqr', 'has_outliers_1_99',
         'has_outliers_3std'])

    correlation = np.nan_to_num(pd.DataFrame(X).corr(), nan=0)
    correlation_list = flatten_diagonally(correlation)[
                       0:int((n_features * n_features - n_features) / 2)]
    correlation_values = list_process(correlation_list)
    meta_vec.extend(correlation_values)
    meta_vec_names.extend(list_process_name('correlation'))

    covariance = np.cov(X.T)
    covariance_list = flatten_diagonally(covariance)[
                      0:int((n_features * n_features - n_features) / 2)]
    covariance_values = list_process(covariance_list)
    meta_vec.extend(covariance_values)
    meta_vec_names.extend(list_process_name('covariance'))

    # sparsity
    rep_counts = []
    for i in range(n_features):
        rep_counts.append(len(np.unique(X[:, i])))
    sparsity_list = np.asarray(rep_counts) / (n_samples)
    sparsity = list_process(sparsity_list)
    meta_vec.extend(sparsity)
    meta_vec_names.extend(list_process_name('sparsity'))

    # ANOVA p value
    p_values_list = []
    all_perm = list(itertools.combinations(list(range(n_features)), 2))
    for j in all_perm:
        p_values_list.append(f_oneway(X[:, j[0]], X[:, j[1]])[1])
    anova_p_value = list_process(np.asarray(p_values_list))
    meta_vec.extend(anova_p_value)
    meta_vec_names.extend(list_process_name('anova_p_value'))

    # entropy
    entropy_list = []
    for i in range(n_features):
        counts = pd.Series(X[:, i]).value_counts()
        entropy_list.append(entropy(counts) / n_samples)
    entropy_values = list_process(entropy_list)
    meta_vec.extend(entropy_values)
    meta_vec_names.extend(list_process_name('entropy'))
    
    ################### probability distribution shape #####################
    # Test whether a sample differs from a normal distribution.
    normality_k2, normality_p = normaltest(X) # (52,)
    is_normal_5 = (normality_p < 0.05).astype(int)
    is_normal_1 = (normality_p < 0.01).astype(int)

    meta_vec.extend(list_process(normality_p))
    meta_vec.extend(list_process(is_normal_5))
    meta_vec.extend(list_process(is_normal_1))

    meta_vec_names.extend(list_process_name('normality_p'))
    meta_vec_names.extend(list_process_name('is_normal_5'))
    meta_vec_names.extend(list_process_name('is_normal_1'))


    skewness_list = skew(X).reshape(-1, 1)
    skew_values = list_process(skewness_list)
    meta_vec.extend(skew_values)
    meta_vec_names.extend(list_process_name('skewness'))


    kurtosis_list = kurtosis(X)
    kurtosis_values = list_process(kurtosis_list)
    meta_vec.extend(kurtosis_values)
    meta_vec_names.extend(list_process_name('kurtosis'))

    moment_5 = moment(X, moment=5) # Calculate the nth moment about the mean for a sample.
    moment_6 = moment(X, moment=6)
    moment_7 = moment(X, moment=7)
    moment_8 = moment(X, moment=8)
    moment_9 = moment(X, moment=9)
    moment_10 = moment(X, moment=10)
    meta_vec.extend(list_process(moment_5))
    meta_vec.extend(list_process(moment_6))
    meta_vec.extend(list_process(moment_7))
    meta_vec.extend(list_process(moment_8))
    meta_vec.extend(list_process(moment_9))
    meta_vec.extend(list_process(moment_10))
    meta_vec_names.extend(list_process_name('moment_5'))
    meta_vec_names.extend(list_process_name('moment_6'))
    meta_vec_names.extend(list_process_name('moment_7'))
    meta_vec_names.extend(list_process_name('moment_8'))
    meta_vec_names.extend(list_process_name('moment_9'))
    meta_vec_names.extend(list_process_name('moment_10'))

    ###################### Task-related #########################
    # ! PCA explained variance and variability

    pca_transformer = sklearn_PCA(n_components=3)
    X_transform = pca_transformer.fit_transform(X)
    # first pc
    pca_fpc = list_process(X_transform[0, :], r_min=False, r_max=False,
                           r_mean=False,
                           r_std=True, r_skew=True, r_kurtosis=True)
    meta_vec.extend(pca_fpc)
    meta_vec_names.extend(
        ['first_pca_std', 'first_pca_skewness', 'first_pca_kurtosis'])
    
    meta_vec.extend(pca_transformer.explained_variance_ratio_)  # ratio to total variance
    meta_vec.extend(pca_transformer.singular_values_)  # absolute singular values
    meta_vec_names.extend(
        ['pca_expl_ratio_1', 'pca_expl_ratio_2', 'pca_expl_ratio_3'])
    meta_vec_names.extend(['pca_sv_1', 'pca_sv_2', 'pca_sv_3'])
    
    # ! SSA stationary or non-stationary
    from model_bank.SSA import analytic_ssa
    import statsmodels.tsa.stattools as ts
    ssa = analytic_ssa(confidence=0.99,PCs=10)
    # import ipdb;ipdb.set_trace()
    try:
        ssa.fit(X,'')
        non_s = (ssa.sc.transform(X)@ssa._eigvecs)[:,-3:]
        meta_vec.extend([ts.adfuller(non_s[:,0], 1, regression='c')[1],
                        ts.adfuller(non_s[:,1], 1, regression='c')[1],
                        ts.adfuller(non_s[:,2], 1, regression='c')[1],])
            # import ipdb;ipdb.set_trace()
        meta_vec.extend(ssa._eigvals[-3:].tolist())
    except Exception as e:
        # import ipdb;ipdb.set_trace()
        print(f"SSA Errors: {e}")
        meta_vec.extend([0]*6)
    meta_vec_names.extend(['ssa_adf_1','ssa_adf_2','ssa_adf_3'])
    meta_vec_names.extend(['ssa_sv_1', 'ssa_sv_2', 'ssa_sv_3'])
    
    # ! CVA Temporal correlation
    from model_bank.CCA import CVA
    cca = CVA(p=5,f=5,confidence=0.99)
    try:
        cca.fit(X,'')
        # import ipdb;ipdb.set_trace()
        meta_vec.extend([np.median(cca.S[:cca.count]),cca.S[:cca.count].mean(),cca.S[:cca.count].std()])
    except Exception as e:
        print(f"CCA Errors: {e}")
        meta_vec.extend([0]*3)
    meta_vec_names.extend(['cca_sv_median', 'cca_sv_mean', 'cca_sv_std'])

    # ! linear vs non-linear
    # variances = np.var(X, axis=0)
    # # keep columns with non-zero variance
    # non_zero_variance_columns = np.where(variances != 0)[0]
    # X_filtered = X[:,non_zero_variance_columns]
    # train_data, test_data = (X_filtered[:int(X_filtered.shape[0]*0.7),:] - X_filtered.mean(0)) / X_filtered.std(0), (X_filtered[int(X_filtered.shape[0]*0.3):,:] - X_filtered.mean(0)) / X_filtered.std(0)

    # imputer_linear = IterativeImputer(estimator=LinearRegression(), random_state=0)
    # imputer_non = IterativeImputer(estimator=RandomForestRegressor(), random_state=0)

    # imputer_linear.fit_transform(train_data)
    # imputer_non.fit_transform(train_data)

    # cur_test = test_data.copy()

    # cur_test = add_missing_values(cur_test)
    # data_imputed_linear = imputer_linear.transform(cur_test)
    # data_imputed_non = imputer_non.transform(cur_test)
    # missing_values_mask = np.isnan(cur_test)

    # true_values = test_data[missing_values_mask]
    # predicted_values_linear = data_imputed_linear[missing_values_mask]
    # predicted_values_non = data_imputed_non[missing_values_mask]
    # mse_linear = mean_squared_error(true_values, predicted_values_linear)
    # mse_non = mean_squared_error(true_values, predicted_values_non)

    # meta_vec.extend([mse_linear,mse_non,mse_linear/mse_non])
    # meta_vec_names.extend(['mse_linear', 'mse_non_linear', 'mse_non_linear_degree'])
    
    return meta_vec, meta_vec_names
