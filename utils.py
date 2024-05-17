import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import random
import os
from scipy import linalg
def tensor_difference(d, tensors, axis):
    d_tensors = tensors
    begin_tensors = []

    for _ in range(d):
        begin_tensors.append(d_tensors[0])
        d_tensors = list(np.diff(d_tensors, axis=axis))

    return begin_tensors, d_tensors
def trun_SVD(d_tensors, K,seed=616):
    svd = TruncatedSVD(n_components=K, n_iter=10,random_state=seed)
    d_tensors = csr_matrix(d_tensors)
    svd.fit(d_tensors)
    VT = svd.components_
    #获得压缩矩阵
    compressed_matrix = svd.transform(d_tensors)
    return compressed_matrix,VT
    # return VT
def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def autocorr(Y, lag=10):
    """
    计算<Y(t), Y(t-0)>, ..., <Y(t), Y(t-lag)>
    :param Y: list [tensor1, tensor2, ..., tensorT]
    :param lag: int
    :return: array(k+1)
    """
    T = len(Y)
    r = []
    #     print("Y")
    #     print(Y)
    for l in range(lag + 1):
        product = 0
        for t in range(T):
            tl = l - t if t < l else t - l
            product += np.sum(Y[t] * Y[tl])
        r.append(product)
    return r


def fit_ar(Y, p=10):
    r = autocorr(Y, p)
    # print("auto-corr:",r)
    R = linalg.toeplitz(r[:p])
    r = r[1:]
    A = linalg.pinv(R).dot(r)
    return A


def fit_ar_ma(Y, p=10, q=1):
    # print("fit_ar_ma")
    N = len(Y)

    A = fit_ar(Y, p)
    B = [0.]
    if q > 0:
        Res = []
        for i in range(p, N):
            res = Y[i] - np.sum([a * Y[i - j] for a, j in zip(A, range(1, p + 1))], axis=0)
            Res.append(res)
        # Res = np.array(Res)
        B = fit_ar(Res, q)
    return A, B
def rmse(pred,real):
    return np.sqrt(np.mean(np.square(pred-real)))
def mae(pred,real):
    return np.mean(np.abs(pred-real))