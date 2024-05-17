from utils import *
import numpy as np
import copy
import tensorly as tl
class SVDARIMA(object):
    def __init__(self, qos, p, d, q, R, K, tol,  \
                 verbose, convergence_loss,seed):
        self.qos = qos
        self.u = qos.shape[1]
        self.s = qos.shape[2]
        self.t = qos.shape[0]
        self.p = p
        self.d = d
        self.q = q
        self.R = R
        self.K = K
        self.tol = tol
        self.verbose = verbose
        self.convergence_loss = convergence_loss
        self.seed = seed
    def init_epsilon(self):
        #随机初始化误差t-1,...,t-q
        eps = [np.random.random([self.u,self.R]) for _ in range(self.q)]
        return eps
    def estimate_ar_ma(self, cores, p, q):
        cores = copy.deepcopy(cores)
        alpha, beta = fit_ar_ma(cores, p, q)

        return alpha, beta
    def update_Kt(self,t,cores,VT,alph,beta,es):
        #AIRMA
        ar = np.sum([alph[i] * cores[t-i-1] for i in range(self.p)],axis=0)
        ma = np.sum([beta[i] * es[i] for i in range(self.q)],axis=0)
        return 1/2*(self.qos[t].dot(VT.T)+ar-ma) ,ar-ma
    def get_A_B(self,compress_matrix_list):
        sum = np.zeros([self.R,self.s])
        for t in range(self.p + self.q + self.d+1 , self.t):
            Kt = compress_matrix_list[t]
            Xt = self.qos[t]
            sum += Kt.T.dot(Xt)
        #svd
        A, S, BT = np.linalg.svd(sum,full_matrices=False)
        return A,BT
    def update_epsilon(self,t,arma,es,cores,beta):
        for j in range(self.q):
            index = (self.q -1 -j)%self.q #t-j
            for i in range(self.p + self.q + self.d+2,self.t):
                es_old = copy.deepcopy(es[index])
                if i == self.p + self.q + self.d:
                    es[index] = np.zeros([self.u,self.R])
                es[index] += cores[i] - arma - beta[j]*es_old
            es [index] = es[index]/((self.p + self.q + self.d +1 - self.t)/beta[j])
        return es
    def compute_convergence(self, new_U, old_U):

        new_old = [n - o for n, o in zip(new_U, old_U)]

        a = np.sum([np.sqrt(tl.tenalg.inner(e, e)) for e in new_old], axis=0)
        b = np.sum([np.sqrt(tl.tenalg.inner(e, e)) for e in new_U], axis=0)
        return a / b
    def train_iter(self,cores, VT, es):
        for t in range(self.p + self.q + self.d , self.t):
            if t == self.p + self.q + self.d:
                # Kt-p
                core, VT = trun_SVD(self.qos[t], self.R, self.seed)
                # cores =[self.qos[i].dot(VT.T) for i in range(t)]
                cores = [q.dot(VT.T) for q in self.qos]
                # cores.append(core)
            # estimate ARMA
            alpha, beta = self.estimate_ar_ma(cores, self.p, self.q)
            # update Kt
            Kt, arma = self.update_Kt(t, cores, VT, alpha, beta, es)
            # decomposition of
            cores[t]=Kt
            A, BT = self.get_A_B(cores)
            V = BT.T.dot(A.T)
            VT= V.T
            # update epsilon
            es = self.update_epsilon(t, arma, es, cores, beta)
        return VT,alpha,beta,es,cores,A

    def _tensor_reverse_diff(self, d, begin, tensors, axis):
        """
        recover original tensors from d-order difference tensors

        Arg:
            d: int, order
            begin: list, the first d elements
            tensors: list of ndarray, tensors after difference

        Return:
            re_tensors: ndarray, original tensors

        """

        re_tensors = tensors
        for i in range(1, d + 1):
            re_tensors = list(np.cumsum(np.insert(re_tensors, 0, begin[-i], axis=axis), axis=axis))

        return re_tensors
    def predect(self,cores,VT,alph,beta,es,begin_tensors=None):
        #get K T+1
        ar = np.sum([alph[i] * cores[self.t-i-1] for i in range(self.p)],axis=0)
        ma = np.sum([beta[i] * es[i] for i in range(self.q)],axis=0)
        arma = ar-ma
        # 恢复
        X_pred = arma.dot(VT)
        if self.d != 0:
            X_pred = self._tensor_reverse_diff(self.d, begin_tensors, X_pred, axis=0)
        return X_pred

    def run(self):
        qos = self.qos
        if self.d!=0:
            begin_tensors, qos = tensor_difference(self.d, qos, axis=0)
            self.qos = qos
        # the truncated SVD method
        compressed_matrix_list = []
        # for x in qos:
        #     compressed_matrix = trun_SVD(x, self.R, self.seed)
        #     compressed_matrix_list.append(compressed_matrix)
        es = self.init_epsilon()
        conver_list =[]
        pred_list = []
        es_list =[]
        cores = []
        VT = np.random.random([self.R,self.s])
        A = np.random.random([self.R, self.R])
        for k in range(self.K):
            oldA = A.copy()
            VT,alpha,beta,es,cores,A= self.train_iter(cores, VT, es)
            if self.d!=0:
                X_pred = self.predect(cores,VT,alpha,beta,es,begin_tensors)
            else:
                X_pred = self.predect(cores,VT,alpha,beta,es)

            conver = self.compute_convergence(A, oldA)
            conver_list.append(conver)
            es_result = np.sum([np.sqrt(tl.tenalg.inner(e, e)) for e in es], axis=0)/(self.q * self.u * self.R)
            es_list.append(es_result)
            pred_list.append(X_pred)
        return pred_list,conver_list


