import numpy as np
import argparse
from model import SVDARIMA
from utils import seed_everything,rmse,mae
import pickle as pkl
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2021, type=int, help='seed')
    parser.add_argument('--dataset', default='rt', type=str, help='dataset name')
    parser.add_argument('--p', default=2, type=int, help='p-order')
    parser.add_argument('--d', default=0, type=int, help='d-order')
    parser.add_argument('--q', default=3, type=int, help='q-order')
    parser.add_argument('--R', default=140, type=int, help='an orthogonal factor matrix of dimension N * R')
    parser.add_argument('--K', default=50, type=int, help='iterations')
    parser.add_argument('--tol', default=0.001, type=float, help='stop criterion')
    parser.add_argument('--verbose', default=0, type=int, help='verbose')
    parser.add_argument('--convergence_loss', default=False, type=bool, help='convergence_loss')
    parser.add_argument('--time_len', default=20, type=int, help='time_len')
    args = parser.parse_args()
    seed_everything(args.seed)
    qos = np.load('data/rt.npy') #t*u*s
    train = qos[:args.time_len-1, :, :]
    test = qos[args.time_len-1, :, :]
    model = SVDARIMA(train, args.p, args.d, args.q, args.R, args.K, args.tol, args.verbose, args.convergence_loss,args.seed)
    pred_list,conver_list =model.run()
    rmse_of_k = [rmse(pred,test) for pred in pred_list]
    mae_of_k = [mae(pred,test) for pred in pred_list]
    #保存 rmse_of_k,mae_of_k:list
    pkl.dump(rmse_of_k,open(f'./result/{args.dataset}/rmse_of_k_{args.time_len}_{args.R}.pkl','wb'))
    pkl.dump(mae_of_k,open(f'./result/{args.dataset}/mae_of_k_{args.time_len}_{args.R}.pkl','wb'))
    pkl.dump(conver_list, open(f'./result/{args.dataset}/conver_list{args.time_len}_{args.R}.pkl', 'wb'))


