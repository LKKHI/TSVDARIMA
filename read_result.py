import pickle


def print_res(time_len,R,sprasity,epoch):
    with open(f'./result/rt_sprasity/mae_of_k_{time_len}_{sprasity}_{R}.pkl','rb') as f:
        mae_list = pickle.load(f)
    with open(f'./result/rt_sprasity/rmse_of_k_{time_len}_{sprasity}_{R}.pkl','rb') as f:
        rmse_list = pickle.load(f)
    print(f'k={time_len},R={R},sprasity= {sprasity},mae={mae_list[epoch]},rmse={rmse_list[epoch]}')
# def print_res(time_len,R,epoch):
#     with open(f'./result/rt/mae_of_k_{time_len}_{R}.pkl','rb') as f:
#         mae_list = pickle.load(f)
#     with open(f'./result/rt/rmse_of_k_{time_len}_{R}.pkl','rb') as f:
#         rmse_list = pickle.load(f)
#     with open(f'./result/rt/conver_list{time_len}_{R}.pkl','rb') as f:
#         conver_list = pickle.load(f)
#     print(f'k={time_len},R={R},mae={mae_list[epoch]},rmse={rmse_list[epoch]}')
for sprasity in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]:
    print_res(20,140,sprasity,20)
# print_res(20,140,20)