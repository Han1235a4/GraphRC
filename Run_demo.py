# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:16:52 2022

@author: 1235a4

"""

import argparse

import torch
import os
import numpy as np
#from ray import tune
#from ray.tune.suggest.hyperopt import HyperOptSearch
#from ray.tune.suggest.bayesopt import BayesOptSearch
#from ray.tune.suggest.basic_variant import BasicVariantGenerator
#from ray.tune.schedulers import FIFOScheduler, ASHAScheduler, MedianStoppingRule
#from ray.tune.suggest import ConcurrencyLimiter
import json
import time
import random
import matplotlib.pyplot as plt
from GraphRC import *




if __name__ == '__main__':
    dataset='Lorenz96'
    data_catch_name=os.path.join('./libcity/cache/dataset_cache/',
                                            'GraphRC_{}.npz'.format(dataset))
    
    # data_catch_name=os.path.join('./libcity/cache/dataset_cache/',
    #                                     'GraphRC_{}_flow.npz'.format(args.dataset))
    
   
    # seed = config.get('seed', 1)
    # set_random_seed(seed)
    if os.path.exists(data_catch_name):
        cat_data = np.load(data_catch_name)
        data_train = torch.from_numpy(cat_data['data_train']).type(torch.float32)

        data_test_x = torch.from_numpy(cat_data['data_test_x']).type(torch.float32)
        data_test_y = torch.from_numpy(cat_data['data_test_y']).type(torch.float32)
        
        data_train = torch.where(torch.isnan(data_train), torch.zeros_like(data_train), data_train)
        data_test_x = torch.where(torch.isnan(data_test_x), torch.zeros_like(data_test_x), data_test_x)
        data_test_y = torch.where(torch.isnan(data_test_y), torch.zeros_like(data_test_y), data_test_y)
        Input_network_initial=cat_data['Input_newtork']
        
        


    else:

        # 加载数据集
        dataset = get_dataset(config)
        
        # 转换数据，并划分数据集
        data=torch.from_numpy(np.squeeze(dataset._load_dyna(args.dataset)))
        L_train=round((data.shape[0]-23)*0.8) #
        data_train=data[:L_train,:]
        data_test_x,data_test_y= dataset._generate_input_data(data[L_train:,:])
        data_feature = dataset.get_data_feature()
        
        ensure_dir('./libcity/cache/dataset_cache/')
        np.savez_compressed(
            data_catch_name,
            data_train=data_train,
            # data_val_x=data_val_x,
            # data_val_y=data_val_y,
            data_test_x=data_test_x,
            data_test_y=data_test_y,
            Input_newtork=data_feature['adj_mx'],
            )



    
    
    #模型输入基本性质
    N_F=1 # feature dimension of model input
    N_V=data_train.shape[1]
    L_train=data_train.shape[0]
    
    #scale function
    NUll_index_train=(torch.where(torch.sum(torch.abs(data_train)<1e-4,1)>0))[0].numpy()   
    # NUll_index_train=(torch.where(torch.sum(data_train,1)==0))[0].numpy()
    NUll_index=(data_train[:-1,:]!=0).numpy()*1.0
    for i in range(N_V):
        NUll_index[np.where(NUll_index[:,i]==0)[0]-1,i]=0
        
    DT_Mean=torch.mean(data_train)
    DT_Std=torch.std(data_train)
    
    DT_MAX=torch.max(data_train)
    DT_MIN=torch.min(data_train)
    
    

    train_data=(data_train-DT_Mean)/DT_Std
    test_data_x=(data_test_x-DT_Mean)/DT_Std
        
       
    
    #模型算法选择    
    

    L_pre=np.array([3,6,12])
    
    
    
    # 模型超参数  Lorenz96 +ATT
    # 模型算法选择    
    Method_index=2#0--normal 1--FC 2--Att    
    torch.manual_seed(1)#设置随机种子 利于结果复现
    t1=time.time()
    Dr=int(200)#池子网络模型中节点的数量
    rho_R=0.47#池子模型的参数，修正池子网络权重
    rho_I=1.816#池子模型的参数，修正池子网络权重

    #(torch.max(train_data)+torch.min(train_data))/2
    #1.5/(torch.max(train_data)-torch.min(train_data))
    b=0.692##池子模型的参数，修正池子输入--偏移量S
    delta=1.534#池子模型的参数，修正池子输入---斜率  
    density_input=0.4#输入网络的密度
    transient=round(L_train*0.1)#需要删除的池子初始状态的数量
    Att_N=int(3)
    M_K=0.5
    
    
    # # 模型超参数  Lorenz96
    # # 模型算法选择    
    # Method_index=0#0--normal 1--FC 2--Att    
    # torch.manual_seed(1)#设置随机种子 利于结果复现
    # t1=time.time()
    # Dr=int(200)#池子网络模型中节点的数量
    # rho_R=0.89#池子模型的参数，修正池子网络权重
    # rho_I=0.843#池子模型的参数，修正池子网络权重

    # #(torch.max(train_data)+torch.min(train_data))/2
    # #1.5/(torch.max(train_data)-torch.min(train_data))
    # b=0.478##池子模型的参数，修正池子输入--偏移量S
    # delta=0.80#池子模型的参数，修正池子输入---斜率  
    # density_input=0.8#输入网络的密度
    # transient=round(L_train*0.1)#需要删除的池子初始状态的数量
    # # transient=int(L_train-48)
    # Att_N=int(3)
    # M_K=0.7
   

    
    #模型初始化
    W_in=(torch.rand(N_F, Dr)*2-1)#输入矩阵
    
    #Directed line reservpir graph    
    Network_weight=torch.rand(Dr,Dr)
    
    R_network_0=np.eye(Dr,k=1)
    R_network=torch.from_numpy(R_network_0)*(Network_weight+Network_weight.T)/2
    R_network=R_network.type(torch.float32)
    
    #random input graph of model input
    Network_weight=torch.rand(N_V,N_V)*rho_I

    
    Input_network_0=Network_initial('WS',network_size=N_V,density=density_input)
    
    # Input_network_0=Input_network_initial
    
    Input_network=torch.from_numpy(Input_network_0)*(Network_weight+Network_weight.T)/2
    Input_network=Input_network.type(torch.float32)
    Input_network
    
    
    #模型建立
    index_p=0 #0 超参数为常量1*1， 1 超参数为 随机变量N_V*1
    RC_learner = reservoir_computing(Dr=Dr,N_V=N_V,N_F=N_F,rho=rho_R,delta=delta,
                                         b=b,transient=transient,index_p=index_p,Att_N=Att_N)
    
    
    R_J_index=np.ones(train_data.shape[0]-1)
    # if NUll_index_train.shape[0]>0:
    #     for i_R,i_J in enumerate(NUll_index_train):
    #         R_J_index[(i_J-1):(i_J+Dr)]=0
    R_J_index[:transient]=0
    NUll_index[np.where(R_J_index==0)[0],:]=0
    W_out,R_state_initial,Pre_train_output,R_state=RC_learner.Training_phase(train_data[:(L_train-1),:],
                                                                                  train_data[1:L_train,:],
                                                                                  R_network,Input_network,
                                                                                  W_in,
                                                                                  Method=Method_index,
                                                                                  index_method=4,
                                                                                  NUM_mask=int(N_V*0),
                                                                                  K=M_K,
                                                                                  R_J_index=R_J_index,
                                                                                  NUll_index=NUll_index)
        
    b_tem=torch.zeros(N_V)
    for iiz in range(N_V):
        R_J_index1=np.where(NUll_index[:,iiz]==1)[0]
        # R_index=np.array(list(set(R_J_index1+1) & set(R_J_index1)))
        R_index=R_J_index1
        b_tem[iiz]=torch.mean(train_data[R_index+1,iiz])-torch.mean(Pre_train_output[R_index,iiz])
     

    Loss_test=np.zeros((L_pre.shape[0],6))
    null_index=list(set((torch.where(torch.sum(torch.abs(data_test_x)<1e-4,dim=(1,2))<1))[0].numpy())& set((torch.where(torch.sum(torch.abs(data_test_y)<1e-4,dim=(1,2))<1))[0].numpy()))
    print(len(null_index))
    # null_index=np.arange(data_test_x.shape[0])
    R_state_test,_=RC_learner.Forward_B(test_data_x[null_index,:-1,:],
                              R_network,Input_network,W_in,None,Method_index,W_out)
    Pre_test_output,_ = RC_learner.Predicting_phase_B(test_data_x[null_index,-1,:],R_state_test[:,-1,:,:],
                                              R_network,Input_network,W_in,W_out,
                                              Pre_L=12,B_tem=b_tem)

    
    Preds=Pre_test_output
    Real1=data_test_y[null_index,:].numpy()
    
    

    Preds1=(Preds*DT_Std+DT_Mean).detach().numpy()
        
    
    # Preds1=(Preds1>0)*1.0*Preds1
    for zz in range(L_pre.shape[0]):
        for i in range(3):
            Loss_test[zz,i],Loss_test[zz,i+3] =Loss_cal(Preds1[:,L_pre[zz]-1,:],
                                      Real1[:,L_pre[zz]-1,:],Method=i)
    print(Loss_test)

        