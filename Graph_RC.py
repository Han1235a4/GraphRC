# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 16:53:21 2022

@author: 1235a4
"""
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 17:03:06 2022

@author: 1235a4
GraphRC的基本模型
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from scipy import special, stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel,RFECV
from sklearn.linear_model import LassoCV,RidgeCV,Ridge,ElasticNetCV,orthogonal_mp,OrthogonalMatchingPursuit
from sklearn import linear_model
import networkx as nx
#use_cuda = torch.cuda.is_available()
use_cuda=False
np.random.seed(1)
torch.manual_seed(1)
#if use_cuda:
#    torch.cuda.manual_seed(2050)
def R_shuffle(node_number=0,path_length=0):
    # np.random.seed(1)

    x = [np.random.random() for i in range(path_length)]
#    x = [path_length/node_number for i in range(path_length)]
    e = [int(i / sum(x) * (node_number-path_length)) + 1 for i in x] 
    re = node_number - sum(e)
    u = [np.random.randint(0, path_length- 1) for i in range(re)] 
    
    for i in range(re):
        e[u[i]] += 1
    return e

def Network_initial(network_name=None,network_size=300,density=0.2,Depth=10,MC_configure=None):
    if network_name is "ER":
        rg=nx.erdos_renyi_graph(network_size,density,seed=1,directed=False)#ER
        R_initial=nx.adjacency_matrix(rg).toarray()
    elif network_name is "regular":
        K_number=int(network_size*density)
        rg=nx.random_graphs.random_regular_graph(K_number,network_size)#规则网络
        R_initial=nx.adjacency_matrix(rg).toarray()
    elif network_name is "WS":
        K_number=int(network_size*density)
        rg=nx.random_graphs.watts_strogatz_graph(network_size,K_number,0.3)#WS 随机网络
        R_initial=nx.adjacency_matrix(rg).toarray()
    elif network_name is "DCG":
        rg=nx.erdos_renyi_graph(network_size,density,directed=True)
        # nx.is_directed_acyclic_graph(G)
        R_initial=nx.adjacency_matrix(rg).toarray()    
    elif network_name is "DAG":
        if MC_configure is not None:
            xx=np.append(0,np.cumsum(MC_configure['number']))
            for i in range(xx.shape[0]-1):
                Reject_index=1
                for j in range(0,xx.shape[0]-1):
                    if len(MC_configure[i+1])==np.sum(np.isin(MC_configure[i+1],MC_configure[j+1]+1)):
                        Reject_index=0
                if Reject_index==1 and (MC_configure[i+1]!=1).all():
                    print("fail to construct the DAN under current Memory commnity strcutrue configuration")                    
                    Reject_index=2
            if Reject_index !=2:
                R_initial_0=np.zeros((network_size,network_size))
                for i in range(xx.shape[0]-1):
                    for j in range(xx.shape[0]-1):
                        if len(MC_configure[i+1])==np.sum(np.isin(MC_configure[i+1]+1,MC_configure[j+1])):
                            R_initial_0[xx[i]:xx[i+1],xx[j]:xx[j+1]]=1
                R_initial= np.triu(R_initial_0,1)
            else:
                R_initial=None
            
        else:
            xx=R_shuffle(network_size,Depth)
            # xx=np.array([3,4,3])
            # xx=np.array([60,60,60,60,60])
            # xx=np.array([30,30,30,30,30,30,30,30,30,30])*3
            rg = nx.complete_multipartite_graph(*tuple(xx))  
            x=nx.adjacency_matrix(rg).toarray()
            R_initial= np.triu(x,1)  
        # R_initial= np.tril(x,1)  
        Real_density=np.sum(R_initial>0)*1.0/(network_size**2)
        if Real_density>0 and density<Real_density:
            R_initial[np.random.rand(*R_initial.shape) <= (1.0-density/Real_density)] = 0 
            
        R_initial= np.triu(R_initial,1)  
    return R_initial

def Loss_cal(Prediction=None,Real_data=None,Method=0):
    #每一行都是一组单独的数据
    Loss=0
    Loss_Final=0
    mask = np.not_equal(Real_data, 0)
    mask = mask.astype('float32')
    mask /= np.mean(mask)
    if Method==1:
        Loss = np.square(np.subtract(Prediction, Real_data)).astype('float32')
        Loss = np.nan_to_num(Loss * mask)
        Loss_int=np.sqrt(np.mean(Loss,1))#每行数据的 RMSE
        Loss_Final=np.median(Loss_int)
    #MAE
    elif Method==0:
        Loss = np.abs(np.subtract(Prediction, Real_data)).astype('float32')
        Loss = np.nan_to_num(Loss * mask)
        Loss_int=np.mean(Loss,1)#每行数据的 MAE
        Loss_Final=np.median(Loss_int)
    #MAPE
    else:
        Loss = np.abs(np.divide(np.subtract(Prediction, Real_data).astype('float32')
                                , Real_data))
        Loss= np.nan_to_num(mask * Loss)
        Loss_int=np.mean(Loss,1)#每行数据的 MAPE
        Loss_Final=np.median(Loss_int)*100
    return Loss_Final

def traing_Wout(train_data,R_state,index=0,k=0.8):
    W_out= torch.zeros(R_state.shape[1], train_data.shape[1])
    W_out = W_out.cuda() if use_cuda else W_out
    if index==0:
        W_out=torch.mm(torch.pinverse(R_state),train_data)#Dr*N
    else:         
        #交叉验证的lasso
        alphas = 10**np.linspace(-10,10,100)
        if index==1:
            alphas = 10**np.linspace(-4,3,15)
            base_cv=LassoCV(alphas = alphas, fit_intercept=False)
            anova_filter=SelectFromModel(base_cv)
        #交叉验证的ridge
        if index==2:
            alphas = 10**np.linspace(-4,3,15)
            base_cv = RidgeCV(alphas = alphas,fit_intercept=False)
            anova_filter=SelectFromModel(base_cv)
        #elasticNETCV 交叉验证
        if index==3:
            alphas = 10**np.linspace(-4,3,15)
            base_cv=ElasticNetCV(alphas = alphas,cv = 20,fit_intercept=False)
            anova_filter=SelectFromModel(base_cv)
        
        #前几个最好的变量,或者只考虑前百分之多少的特征
        if index==4:
            if int(R_state.shape[1]*k)>1:
                anova_filter = SelectKBest(f_regression, k=int(R_state.shape[1]*k))#int(self.n_reservoir*0.8 ))#k_number)
            else:
                anova_filter = SelectKBest(f_regression, k=int(1))
        #递归特征消除 Recursive feature elimination (RFE)
        if index==5:
            base=linear_model.LinearRegression(fit_intercept=True)
            anova_filter = RFECV(base)
        #正交匹配追踪
        if index==6:
            base_cv=OrthogonalMatchingPursuit(fit_intercept=False)
            anova_filter=SelectFromModel(base_cv)
        
        clf = Pipeline([
         ('feature_selection', anova_filter),
         ('Linearregression', linear_model.LinearRegression(fit_intercept=False))
         ])
        for X_i in range(train_data.shape[1]):
            clf.fit(R_state,train_data[:, X_i])
            W_out.data[clf.named_steps['feature_selection'].get_support(),X_i]=torch.from_numpy(clf.named_steps['Linearregression'].coef_)
            # print(clf.named_steps['feature_selection'].estimator_.alpha_)
    return W_out
# reservoir computing 模型--前向传播模型。
class reservoir_computing(nn.Module):
    def __init__(self,Dr=300,N_V=40,N_F=1,rho=1,delta=0.1,b=0,transient=1000,index_p=0,Att_N=2,input_window=12,output_window=12):
        """
        L: 输入时间序列的长度
        N: 节点时间序列的维度
        N_V: 输入数据节点的数量
        Dr: 池子网络的节点数量
        rho: 池子网络矩阵的权重调整参数，一般的RC网络中为该邻接矩阵的奇异值。
        delta： input-to-reservoir 矩阵的尺度参数
        b： 储备池输入方程里面的偏移量
        transient: 待删除的池子状态变量的
        Att_N: attention filter 的数量
        index_p: 模型超参数是否为随机变量
        """
        super().__init__()#对继承自父类的属性进行初始化，这里的父类指的是nn.module
        self.N_V=N_V
        self.Dr=Dr
#        self.rho=rho
#        self.delta=delta
#        self.b=b
        self.transient=transient
        self.N_F=N_F
        self.Att_N=Att_N
        self.Input_window=input_window
        self.Output_window=output_window
        self.x_offsets = np.sort(np.concatenate((np.arange(-self.Input_window + 1, 1, 1),)))
        self.y_offsets = np.sort(np.arange(1, self.Output_window + 1, 1))
#        self.index_p=0
        # 在train 和 foward 里面不进行训练 所以为detach 而在
#        self.delta_0=delta
#        self.rho_0=rho
#        self.b_0=b
        
        if index_p==0:
            self.delta=torch.rand(1).fill_(delta)
            self.rho=torch.rand(1).fill_(rho)
            self.b=torch.rand(1).fill_(b)
        else:
            self.delta_1=Parameter(torch.rand(self.Dr,1).fill_(delta))
            self.rho_1=Parameter(torch.rand(self.Dr,1).fill_(rho))
            self.b_1=Parameter(torch.rand(self.Dr,1).fill_(b))
        # torch.manual_seed(1)

            
    def Forward(self, train_data, R_network,Input_network,W_in,R_state_initial=None,Method=0,W_out=None):
        """
        Constructor
        :train_data 输入训练 验证 测试的 数据集合L*N*F(L 为数据的长度, N 数据网络的节点数量,F 为节点信号的维度)
        :R_network: 池子网络邻接矩阵 Dr*Dr （Dr 为储备池网络的节点数量）
        :W_in: 输入变换的矩阵 F*Dr
        :R_state: 池子网络的状态,L*N*Dr
        Method:输出层使用模型 0--normal 1--Atten
        
        """

        L_T=train_data.shape[0]
        R_state = torch.zeros(L_T, self.N_V,self.Dr)
        if R_state_initial is not None:
            R_state[0,:]=R_state_initial            
#        W_out= torch.zeros(self.N_V,self.Dr)
        Pre_train_output=torch.zeros(L_T,self.N_V)
#        Pre_train_output= torch.zeros(self.L-self.transient, self.N)
        if use_cuda:
            R_state = R_state.cuda()
            W_out=W_out.cuda()
#            Pre_train_output=Pre_train_output.cuda()

        R_state[0,:]=(torch.tanh(self.delta*torch.mm(train_data[0,:].view(-1,1),W_in)
                         +torch.mm(Input_network.T,self.rho*torch.mm(R_state[0,:],R_network))
                         +self.b))         #R_state[0,:] 相当于-1
        if Method==0:
            for i in range(1,(L_T)):
                R_state[i,:]=(torch.tanh(self.delta*torch.mm(train_data[i,:].view(-1,1),W_in)
                                         +torch.mm(Input_network.T,self.rho*torch.mm(R_state[i-1,:],R_network))
                                         +self.b))
                
            if W_out is not None:
                for i in range(self.N_V):
                    Pre_train_output[:,i]=torch.mm(R_state[:,i,:],W_out[i,:].view(-1,1)).view(L_T)
                    
        if Method==2:
            R_state_A= torch.zeros(L_T, self.Att_N*self.Dr)  # 注意向量
            R_state_A[0,:]=torch.mm(self.R_Mask,R_state[0,:]).reshape(-1)
            for i in range(1,(L_T)):
                R_state[i,:]=(torch.tanh(self.delta*torch.mm(train_data[i,:].view(-1,1),W_in)
                                         +torch.mm(Input_network.T,self.rho*torch.mm(R_state[i-1,:],R_network))
                                         +self.b))
                R_state_A[i,:]=torch.mm(self.R_Mask,R_state[i,:]).reshape(-1)
                
            if W_out is not None:
                for i in range(self.N_V):
                    R_state_Combine=torch.cat((R_state[:,i,:],R_state_A),1) #单个输入节点对应的储备池向量加上 注意力向量
                    Pre_train_output[:,i]=torch.mm(R_state_Combine,W_out[i,:].view(-1,1)).view(L_T)
        
            
        return R_state,Pre_train_output
    
    def Training_phase(self, train_data, test_data, R_network,Input_network,W_in,
                       Method=0,
                       index_method=0,
                       NUM_mask=5,
                       K=0,
                       R_J_index=None,
                       NUll_index=None,
                       R_mask=None):
            """
            Constructor
            Method  模型的改进方法，0--node_level; 1--Fc; 2--node_level+attention
            Index_method W_out 训练的方法
            NUM_mask 激活的attention filter 中mask的数量 
            R_J_index：正常数据的index
            Index_usr:1 使用验证集训练W_out
            """

            L_T=train_data.shape[0]

            if R_J_index is None:
                R_J_index=np.ones(L_T)
                R_J_index[:self.transient]=0
                

            R_J_index=np.where(R_J_index==1)[0]

            R_state = torch.zeros(L_T, self.N_V,self.Dr)
            # R_state=torch.rand(L_T, self.N_V,self.Dr)*2-1
            Pre_train_output=torch.zeros(L_T,self.N_V)
            self.Method=Method
            W_out= torch.zeros(self.N_V,self.Dr)
            
            if Method==2:
                # torch.manual_seed(1)
                R_Mask_0 =torch.rand(self.Att_N,self.N_V)
#                R_Mask_0 =torch.zeros(self.Att_N,self.N_V)
#                Index_att=np.random.choice(self.N_V,self.N_V,replace=False)
                for i in range(self.Att_N):
                    Index_att=np.random.choice(self.N_V,NUM_mask,replace=False)

                self.R_Mask= R_Mask_0/torch.sum(R_Mask_0,1).reshape(-1,1) # 行和为1
                
                #可删除
                if R_mask is not None:
                    self.R_Mask=R_mask

                R_state_A= torch.zeros(L_T, self.Att_N*self.Dr)
                W_out= torch.zeros(self.N_V,(self.Att_N+1)*self.Dr)
                if use_cuda:
                    self.R_Mask=self.R_Mask.cuda()
                    R_state_A=R_state_A.cuda()
            else:
                self.R_Mask=torch.zeros(self.Att_N,self.N_V)
                     
    #        Pre_train_output= torch.zeros(self.L-self.transient, self.N)
            if use_cuda:
                R_state = R_state.cuda()
                W_out=W_out.cuda()
    #            Pre_train_output=Pre_train_output.cuda()
            # R_state[0,:]=(torch.tanh(self.delta*torch.mm(train_data[0,:].view(-1,1),W_in)
            #      +torch.mm(Input_network.T,self.rho*torch.mm(R_state[0,:],R_network))
            #      +self.b))         #R_state[0,:] 相当于-1
            if Method==0:
                for i in range(1,(L_T)):
                    #原始模型
                    R_state[i,:]=(torch.tanh(self.delta*torch.mm(train_data[i,:].view(-1,1),W_in)
                                             +torch.mm(Input_network.T,self.rho*torch.mm(R_state[i-1,:],R_network))
                                             +self.b))

                for i in range(self.N_V):
                    W_out[i,:]=traing_Wout(test_data[R_J_index,i].view(-1,1),
                         R_state.data[R_J_index,i,:],index=index_method,k=K).view(self.Dr)
                for i in range(self.N_V):
                    Pre_train_output[:,i]=torch.mm(R_state[:,i,:],W_out[i,:].view(-1,1)).view(L_T)
                    
            if Method==1:                
            #FC                
                for i in range(1,(L_T)):
                    R_state[i,:]=(torch.tanh(self.delta*torch.mm(train_data[i,:].view(-1,1),W_in)
                                             +torch.mm(Input_network.T,self.rho*torch.mm(R_state[i-1,:],R_network))
                                             +self.b))
                W_out= torch.zeros(self.Dr*self.N_V,self.N_V)
                R_state_final=R_state.reshape(L_T,self.Dr*self.N_V)
                W_out=traing_Wout(test_data.data[R_J_index,:],R_state_final.data[R_J_index,:],
                                  index=index_method,k=0.8)
                Pre_train_output=torch.mm(R_state_final.data,W_out)
                Pre_train_output = Pre_train_output.cuda() if use_cuda else  Pre_train_output
                
            if Method==2:
                for i in range(1,(L_T)):
                    R_state[i,:]=(torch.tanh(self.delta*torch.mm(train_data[i,:].view(-1,1),W_in)
                                              +torch.mm(Input_network.T,self.rho*torch.mm(R_state[i-1,:],R_network))
                                              +self.b))
                    
                    R_state_A[i,:]=torch.mm(self.R_Mask,R_state[i,:]).reshape(-1)
                for i in range(self.N_V):
                    # torch.cat 0 是 按列合并， 1 是按行合并 LT*(Dr+Att_N*Dr)
                    if NUll_index is not None:
                        R_J_index1=np.where(NUll_index[:L_T,i]==1)[0]
                    else:
                        R_J_index1=R_J_index
                    R_state_Combine=torch.cat((R_state[:,i,:],R_state_A),1) 
                    W_out[i,:]=traing_Wout(test_data.data[R_J_index1,i].view(-1,1),
                          R_state_Combine.data[R_J_index1,:],index=index_method,k=K).view((self.Att_N+1)*self.Dr)
                         # R_state_Combine.data[self.transient:,:],index=index_method,k=0.5).view((self.Att_N+1)*self.Dr)
                for i in range(self.N_V):
                    R_state_Combine=torch.cat((R_state[:,i,:],R_state_A),1)
                    Pre_train_output[:,i]=torch.mm(R_state_Combine,W_out[i,:].view(-1,1)).view(L_T)
                    
                    
            return W_out,self.R_Mask,Pre_train_output,R_state
        

    def Predicting_phase(self, input_initial0,R_state_initial, R_network,Input_network, W_in,
                         W_out, Pre_L=100,B_tem=None):
        
        input_initial=input_initial0.clone().detach()
        R_state=[]
        test_output=[]
        
        if B_tem is None:
            B_tem=torch.zeros(self.N_V)
            
        if use_cuda:
            self.R_Mask=self.R_Mask.cuda()
#            R_state = torch.tensor(R_state).cuda()
#            test_output=torch.tensor(test_output).cuda()
            input_initial=input_initial.cuda()
            self.delta=self.delta.cuda()
            self.rho=self.rho.cuda()
            self.b=self.b.cuda()
            W_out=W_out.cuda()
            Input_network=Input_network.cuda()
            W_in=W_in.cuda()
            R_network=R_network.cuda()
            R_state_initial=R_state_initial.cuda()
            
        if self.Method==0:
            #Node_Level
            for i in range(Pre_L):            
                R_state_initial=(torch.tanh(self.delta*torch.mm(input_initial.view(-1,1),W_in)
                                 +torch.mm(Input_network.T,self.rho*torch.mm(R_state_initial,R_network))
                                 +self.b))
                
                for ii in range(self.N_V):
                    input_initial[ii]=torch.mm(R_state_initial[ii,:].view(1,-1),W_out[ii,:].view(-1,1))+B_tem[ii]
                R_state.append(R_state_initial)
                input_initial_final=input_initial.clone().view(1,-1)
                test_output.append(input_initial_final)
                
            R_state_final=torch.stack(R_state, dim=0).reshape(Pre_L,self.N_V,self.Dr)
                
        if self.Method==1:
        #FC
            for i in range(Pre_L):  
                R_state_initial=(torch.tanh(self.delta*torch.mm(input_initial.view(-1,1),W_in)
                                     +torch.mm(Input_network.T,self.rho*torch.mm(R_state_initial,R_network))
                                     +self.b))
                input_initial=torch.mm(R_state_initial.view(1,-1),W_out)
                R_state.append(R_state_initial)
                input_initial_final=input_initial.clone().view(1,-1)
                test_output.append(input_initial_final)
                
            R_state_final=torch.stack(R_state, dim=0).reshape(Pre_L,self.N_V,self.Dr)
                
        if self.Method==2:
         #Node_Level+attention
            for i in range(Pre_L):
                R_state_initial=(torch.tanh(self.delta*torch.mm(input_initial.view(-1,1),W_in)
                                     +torch.mm(Input_network.T,self.rho*torch.mm(R_state_initial,R_network))
                                     +self.b))
                
                R_state_A_initial=torch.mm(self.R_Mask,R_state_initial).reshape(-1)
                R_state_Combine_1=[]
                for ii in range(self.N_V):
                    R_state_Combine=torch.cat((R_state_initial[ii,:],R_state_A_initial),0)
                    input_initial[ii]=torch.mm(R_state_Combine.view(1,-1),W_out[ii,:].view(-1,1))+B_tem[ii]
                    R_state_Combine_1.append(R_state_Combine.view(1,-1))

                R_state.append(torch.stack(R_state_Combine_1, dim=1).reshape(self.N_V,-1))
                input_initial_final=input_initial.clone().view(1,-1)
                test_output.append(input_initial_final)
                
            R_state_final=torch.stack(R_state, dim=0).reshape(Pre_L,self.N_V,(self.Att_N+1)*self.Dr)
            
        test_output_final=torch.stack(test_output, dim=0).reshape(Pre_L,self.N_V)  
        
        return test_output_final,R_state_final
    
    def Training_phase_B(self, train_data, test_data, R_network,Input_network,W_in,
                       Method=0,
                       index_method=0,
                       NUM_mask=5,
                       K=0,
                       R_J_index=None):
        
        B_T=train_data.shape[0]
        L_T=train_data.shape[1]
        R_state = torch.zeros(B_T,L_T, self.N_V,self.Dr)
        
        R_J_index=np.where(R_J_index==1)[0]

        R_state = torch.zeros(B_T,L_T, self.N_V,self.Dr)
        Pre_train_output=torch.zeros(B_T,self.N_V)
        
        self.Method=Method
        W_out= torch.zeros(self.N_V,self.Dr)
            
        if Method==2:
            # torch.manual_seed(1)
            R_Mask_0 =torch.rand(self.Att_N,self.N_V)

            for i in range(self.Att_N):
                Index_att=np.random.choice(self.N_V,NUM_mask,replace=False)

            self.R_Mask= R_Mask_0/torch.sum(R_Mask_0,1).reshape(-1,1) # 行和为1
            

            R_state_A= torch.zeros(B_T,L_T, self.Att_N*self.Dr)
            W_out= torch.zeros(self.N_V,(self.Att_N+1)*self.Dr)
            if use_cuda:
                self.R_Mask=self.R_Mask.cuda()
                R_state_A=R_state_A.cuda()
        else:
            self.R_Mask=torch.zeros(self.Att_N,self.N_V)
            
       
        
        if use_cuda:
            R_state = R_state.cuda()
            W_out=W_out.cuda()
#            Pre_train_output=Pre_train_output.cuda()

        R_state[:,0,:]=(torch.tanh(self.delta*torch.matmul(train_data[:,0,:].unsqueeze(-1),W_in)
                         +torch.matmul(Input_network.T,self.rho*torch.matmul(R_state[:,0,:],R_network))
                         +self.b))         #R_state[0,:] 相当于-1
        if Method==0:
            for i in range(1,(L_T)):
                R_state[:,i,:]=(torch.tanh(self.delta*torch.matmul(train_data[:,i,:].unsqueeze(-1),W_in)
                                         +torch.matmul(Input_network.T,self.rho*torch.matmul(R_state[:,i-1,:],R_network))
                                         +self.b))
                
            for i in range(self.N_V):
                W_out[i,:]=traing_Wout(test_data[:,i].view(-1,1),
                         R_state.data[:,-1,i,:],index=index_method,k=K).view(self.Dr)
                
            for i in range(self.N_V):
                Pre_train_output[:,i]=torch.matmul(R_state.data[:,-1,i,:],W_out[i,:].view(-1,1)).squeeze(-1)
                    
        if Method==2:
            R_state_A= torch.zeros(B_T,L_T, self.Att_N*self.Dr)  # 注意向量
            R_state_A[:,0,:]=torch.matmul(self.R_Mask,R_state[:,0,:]).reshape(B_T,-1)
            for i in range(1,(L_T)):
                R_state[:,i,:]=(torch.tanh(self.delta*torch.matmul(train_data[:,i,:].unsqueeze(-1),W_in)
                                         +torch.matmul(Input_network.T,self.rho*torch.matmul(R_state[:,i-1,:],R_network))
                                         +self.b))
                R_state_A[:,i,:]=torch.matmul(self.R_Mask,R_state[:,i,:]).reshape(B_T,-1)
            for i in range(self.N_V):

                R_state_Combine=torch.cat((R_state[:,-1,i,:],R_state_A[:,-1,:]),1)
                W_out[i,:]=traing_Wout(test_data[R_J_index,i].view(-1,1),
                      R_state_Combine.data[R_J_index,:],index=index_method,k=K).view((self.Att_N+1)*self.Dr)
            
            for i in range(self.N_V):
                R_state_Combine=torch.cat((R_state[:,-1,i,:],R_state_A[:,-1,:]),1) #单个输入节点对应的储备池向量加上 注意力向量
                Pre_train_output[:,i]=torch.matmul(R_state_Combine,W_out[i,:].view(-1,1)).squeeze(-1)
        
        return R_state,Pre_train_output,W_out
    
    
    def Forward_B(self, train_data, R_network,Input_network,W_in,R_state_initial=None,Method=0,W_out=None):
        """
        Constructor
        :train_data 输入训练 验证 测试的 数据集合L*N*F(L 为数据的长度, N 数据网络的节点数量,F 为节点信号的维度)
        :R_network: 池子网络邻接矩阵 Dr*Dr （Dr 为储备池网络的节点数量）
        :W_in: 输入变换的矩阵 F*Dr
        :R_state: 池子网络的状态,L*N*Dr
        Method:输出层使用模型 0--normal 1--Atten
        使用batch数据 （batch_size, length,input_node）
        """
        
        B_T=train_data.shape[0]
        L_T=train_data.shape[1]
        R_state = torch.zeros(B_T,L_T, self.N_V,self.Dr)
        if R_state_initial is not None:
            R_state[:,0,:]=R_state_initial            
#        W_out= torch.zeros(self.N_V,self.Dr)
        Pre_train_output=torch.zeros(B_T,L_T,self.N_V)
#        Pre_train_output= torch.zeros(self.L-self.transient, self.N)
        if use_cuda:
            R_state = R_state.cuda()
            W_out=W_out.cuda()
#            Pre_train_output=Pre_train_output.cuda()

        R_state[:,0,:]=(torch.tanh(self.delta*torch.matmul(train_data[:,0,:].unsqueeze(-1),W_in)
                         +torch.matmul(Input_network.T,self.rho*torch.matmul(R_state[:,0,:],R_network))
                         +self.b))         #R_state[0,:] 相当于-1
        if Method==0:
            for i in range(1,(L_T)):
                R_state[:,i,:]=(torch.tanh(self.delta*torch.matmul(train_data[:,i,:].unsqueeze(-1),W_in)
                                         +torch.matmul(Input_network.T,self.rho*torch.matmul(R_state[:,i-1,:],R_network))
                                         +self.b))
                
            if W_out is not None:
                for i in range(self.N_V):
                    Pre_train_output[:,:,i]=torch.matmul(R_state[:,:,i,:],W_out[i,:].view(-1,1)).squeeze(-1)
                    
        if Method==2:
            R_state_A= torch.zeros(B_T,L_T, self.Att_N*self.Dr)  # 注意向量
            R_state_A[:,0,:]=torch.matmul(self.R_Mask,R_state[:,0,:]).reshape(B_T,-1)
            for i in range(1,(L_T)):
                R_state[:,i,:]=(torch.tanh(self.delta*torch.matmul(train_data[:,i,:].unsqueeze(-1),W_in)
                                         +torch.matmul(Input_network.T,self.rho*torch.matmul(R_state[:,i-1,:],R_network))
                                         +self.b))
                R_state_A[:,i,:]=torch.matmul(self.R_Mask,R_state[:,i,:]).reshape(B_T,-1)
                
            if W_out is not None:
                for i in range(self.N_V):
                    R_state_Combine=torch.cat((R_state[:,:,i,:],R_state_A),2) #单个输入节点对应的储备池向量加上 注意力向量
                    Pre_train_output[:,:,i]=torch.matmul(R_state_Combine,W_out[i,:].view(-1,1)).squeeze(-1)
        
        return R_state,Pre_train_output
    
    
    def Predicting_phase_B(self, input_initial0,R_state_initial, R_network,Input_network, W_in,
                     W_out, Pre_L=100,B_tem=None):
    
        input_initial=input_initial0.clone().detach()
        R_state=[]
        test_output=[]
        
        if B_tem is None:
            B_tem=torch.zeros(self.N_V)
            
        if use_cuda:
            self.R_Mask=self.R_Mask.cuda()
    #            R_state = torch.tensor(R_state).cuda()
    #            test_output=torch.tensor(test_output).cuda()
            input_initial=input_initial.cuda()
            self.delta=self.delta.cuda()
            self.rho=self.rho.cuda()
            self.b=self.b.cuda()
            W_out=W_out.cuda()
            Input_network=Input_network.cuda()
            W_in=W_in.cuda()
            R_network=R_network.cuda()
            R_state_initial=R_state_initial.cuda()
            
        if self.Method==0:
            #Node_Level
            for i in range(Pre_L):            
                R_state_initial=(torch.tanh(self.delta*torch.matmul(input_initial.unsqueeze(-1),W_in)
                                 +torch.matmul(Input_network.T,self.rho*torch.matmul(R_state_initial,R_network))
                                 +self.b))
                
                for ii in range(self.N_V):
                    input_initial[:,ii]=torch.matmul(R_state_initial[:,ii,:],W_out[ii,:])+B_tem[ii]
                R_state.append(R_state_initial)
                input_initial_final=input_initial.clone()
                test_output.append(input_initial_final)
                
            R_state_final=torch.stack(R_state, dim=1)

                
        if self.Method==2:
         #Node_Level+attention
            for i in range(Pre_L):
                R_state_initial=(torch.tanh(self.delta*torch.matmul(input_initial.unsqueeze(-1),W_in)
                                     +torch.matmul(Input_network.T,self.rho*torch.matmul(R_state_initial,R_network))
                                     +self.b))
                
                R_state_A_initial=torch.matmul(self.R_Mask,R_state_initial).reshape(R_state_initial.shape[0],-1)
                R_state_Combine_1=[]
                for ii in range(self.N_V):
                    R_state_Combine=torch.cat((R_state_initial[:,ii,:],R_state_A_initial),1)
                    input_initial[:,ii]=(torch.matmul(R_state_Combine,W_out[ii,:].view(-1,1))+B_tem[ii]).squeeze(-1)
                    R_state_Combine_1.append(R_state_Combine)
    
                R_state.append(torch.stack(R_state_Combine_1, dim=1))
                input_initial_final=input_initial.clone()
                test_output.append(input_initial_final)
                
            R_state_final=torch.stack(R_state, dim=1)
            
        test_output_final=torch.stack(test_output, dim=1)
        
        return test_output_final,R_state_final
    
    
#     def Predicting_phase_R(self, input_initial0,R_state_initial, R_network,Input_network, W_in,
#                      W_out, Pre_L=100,B_tem=None,Del_index=None):
    
#         input_initial=input_initial0.clone().detach()
#         R_state=[]
#         test_output=[]
        
#         if B_tem is None:
#             B_tem=torch.zeros(self.N_V)
        
            
#         if use_cuda:
#             self.R_Mask=self.R_Mask.cuda()
# #            R_state = torch.tensor(R_state).cuda()
# #            test_output=torch.tensor(test_output).cuda()
#             input_initial=input_initial.cuda()
#             self.delta=self.delta.cuda()
#             self.rho=self.rho.cuda()
#             self.b=self.b.cuda()
#             W_out=W_out.cuda()
#             Input_network=Input_network.cuda()
#             W_in=W_in.cuda()
#             R_network=R_network.cuda()
#             R_state_initial=R_state_initial.cuda()
            
#         if self.Method==0:
#             #Node_Level
#             for i in range(Pre_L):            
#                 R_state_initial=(torch.tanh(self.delta*torch.mm(input_initial.view(-1,1),W_in)
#                                  +torch.mm(Input_network.T,self.rho*torch.mm(R_state_initial,R_network))
#                                  +self.b))
                
#                 for ii in range(self.N_V):
#                     input_initial[ii]=torch.mm(R_state_initial[ii,torch.arange(self.Dr)!=Del_index].view(1,-1),W_out[ii,:].view(-1,1))+B_tem[ii]
#                 R_state.append(R_state_initial)
#                 input_initial_final=input_initial.clone().view(1,-1)
#                 test_output.append(input_initial_final)
                
#             R_state_final=torch.stack(R_state, dim=0).reshape(Pre_L,self.N_V,self.Dr)
                
#         if self.Method==1:
#         #FC
#             for i in range(Pre_L):  
#                 R_state_initial=(torch.tanh(self.delta*torch.mm(input_initial.view(-1,1),W_in)
#                                      +torch.mm(Input_network.T,self.rho*torch.mm(R_state_initial,R_network))
#                                      +self.b))
#                 input_initial=torch.mm(R_state_initial.view(1,-1),W_out)
#                 R_state.append(R_state_initial)
#                 input_initial_final=input_initial.clone().view(1,-1)
#                 test_output.append(input_initial_final)
                
#             R_state_final=torch.stack(R_state, dim=0).reshape(Pre_L,self.N_V,self.Dr)
                
#         if self.Method==2:
#          #Node_Level+attention
#             for i in range(Pre_L):
#                 R_state_initial=(torch.tanh(self.delta*torch.mm(input_initial.view(-1,1),W_in)
#                                      +torch.mm(Input_network.T,self.rho*torch.mm(R_state_initial,R_network))
#                                      +self.b))
                
#                 R_state_A_initial=torch.mm(self.R_Mask,R_state_initial).reshape(-1)
#                 R_state_Combine_1=[]
#                 for ii in range(self.N_V):
#                     R_state_Combine=torch.cat((R_state_initial[ii,:],R_state_A_initial),0)
#                     input_initial[ii]=torch.mm(R_state_Combine.view(1,-1),W_out[ii,:].view(-1,1))+B_tem[ii]
#                     R_state_Combine_1.append(R_state_Combine.view(1,-1))

#                 R_state.append(torch.stack(R_state_Combine_1, dim=1).reshape(self.N_V,-1))
#                 input_initial_final=input_initial.clone().view(1,-1)
#                 test_output.append(input_initial_final)
                
#             R_state_final=torch.stack(R_state, dim=0).reshape(Pre_L,self.N_V,(self.Att_N+1)*self.Dr)
            
#         test_output_final=torch.stack(test_output, dim=0).reshape(Pre_L,self.N_V)  
        
#         return test_output_final,R_state_final
        