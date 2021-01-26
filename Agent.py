#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:18:34 2021

@author: zhangwenyong
"""
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Networks import Q_Network, Duel_Q_Network
import matplotlib.pyplot as plt
from ReplayMemory import ExperienceReplayMemory, PrioritizedReplayMemory, RecurrentExperienceReplayMemory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QAgent:
    def __init__(self, env, max_iter, double = False, \
                 dueling = False, memory_type = 'experienced'):
        
        self.env = env
        self.lr = 0.001
        self.epoch_num = max_iter
        self.step_max = len(self.env.data)-1
        self.memory_size = 200
        self.batch_size = 20
        self.epsilon = 0.25
        self.epsilon_decrease_rate = 0.999
        self.epsilon_min = 0.2
        self.start_reduce_epsilon = 200
        self.train_freq = 10
        self.update_q_freq = 40
        self.gamma = 0.9
        self.show_log_freq = 5
        self.double = double
        self.mem = memory_type
        if self.mem.lower() == 'experienced':
            self.memory = ExperienceReplayMemory(self.memory_size)
        elif self.mem.lower() == 'prioritized':
            self.memory = PrioritizedReplayMemory(self.memory_size)
        elif self.mem.lower() == 'recurrent':
            self.memory = RecurrentExperienceReplayMemory(self.memory_size)
        self.mse = nn.MSELoss()
        self.clip_norm = 1.0
        self.total_step = 0
        self.total_rewards = []
        self.total_losses = []
    
        self.portfolio_values = []
        
        if dueling:
            self.Q_ast = self.Q = Duel_Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)
        else:
            self.Q_ast = self.Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3) 
       
        self.Q_ast.load_state_dict(self.Q.state_dict())
        
        self.optimizer = optim.Adam(self.Q.parameters(), lr = self.lr)
        
    
    def huber(self, x):
        cond = (x.abs() < 1.0).float().detach()
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)
    
    def to_tensor(self, array):
        return torch.from_numpy(array).float().to(device)

    def to_array(self, tensor):
        return tensor.detach().numpy()
     
    def get_action(self, state):
        e = np.random.rand()
        self.Q.eval()
        if e > self.epsilon:
            state = self.to_tensor(np.array(state))
            with torch.no_grad():
               qs = self.to_array(self.Q(state).view(-1, 1))
            action = np.argmax(qs)
        else:
            action = np.random.randint(3)
        return action
            
    def step(self):
        pass
    
    def learn_onestep(self, batch):
        b_pobs, b_pact, b_reward, b_obs, b_done, indices, weights = batch
        
        self.Q.train()
        q = self.Q(self.to_tensor(b_pobs)).gather(1, self.to_tensor(b_pact.reshape(-1,1)).long()).squeeze()
        with torch.no_grad():
            if self.double:
                self.Q.eval()
                argmax_a = np.argmax(self.to_array(self.Q(self.to_tensor(b_obs))),1)
                Q_t_plus_one = self.Q_ast(self.to_tensor(b_pobs)).gather(1, self.to_tensor(argmax_a.reshape(-1,1)).long()).squeeze()
                Q_t_plus_one = self.to_array(Q_t_plus_one)
                        
            else:
                Q_t_plus_one = np.max(self.to_array(self.Q_ast(self.to_tensor(b_obs))), axis=1)
            
            Q_target = b_reward + self.gamma*Q_t_plus_one*(1- b_done)
        Q_target = self.to_tensor(Q_target)
        
        diff = Q_target - q
        if self.mem.lower() == 'prioritized':
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.huber(diff).squeeze() * weights
        else:
            loss = self.mse(q, Q_target)
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.Q.parameters(), self.clip_norm)
        self.optimizer.step()
        return loss
    
    def learn(self):
        start = time.time()
        for epoch in range(self.epoch_num):
            pobs = self.env.reset()
            done = False
            step = 0
            total_reward = 0
            total_loss = 0
            
            while not done and step<self.step_max:
                pact = self.get_action(pobs)
                obs, reward, done = self.env.step(pact)
                self.memory.push((pobs, pact, reward, obs, done))
                
                if len(self.memory.memory) == self.memory_size:
                    if self.total_step % self.train_freq == 0:
                        shuffled_memory = np.random.permutation(self.memory.memory)
                        memory_idx = range(len(shuffled_memory))
                        for i in memory_idx[::self.batch_size]:
                           batch,indices, weights = self.memory.sample(self.batch_size)
                           b_pobs = np.stack(transition[0] for transition in batch)
                           b_pact = np.stack(transition[1] for transition in batch)
                           b_reward = np.stack(transition[2] for transition in batch)
                           b_obs = np.stack(transition[3] for transition in batch)
                           b_done = np.stack(transition[4] for transition in batch)
                           batch = b_pobs, b_pact, b_reward, b_obs, b_done, indices, weights
                           
                           loss =  self.learn_onestep(batch)
                           total_loss += loss
                           
                    if self.total_step % self.update_q_freq == 0:
                           self.Q_ast.load_state_dict(self.Q.state_dict())
                           
                if self.epsilon > self.epsilon_min  and self.total_step > self.start_reduce_epsilon:    
                    self.epsilon *=self.epsilon_decrease_rate
                
                total_reward += reward
                pobs = obs
                step += 1
                self.total_step += 1
                self.portfolio_values.append(self.env.market_value)
                
            self.total_rewards.append(total_reward)
            self.total_losses.append(total_loss.item())
            
            if (epoch+1) % self.show_log_freq == 0:
               log_reward = sum(self.total_rewards[((epoch+1)-self.show_log_freq):])/self.show_log_freq
               log_loss = sum(self.total_losses[((epoch+1)-self.show_log_freq):])/self.show_log_freq
               elapsed_time = time.time()-start
               print('\t'.join(map(str, [epoch+1, self.epsilon, self.total_step, log_reward, log_loss, elapsed_time])))
            
               start = time.time()
        

    def train_eval(self):                
        plt.style.use('ggplot')
        plt.plot(self.total_losses, alpha = 0.6, color = 'blue')
        plt.title('train loss')
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.show()
        plt.plot(self.total_rewards, alpha = 0.6, color = 'yellow')
        plt.title('train loss')
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.show()
    
    def backtest(self, env, Q):
        # train
        pobs = env.reset()
        profits = []
        for _ in range(len(env.data)-1):
            pact = self.to_array(Q(self.to_tensor(np.array(pobs)).view(1,-1)))
            pact = np.argmax(pact)
            #pact = np.random.randint(3)
            obs, _, _ = env.step(pact)
            pobs = obs
            profits.append(env.market_value)
        profits = np.array(profits)
        benchmark = env.price
        # train
        
        
        PnL = np.cumprod(1 + np.diff(profits)/profits[:-1])
        benchmk = np.cumprod(1 + np.diff(benchmark)/benchmark[:-1])
         
        plt.figure(figsize=(10,7))
        plt.style.use('ggplot')
        plt.plot(PnL, alpha = 0.4, color = 'red')
        plt.plot(benchmk, alpha = 0.4, color = 'blue')
        plt.title('wealth curve')
        plt.xlabel('days')
        plt.ylabel('profit')
        plt.legend(['PnL', 'Index'])
        plt.show()
        print(env.states_buy)
        #return train_profits, test_profits
        def MaxDrawdown(return_list):
            #a = np.maximum.accumulate(return_list)
            l = np.argmax((np.maximum.accumulate(return_list) - return_list) /np.maximum.accumulate(return_list))
            k = np.argmax(return_list[:l])
            return (return_list[k] - return_list[l])/(return_list[l])
        
        from prettytable import PrettyTable
        table = PrettyTable(['performance', 'value'])
        mdd = MaxDrawdown(PnL)
        table.add_row(['MaxDrawdown', mdd.astype(np.float16)])
        t = env.data.shape[0]
        returns = np.diff(PnL)/PnL[:-1]
        anul_return = PnL[-1]**(1/t*255)-1
        anul_vol = returns.std()*(52**(1/2))
        spr = anul_return/anul_vol
        table.add_row(['Sharpe Ratio', spr.astype(np.float16)])
        #Sortino Ratio
        down_return = returns[returns<0]
        anul_downvol = down_return.std()*(52**(1/2))
        sorr = anul_return/anul_downvol
        table.add_row(['Sortino Ratio', sorr.astype(np.float16)])
        #IR
        cum_return = (PnL - benchmk)[-1]**(1/t*255)
        alpha_ = returns - np.diff(benchmk)/benchmk[:-1]
        alpha_vol = alpha_.std()*(52**(1/2))
        ir = cum_return/alpha_vol
        table.add_row(['information rate', ir.astype(np.float16)])

    
    def save_agent(self, model_name):
        torch.save(self.Q.state_dict(), model_name)
        
    def load_agent(self,model_name):
        self.Q.load_state_dict(torch.load(model_name))
        self.Q_ast.load_state_dict(torch.load(model_name))
    
'''
if __name__=='__main__':
  from stock_env import stock_env #导入股票交易环境
  data = pd.read_csv('HS300_OHLC.csv',index_col=0)
  data = data.drop(columns = ['code'])
  data = data.set_index('time') #导入股票数据
  env = stock_env(data) #定义环境
  agent = QAgent(env, 50)   #定义智能体
  agent.learn() #与环境进行交互学习
  agent.train_eval() #输出策略回测结果
        
'''        
       
