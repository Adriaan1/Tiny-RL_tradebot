#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:17:21 2021

@author: zhangwenyong
"""
import numpy as np
import pandas as pd

class stock_env:
    
    def __init__(self, data, history_t = 8):
        self.data = data
        self.data['position'] = 0
        self.price = self.data['close'].values
        self.history_t = history_t
        #self.reset()
        self.init_money = 10000000
        self.buy_rate = 0.0003  # 买入费率
        self.buy_min = 5  # 最小买入费率
        self.sell_rate = 0.0003  # 卖出费率
        self.sell_min = 5  # 最大买入费率
        self.stamp_duty = 0.001  # 印花税
        
        self.stock_value = 0 
        self.market_value = 0 
        self.states_sell = []
        self.states_buy = []
    def reset(self):
        self.t = 0
        self.data['position'] = 0
        self.done = False
        self.hold_money = self.init_money
        self.buy_num = 0 # 买入数量
        self.hold_num = 0 # 持有股票数量
        self.stock_value = 0 # 股票账户价值
        self.market_value = 0 # 总市值（加上现金）= 股票账户价值+现金
        self.last_value = self.init_money # 上一天市值
        self.total_profit = 0 # 总盈利
        self.reward = 0 # 收益
        
        self.states_sell = [] #卖股票时间
        self.states_buy = [] #买股票时间
        
        self.profit_rate_account = [] # 账号盈利
        self.profit_rate_stock = [] # 股票波动情况
        
        #self.profits = 0 #cash value
        #self.positions = []
        #self.position_value = 0
        self.history_pos = [0 for _ in range(self.history_t)]
        self.history_ret = [0 for _ in range(self.history_t)]
        #self.cash_values = []
        return [self.market_value] + self.history_ret # obs
    
    def step(self, act):
        #reward = 0
        old_sharpe = self.get_sharpe()
        self.data.iloc[self.t,:]['position'] = act - 1
        self.history_pos.pop(0)
        self.history_pos.append(act - 1)
        
        #self.reward = (self.price[self.t + 1] - self.price[self.t]) / (self.price[self.t]+0.0001)
        # act = 0: sell, 1: hold, 2: buy
        if act == 2 and self.hold_money >= self.price[self.t]*100 and self.t < (self.price.shape[0] - self.history_t//2):
            self.buy_num = self.hold_money//self.price[self.t]//100 #买入手数
            self.buy_num = self.buy_num*100
            
            self.hold_num += self.buy_num
            self.stock_value += self.price[self.t]*self.buy_num
            self.hold_money = self.hold_money - self.price[self.t]*self.buy_num 
            self.states_buy.append(self.t)
            reward = self.reward
            #print('day:%d, buy price:%f, buy num:%d, hold num:%d, hold money:%.3f'% \
            #              (self.t, self.price[self.t], self.buy_num, self.hold_num, self.hold_money))
            #self.positions.append(self.data.iloc[self.t,:]['close'])
        
        elif act == 0 and self.hold_num > 0: #sell
            sell_num = self.hold_num
            tmp_money = sell_num * self.price[self.t]
            self.hold_money = self.hold_money + tmp_money
            self.hold_num = 0
            self.stock_value = 0
            #reward = -1.0*self.reward
            self.states_sell.append(self.t)
            #print('day:%d, sell price:%f, total balance %f,'
             #       % (self.t, self.price[self.t], self.hold_money))
        
        elif act == 1 and self.hold_num > 0:  #hold
            self.buy_num = 0
            #reward = self.reward
            
        else:
            self.buy_num = 0
            #reward = 1
        self.stock_value = self.price[self.t] * self.hold_num 
        self.market_value = self.stock_value + self.hold_money 
        self.total_profit = self.market_value - self.init_money
        #self.reward = (self.market_value - self.last_value)/(self.last_value+0.0001)
        
        
        
        '''
        if np.abs(reward)<=0.015:
            self.reward = reward * 0.2
        elif np.abs(reward)<=0.03:
            self.reward = reward * 0.7
        elif np.abs(reward)>=0.05:
            if reward < 0 :
                self.reward = (reward+0.05) * 0.1 - 0.05
            else:
                self.reward = (reward-0.05) * 0.1 + 0.05
        '''
        mkt_pact = (self.market_value - self.last_value)/self.last_value
        
        self.last_value = self.market_value
        
        self.profit_rate_account.append((self.market_value - self.init_money) / self.init_money)
        self.profit_rate_stock.append((self.price[self.t] - self.price[0]) / self.price[0])
        self.t += 1
        
        self.history_ret.pop(0)
        self.history_ret.append((self.price[self.t] - \
                            self.price[self.t-1])/(self.price[self.t-1]+0.0001))
        
        #reward = self.reward
        reward = self.get_sharpe() / old_sharpe - 1 
        reward = np.nan_to_num(reward)
        reward = np.clip(reward,-1,1)
        
        s_ = [mkt_pact] + self.history_ret
        self.done = True if self.t == self.price.shape[0] else False
        #print('hold:{0}, buy:{1}'.format(self.hold_num,self.buy_num) ) 
        #print('reward:{0}, stock: {1}, cash: {2}'.format(reward, self.stock_value,self.hold_money) )
        return s_, reward, self.done # obs, reward, done
    
    
    def get_returns(self, rfr = 0.02 / 225):
        benchmark_returns = np.array(self.history_ret)
        current_position = np.array(self.history_pos)
        returns = current_position * benchmark_returns
        return returns
    
    def get_sharpe(self, rfr = 0.02 / 225):
        returns = self.get_returns(rfr)
        s = np.nanmean(returns) / (np.nanstd(returns) * np.sqrt(self.t)+0.00001)
        s = np.nan_to_num(s)
        s = np.clip(s,-1,1)
        return s
        