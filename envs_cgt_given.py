try:
    import gymnasium as gym
except ImportError:
    class _Gym:
        class Env:
            pass
    gym = _Gym()
import random
from itertools import combinations
import numpy as np


ALPHA = 2
BETA = 0.25
C = 1.25


class DemandResponseMarketEnv(gym.Env):
    def __init__(self, num_players, num_states=5, num_actions=5):
        super(DemandResponseMarketEnv, self).__init__()
        self.short_name = "energy"
        self.num_players = num_players
        self.num_states = num_states
        self.num_actions = num_actions
        self.cost_UB = 16
        self.reset()

    def reset(self, state=None):
        self.state = self.num_states - 1 if state is None else state
        return self.state

    # action is tuple of size num_players representing joint action
    def step(self, action):
        rewards = [
            a**2 * ALPHA - a**2 * BETA * sum(action) - a * C**self.state
            for a in action
        ]
        cost = sum(action)
        w = random.randint(0, self.num_states - 1)
        if random.random()<0.9:
            self.state = round(2 * sum(action) / len(action) + w)  # just guessing here
        else:
            self.state = w
        self.state = max(min(self.state, self.num_states - 1), 0)
        potential = (
            ALPHA * sum([a for a in action])
            - BETA * sum([a**2 for a in action])
            - BETA * sum([a1 * a2 for (a1, a2) in combinations(action, 2)])
            - self.num_players * C**self.state
        )
        return self.state, rewards, cost, potential, False


POLLUTION_FREE, POLLUTED = 0, 1
CLEAN, DIRTY = 0, 1
PROFIT_PER_ITEM = 2
TAX_BY_NUM_PLAYERS = {2: 4, 4: 8, 8: 32}


class PollutionTaxEnv(gym.Env):
    def __init__(self, num_players, num_states=2, num_actions=2):
        super(PollutionTaxEnv, self).__init__()
        self.short_name = "pollution"
        self.num_players = num_players
        self.num_states = num_states
        self.num_actions = num_actions
        self.cost_UB = 12
        self.reset()

    def reset(self, state=None):
        self.state = POLLUTION_FREE if state is None else state
        return self.state

    # action is tuple of size num_players representing joint action
    def step(self, action):
        rewards = [(PROFIT_PER_ITEM if a == CLEAN else 2 * PROFIT_PER_ITEM)- (
        TAX_BY_NUM_PLAYERS[self.num_players]
        if self.state == POLLUTED
        else 0)for a in action]
        cost = 2 * (self.num_players - sum(action)) / self.num_players
        potential = sum(rewards)
        self.state = POLLUTION_FREE if all([a == CLEAN for a in action]) else POLLUTED
        return self.state, rewards, cost, potential, False

class gridW(gym.Env):
    def __init__(self,N=4):
        self.N = N
        self.grid = np.zeros((N-1,N-1))
    def grid_to_number(self,x,y):
        return x*self.N+y
    def number_to_grid(self,z):
        x = z%self.N
        y = z//self.N
        return x,y
    def reset(self,state=None):
        if state==None:
            self.state = self.grid_to_number(0,0)
        else:
            self.state = state
        return self.state
    def step(self,a):
        x,y = self.number_to_grid(self.state)
        done = False
        if a==0: #upward movement
            x = x+1
            if x==self.N:
                x = self.N-1
        elif a==1: #Rightward-movement
            y = y+1
            if y==self.N:
                y=self.N-1
        elif a==2:  #Leftward movement
            y = y-1
            if y<0:
                y=0
        else:
            x = x-1
            if x<0:
                x=0
        self.state = self.grid_to_number(x,y)
        if self.state == self.grid_to_number(0,1):
            r=2
        elif self.state == self.grid_to_number(1,0):
            r=1
        elif self.state == self.grid_to_number(2,3):
            r=10
            done = True
        return self.state,r,done,False

class multi_gridW(gym.Env):
    def __init__(self,N=4,m=2,rew_ind=(2,3)):
        self.m = m
        self.grid_objs = [gridW(N) for _ in range(m)]
        self.h=None
    def reset(self):
        self.states=[]
        self.h=0
        for i in range(self.m):
            self.states.append(self.grid_objs[i].reset())
        return self.states
    def step(self,a_bar):
        if self.h!=None:
            raise ValueError("First reset the environment!")
        special_indices = [grid_objs[0].grid_to_number(rew_ind),0]
        states_m = {0:[],1:[]}
        cost = 0
        sum_r=0
        done = False
        if self.h==6:
            done = True
        self.h+=1
        for i in range(self.m):
            a_i = a_bar[i]
            state,r,done,_ = grid_objs[i].step(a_i)
            sum_r+=r
            states_m[i].append(state)
        if states_m[0][0] == states_m[1][0]:
            cost = 1
        return states_m,sum_r,cost,done

class congestion_game(gym.Env):
    def __init__(self,m=6):
        self.states = [0,1] #o is safe and 1 is unsafe
        self.m = m
        self.N = len(self.states)
        self.actions = [0,1,2,3] #corresponding to A,B,C and D
        self.weights = [np.linspace(0.5,0.95,4)/m,np.linspace(0.1,0.8,4)/m]
        #Transition is action conditional and model is unknown
        self.offset = (0.1 - 1e-8)/m
        self.m = m
    def reset(self):
        self.curr = np.random.choice([0,1],p=[0.5,0.5])
        self.h = 0
        return self.curr
    def step(self,a_bar):
        self.a_bar = a_bar
        a_choice=zip(self.actions,0)
        done = False
        for i in range(self.m):
            a_choice[int(a_bar[i])]+=1
        r_list = [0]*self.m
        for i in range(self.m):
            r_list[i] = self.weights[int(Self.curr)]*a_choice[int(a_bar[i])] - (self.curr==1)*self.offset
        cost = (self.curr==1)
        if self.curr==0:
            if a_choice.any() > self.m//2:
                self.curr=1
        else:
            if a_choice.any() <=self.m//4:
                self.curr = 0
        if self.h <= 3:
            done = False
        else:
            done = True
        self.h+=1
        return self.curr,r_list,cost,done