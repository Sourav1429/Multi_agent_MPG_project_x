import numpy as np
import torch
import numpy as np
from tqdm import tqdm
from itertools import product,combinations
from envs import (
    DemandResponseMarketEnv,
    PollutionTaxEnv,
    multi_gridW,
    congestion_game,
    POLLUTION_FREE,
    POLLUTED,
    CLEAN,
    PROFIT_PER_ITEM,
    TAX_BY_NUM_PLAYERS,
)
from scipy.optimize import minimize
import pickle


class Player:
    def __init__(self, env):
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.policy = (np.ones((self.num_states, self.num_actions) )/ self.num_actions)
        if isinstance(env, PollutionTaxEnv):
            self.policy[:, 0] = 0.7
            self.policy[:, 1] = 0.3
        elif isinstance(env, DemandResponseMarketEnv):
            self.policy[:, 0] = 1.0
            self.policy[:, 1:] = 0.0
    def get_action(self, state):
        return np.random.choice(self.num_actions,p=self.policy[state])
    def get_policy(self):
        return self.policy

class project_x_dec_tabular:
    def __init__(self,args):
        self.args = args
        self.individual_vf = {}
        if args.env_nm == "energy":
        self.env = DemandResponseMarketEnv(num_players=args.M,num_states=args.nS,num_actions=args.nA)

        elif args.env_nm == "pollution":
            self.env = PollutionTaxEnv(num_players=args.M,num_states=args.nS,num_actions=args.nA)
        else:
            raise ValueError(f"Unknown environment: {args.env_nm}")
        self.agent_list = self.agent_list = [Player(self.env)for _ in range(self.args.M)]
        self.Cf = []
    def policy_gradient(pi,rew):
        pass
    def policy_evaluator(self,pol,r,type=0):
        value_fun = {(s, i): 0 for s in range(self.args.nS) for i in range(len(self.args.m))}
        value_fun_cost = {s: 0 for s in range(self.args.nS)}
        potential_value = {s: 0 for s in range(self.args.nS)}
        for _ in range(self.args.n_samples):
            for state in range(self.args.n_samples):
                env.reset(state)
                for t in range(self.args.T):
                    joint_action = [p.get_action(state) for p in self.args.players]
                    _, rewards, cost, potential, _ = env.step(joint_action)
                    for i in range(self.args.m):
                        value_fun[state, i] += (gamma**t) * rewards[i]
                    value_fun_cost[state] += (gamma**t) * cost
                    potential_value[state] += (gamma**t) * potential
        value_fun.update((x, v / num_samples) for (x, v) in value_fun.items())
        value_fun_cost.update((x, v / num_samples) for (x, v) in value_fun_cost.items())
        potential_value.update((x, v / num_samples) for (x, v) in potential_value.items())
        return value_fun, value_fun_cost, potential_value
    def project_pol(self,pol,z=1):
        n_features = pol.shape[0]
        u = np.sort(pol)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(pol - theta, 0)
        return w
    def run_algo(self):
        Pi = [self.agent_list[i].policy for i in range(self.args.m)] # first breaking point since the agent's list should give me the current policy but where is the agent's list
        for t in range(self.args.T):
            get_vf = []
            g1=[]
            g2=[]
            cf = self.policy_evaluator(Pi,self.args.cost) # second breaking point how to give the cost in the args
            for i in range(self.args.m):
                vf = self.policy_evaluator(Pi[i],self.args.rew[i]) # third breaking point reward matrix is dependent on combined action so cannot give this directly
                if self.individual_vf.get(i,None)==None:
                    self.individual_vf[i] = [vf]
                else:
                    self.individual_vf[i].append(vf)
                g1.append(self.policy_gradient(Pi[i],self.args.rew[i])) # Did not define the policy_gradient ()
                g2.append(self.policy_gradient(Pi[i],self.args.cost)) # Did not define the policy_gradient ()
            g2_prime = g1-self.args.lamdba*self.sigmoid([self.args.b-cf]/self.args.tau)*g2
            Pi = self.project_pol(Pi + self.args.alpha*g2_prime)
            self.Cf.append(cf)
        self.individual_vf['cost'] = self.Cf
        data = pd.DataFrame(self.store_values)
        print(data.head())
        try:
            data.to_excel(self.args.save_file_nm,index=False) 
        except Exception:
            with open(self.args.pickle_save,"wb") as f: 
                pickle.dump(data,f)
            f.close()
        print("All files stored and saved")
        return Pi

        

