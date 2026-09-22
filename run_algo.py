import argparse
import numpy as np

from multi_agent import project_x_tabular
from multi_agent_decentralised import project_x_dec_tabular
env_nm = "energy"      
m = 8                     
nS = 5                    
nA = 5                    
init_pi = np.zeros((m, nS, nA))
T = 120
excel_file_nm = f"storage_{m}_players_{nS}_states_tabular_{env_nm}.xlsx"
pickle_save = "storage.pkl"
if env_nm == "pollution":
    for i in range(m):
        for s in range(nS):
            init_pi[i,s,0] = 0.7
            init_pi[i,s,1] = 0.3
elif env_nm == "energy":
    for i in range(m):
        for s in range(nS):
            init_pi[i,s,0] = 1
print("Number of agents is:",m)
if choice==0: #0-----> centralised
    parser = argparse.ArgumentParser(description="Project_x_tabular")
    parser.add_argument("--env_nm",default=env_nm,type=str)
    parser.add_argument("--T",default=120,type=int,help="Number of algorithm iterations")
    parser.add_argument("--horizon", default=10, type=int) # evaluation horizon
    parser.add_argument("--M",type=int,default=m,help="Number of players")
    parser.add_argument("--gamma",type=float,default=0.9,help="Discount factor")
    parser.add_argument("--mu",type=float,default=0.8,help="Proximal parameter")
    parser.add_argument("--lambda",dest="lamdba",type=float,default=20,help="Lambda")
    parser.add_argument("--eta",type=float,default=0.1,help="Policy update step size")
    parser.add_argument("--b",type=float,default=16,help="baseline threshold limit")
    parser.add_argument("--init_pi",default=init_pi,help="Initial policy")
    args = parser.parse_args()
    agent = project_x_tabular(args)
    agent.run_algo_tabular()
else:
    parser =argsparse.ArgumentParser(description="Project_x_tabular_decentralised")
    parser.add_argument("--env-nm",default=env_nm,type=str)
    parser.add_argument("--T",default=T,type=int,help="Number of algorithm iterations")
    parser.add_argument("--horizon", default=10, type=int) # evaluation horizon
    parser.add_argument("--M",type=int,default=m,help="Number of players")
    parser.add_argument("--gamma",type=float,default=0.9,help="Discount factor")
    parser.add_argument("--mu",type=float,default=0.8,help="Proximal parameter")
    parser.add_argument("--lambda",dest="lamdba",type=float,default=20,help="Lambda")
    parser.add_argument("--eta",type=float,default=0.1,help="Policy update step size")
    parser.add_argument("--b",type=float,default=16,help="baseline threshold limit")
    parser.add_argument("--init_pi",default=init_pi,help="Initial policy")
    parser.add_argument("--save_file_nm",default=excel_file_nm,help="File to save")
    parser.add_argument("--pickle_save",default=pickle_save,help="Pickle file in emergency if the excel file is not saved")
    parser.add_argument("--alpha",default=1e-2,help="Learning rate")
    parser.add_argument("--lamdba",default=20,help="Lagrangian parameter")
    parser.add_argument("--tau",default=10,help="Temperature for softplus metric")
    parser.add_argument("--nS",default=nS, help="Number of states")
    parser.add_argument("--nA",default=nA,help="Number of actions")
    args = parser.parse_args()
    agent = project_x_dec_tabular(args)
    agent.run_algo()