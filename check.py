import argparse
import numpy as np

from multi_agent import project_x_tabular
from scipy.optimize import minimize
env_nm = "pollution"      
m = 2                      
nS = 2                    
nA = 2                    
init_pi = np.zeros((m, nS, nA))
for i in range(m):
    for s in range(nS):
        init_pi[i,s,0] = 0.7
        init_pi[i,s,1] = 0.3
parser = argparse.ArgumentParser(description="Project_x_tabular")
parser.add_argument("--env_nm",default=env_nm,type=str)
parser.add_argument("--T",default=20,type=int,help="Number of algorithm iterations")
parser.add_argument("--horizon", default=10, type=int) # evaluation horizon
parser.add_argument("--M",type=int,default=m,help="Number of players")
parser.add_argument("--gamma",type=float,default=0.9,help="Discount factor")
parser.add_argument("--mu",type=float,default=0.8,help="Proximal parameter")
parser.add_argument("--lambda",dest="lamdba",type=float,default=20,help="Lambda")
parser.add_argument("--eta",type=float,default=0.1,help="Policy update step size")
parser.add_argument("--b",type=float,default=12,help="baseline threshold limit")
parser.add_argument("--init_pi",default=init_pi,help="Initial policy")
args = parser.parse_args()
agent = project_x_tabular(args)
#phi,Jc = agent.evaluate_policy()
# print("My algo:",phi,"---",Jc)
pi = init_pi
phi_current, jc_current = agent.evaluate_policy(init_pi)

t0 = min(
    phi_current / agent.lam,
    agent.b - jc_current
)

print("\n===== SLSQP START =====")
print(f"Phi = {phi_current:.10f}")
print(f"Jc  = {jc_current:.10f}")
print(f"lambda = {agent.lam}")
print(f"b = {agent.b}")
print(f"t0 = {t0:.10f}")

print(
    "potential constraint =",
    phi_current / agent.lam - t0
)

print(
    "cost constraint =",
    agent.b - jc_current - t0
)

print("policy:")
print(pi)