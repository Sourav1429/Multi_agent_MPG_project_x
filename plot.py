import pickle
import pandas as pd

file_nm = "/project/ag2682/sg2786/storage_data_2_agents_energy.pkl"

with open(file_nm,"rb") as f:
  data2 = pickle.load(f)
f.close()

import pandas as pd
data2 = pd.DataFrame(data2)
data2.head()

import matplotlib.pyplot as plt
potential = data2['potential'].values
Jc = data2['Cost']
Jc = Jc

plt.figure()
plt.plot(potential)
plt.xlabel('Iteration')
plt.ylabel('Potential-function')
plt.savefig('Multi_agent_MPG_potential_function_2_agents_energy.png')
#plt.show()


plt.figure()
plt.plot(Jc,label="Constraint-cost")
plt.axhline(y=12, color='black', linestyle='-',label='Safety threhold')
plt.axhspan(0,12,color='blue',alpha=0.15,label='Safe zone')
plt.axhspan(12,20,color='red',alpha=0.1,label='Unsafe zone')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Cost-function')
plt.savefig('Multi_agent_MPG_cost_function_2_agents_energy.png')
#plt.show()