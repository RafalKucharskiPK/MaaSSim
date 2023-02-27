#!/usr/bin/env python
# coding: utf-8

# 
# # Shared rides with pricing - Delft 
# 
#  ### Choice Function (Deterministic):
#  `pool_price.py`
#  * Pickup Distance: distance from driver initial position to the first pickup point
#  * Travel Distance: distance from driver's initial position to the drop off point of the last passenger
#  * Operating Cost: This include all the expenses
#  * Profit: Driver revenue to serve the request
#             
#  
#   ### KPI:
#    
#    * Profit of Individual driver
#    * Profit of all the drivers
#    * No.of rejected rides
#    * U - PAX (Utility) 
#   
#    ### TBD- Choice Function (Probablistic):
#   
#   * choice logic to be applied inside `pool_price.py` 
#   * P(R)= exp(beta * Profit_R)/ sum_all the rides( exp(beta * Profit_R)
#  
# 
# 

# -------------------------------------------------------------------------------------------------------
# 
# # Pricing and Driver Earnings for a Two-Sided Mobility Platform: A Case of Amsterdam, the Netherlands
# 
# or 
# 
# # The Effects of Profit-Based Pricing on Driver Earnings and Performance of Two-Sided Mobility Platforms
# 
# # Abstract  
# 
# In this paper, we investigate how the  pricing of ride-pooling affects driver earnings. We also examine how profit-based setting affects these performance indicators. To this end, we applied a matching algorithm  to the case of ride-pooling and give a choice set to the driver for the case of Amsterdam, the Netherlands. For our simulation, we utilize an agent-based simulator reproducing the transport systems for two-sided mobility platforms (like Uber and Lyft) and applied three state-of-the-art pricing strategies such as <strong>profit maximization</strong>,  <strong>solo ride-hailing</strong>, and <strong>nearest pickup ride-pooling</strong>. We find that the profit maximization pricing strategy outperforms the other and traveler utility can be further improved by $\%X$ while reducing the total cost to serve the pooled rides. While offering a discount for profit maximization travel time is significantly higher $\%X$  than for private rides. 
# 
# -------------------------------------------------------------------------------------------------------

# ## Mode of Simulation 
# 
# Three type of simulation 
# 
# <strong>1. Profit maximization</strong> 
# 
# <strong>2. Solo ride-hailing</strong>
# 
# <strong>3. Nearest pickup ride-pooling</strong>

# In[ ]:





# ## Load ExMAS and MaaSSim
# 

# In[1]:


import os, sys # add MaaSSim to path (not needed if MaaSSim is already in path)
module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from MaaSSim.utils import get_config, load_G, prep_supply_and_demand, generate_demand, generate_vehicles, initialize_df  # simulator
from MaaSSim.data_structures import structures as inData
from MaaSSim.simulators import simulate
from MaaSSim.visualizations import plot_veh
from MaaSSim.shared import prep_shared_rides
import logging
import matplotlib.pyplot as plt

import pandas as pd
import ExMAS


# ## Delft, Netherlands

# In[2]:


params = get_config('../../data/config/delft.json')  # load configuration

params.times.pickup_patience = 3600 # 1 hour of simulation
params.simTime = 4 # 6 minutes hour of simulation
params.nP = 400 # reuqests (and passengers)
params.nV = 20 # vehicles


# ## Parameters for ExMAS

# In[3]:


params.t0 = pd.Timestamp.now()
params.shareability.avg_speed = params.speeds.ride
params.shareability.shared_discount = 0.25
params.shareability.delay_value = 1
params.shareability.WtS = 1.3
params.shareability.price = 1.5 #eur/km
params.shareability.VoT = 0.0035 #eur/s
params.shareability.matching_obj = 'u_veh' #minimize VHT for vehicles
params.shareability.pax_delay = 0
params.shareability.horizon = 600
params.shareability.max_degree = 4
params.shareability.nP = params.nP
params.shareability.share = 1
params.shareability.without_matching = True
params.shareability.operating_cost = 0.5
params.shareability.comm_rate = 0.2

inData = load_G(inData, params)  # load network graph 

inData = generate_demand(inData, params, avg_speed = False)
inData.vehicles = generate_vehicles(inData,params.nV)
inData.vehicles.platform = inData.vehicles.apply(lambda x: 0, axis = 1)
inData.passengers.platforms = inData.passengers.apply(lambda x: [0], axis = 1)
inData.requests['platform'] = inData.requests.apply(lambda row: inData.passengers.loc[row.name].platforms[0], axis = 1) 
inData.platforms = initialize_df(inData.platforms)
inData.platforms.loc[0]=[1,'Uber',30]
params.shareability.share = 1
params.shareability.without_matching = True

 #
#prepare schedulesrf[(rf['indexes_orig'].map(len) > 1) & (rf['driver_revenue']==rf['driver_revenue'].max())].iloc[0]
# # Strategy 1: 
# # params.kpi = 1 (Profit Maximazation)
# 

# ### Profit Mazimization - Begin 

# In[4]:


inData = ExMAS.main(inData, params.shareability, plot=False) # create shareability graph (ExMAS) 


# In[5]:


inData = prep_shared_rides(inData, params.shareability) # prepare schedules


# In[6]:


inData.sblts.rides


# In[7]:


params.kpi = 1


# In[9]:


sim = simulate(params = params, inData = inData, logger_level = logging.CRITICAL) # simulate

sim.res[0].pax_exp


# In[10]:


sim.res[0].veh_exp


# In[11]:


sim.res[0].veh_exp['REVENUE'].to_list()


# In[12]:


import seaborn as sns
sns.set_style("whitegrid")
sim.res[0].veh_exp['Vehicles'] = sim.res[0].veh_exp.index

ax =sns.barplot(data=sim.res[0].veh_exp, x="Vehicles", y="REVENUE")
#for i in ax.containers:
    #ax.bar_label(i,)


# # Total Revenue of all the driver 

# In[13]:


sim.res[0].all_kpi # All driver revenue 


# In[ ]:





# # Strategy 2: 
# 
# # params.kpi = 2 (Solo ride-hailing) 
# 

# In[68]:


params.kpi = 2


# In[69]:


sim = simulate(params = params, inData = inData, logger_level = logging.WARNING) # simulate


# In[70]:


sim.res[0].veh_exp


# In[71]:


sim.res[0].veh_exp['REVENUE'].to_list()


# In[72]:


import seaborn as sns
sns.set_style("whitegrid")

sim.res[0].veh_exp['Vehicles'] = sim.res[0].veh_exp.index

ax =sns.barplot(data=sim.res[0].veh_exp, x="Vehicles", y="REVENUE")
#for i in ax.containers:
    #ax.bar_label(i,)


# # Total revenue of all the driver

# In[73]:


sim.res[0].all_kpi # All driver revenue 


# # Strategy 3: 
# # params.kpi = 3 (Nearest pickup ride-pooling)
# 

# In[74]:


params.kpi = 3


# In[75]:


sim = simulate(params = params, inData = inData, logger_level = logging.WARNING) # simulate


# In[76]:


sim.res[0].veh_exp


# In[77]:


sim.res[0].veh_exp['REVENUE'].to_list()


# In[78]:


import seaborn as sns

sns.set_style("whitegrid")

sim.res[0].veh_exp['Vehicles'] = sim.res[0].veh_exp.index

ax =sns.barplot(data=sim.res[0].veh_exp, x="Vehicles", y="REVENUE")

#ax.set(xlabel=None)
#for i in ax.containers:
    #ax.bar_label(i,)


# # Total revenue of all the driver 

# In[79]:


sim.res[0].all_kpi # All driver revenue 


# In[ ]:





# # All in one Simulation  

# In[ ]:


responses = []
avg_kpi = []
idle_time = []

for i in range(1, 4):
    params.kpi = i
    sim = simulate(params = params, inData = inData, logger_level = logging.CRITICAL) # simulate
    sim.res[0].veh_exp['Vehicles'] = sim.res[0].veh_exp.index
    sim.res[0].veh_exp['ds'] = f"{i}"
    
    responses.append(sim.res[0].veh_exp)
    
    vehicles = sim.res[0].veh_exp.loc[sim.res[0].veh_exp["nRIDES"] > 0]
    no_of_veh = len(vehicles)
    
    avg_kpi.append(sim.res[0].all_kpi/no_of_veh)
    idle_time.append(vehicles['IDLE'].sum()/no_of_veh)
    


# # Performance Parameters for Driver

# In[ ]:


import pandas as pd
index = pd.Index(['Revenue', 'Profit', 'Cost', 'Idle Time'])
driver_data = pd.DataFrame({"Profit Maximization":[], "Pooled Ride": [], "Private Ride": []})
driver_data.loc['Revenue'] = avg_kpi
driver_data.loc['Idle Time'] = idle_time
driver_data.loc['Cost'] = driver_data.loc['Revenue'].apply(lambda x: x*params.shareability.operating_cost)


# In[ ]:


driver_data


# In[ ]:


csv_data = driver_data.to_csv('D:/Development/GitHub-ProjectV2.0/MaaSSim/docs/tutorials/Results/nV20.csv')


# In[14]:


print('\nCSV String:\n', csv_data)


# In[ ]:




